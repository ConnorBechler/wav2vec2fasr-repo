"""
wav2vec2 for forced alignment
built from (1) and (2)
(1) https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html
(2) https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py
"""
from transformers import Wav2Vec2Processor, AutoModelForCTC, Wav2Vec2CTCTokenizer
import pympi
import librosa
import torch
from dataclasses import dataclass
import re
from pathlib import Path
from numpy import ndarray
from wav2vec2fasr import segment
from wav2vec2fasr import orthography as ort
from wav2vec2fasr.orthography import def_tok

def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        # failed
        return None
    return path[::-1]

# Merge the labels
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start

def merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments

def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words

def align_audio(processor: Wav2Vec2Processor, 
                logits = None, 
                transcript : str = None, 
                model : AutoModelForCTC =None, 
                audio = None, 
                strides = (0,0)) -> tuple:
    """
    Returns millisecond character and word alignments relative to start of audio, as well as transcript if requested
    Structure largely adapted from https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py
    
    Args:
        processor (Wav2Vec2Processor) : Required wav2vec2 processor used for processing the audio
        logits : Optional argument for loading pregenerated logits (if you have them from transcription already)
        transcript (str) : Optional argument for specifying the transcription (if it differs from the logits, for example)
        model (AutoModelForCTC) : Necessary if you do not provide the logits, wav2vec2 model for generating logits
        audio (str or Path or ndarray) : Either the path to an audio file or the audio as an ndarray
    
    Returns:
        tuple[list, list, str] : character, word, and phrase alignments, respectively
    """
    # If logits are not provided, generate them using the model
    if logits == None:
        # If audio is a path, load from path, otherwise interpret as ndarray
        if type(audio) == type("string") or type(audio) == type(Path("/")):
            lib_aud, sr = librosa.load(audio, sr=16000)
        else: 
            lib_aud = audio
        # Process audio and generate logits
        input_values = processor(lib_aud, return_tensors="pt", padding=True, sampling_rate=16000).input_values
        #TODO: Decide if the following padding function is necessary
        diff = 1200 - len(input_values[0])
        if  diff > 0 :
            pad = [-500 for x in range(round(diff /2))]
            input_values =  torch.tensor([pad + list(input_values[0]) + pad])
        logits = model(input_values.to('cpu')).logits
        # Line below slices logit tensor to only include predictions for window within strides
        logits = torch.tensor([logits[0][round(strides[0]/20): len(logits[0])-round(strides[1]/20)].detach().numpy()]) 
    #Normalize logits into log domain to avoid "numerical instability" 
    # (https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html#generate-frame-wise-label-probability)????
    logits = torch.log_softmax(logits, dim=-1)
    # If transcript is not provided, also generate transcript
    if transcript == None: 
        transcript = processor.batch_decode(torch.argmax(logits, dim=-1))[0]
    align_dictionary = {char: code for char,code in processor.tokenizer.get_vocab().items()}
    emission = logits[0].cpu().detach()
    # Replace spaces with vertical pipes symbolizing word boundaries/gaps
    clean_transcript = transcript.replace(' ', '|')
    # Combine multiple adjacent pipes into single pipe
    clean_transcript = re.sub('\|+', '|', clean_transcript)
    tokens = [align_dictionary[c] for c in clean_transcript]
    blank_id = 0
    for char, code in align_dictionary.items():
        if char.lower() == '[pad]' or char.lower() == '<pad>':
            blank_id = code
    trellis = get_trellis(emission, tokens, blank_id)
    path = backtrack(trellis, emission, tokens, blank_id)
    if path != None:
        char_segments = merge_repeats(path, clean_transcript)
        char_alignments = []
        word_alignments = []
        word_idx = 0
        word_start = None
        word_end = None
        word = ""
        for cdx, char in enumerate(clean_transcript):
            start, end, score = None, None, None
            if char != "|":
                char_seg = char_segments[cdx]
                start = int(char_seg.start*20) + strides[0]
                #Added the line below because of bizarre error with 
                if start < 0 : start = 1
                if word_start == None:
                    word_start = start
                end = int(char_seg.end*20) + strides[0]
                word_end = end
                score = round(char_seg.score, 3)
                word += char
                char_alignments.append(
                    {
                        "char": char,
                        "start": start,
                        "end": end,
                        "score": score,
                        "word-idx": word_idx,
                    }
                )
            if cdx == len(clean_transcript)-1 or char == "|":
                word_alignments.append(
                    {
                        "word" : word,
                        "start" : word_start,
                        "end": word_end,
                        "word-idx": word_idx,
                    }
                )
                word_start, word = None, ""
                word_idx += 1
        return(char_alignments, word_alignments, transcript)
    else: return(None, None, transcript)

def chunk_and_align(audio_path : any, 
                    model_dir : any, 
                    chunking_method='rvad_chunk', 
                    output='.eaf') -> dict:
    """
    Function for (1) chunking a specified audio file by a specified method and
     (2) transcribing and aligning it using a given wav2vec 2.0 model
    Args:
        audio_path (str | pathlib.Path) : path to wav or mp3 file
        model_dir (str | pathlib.Path) : path to a wav2vec 2.0 model
        chunking_method (str) : chunking method name from segment.chunk_audio function (check there for options)
        output (str) : either .TextGrid or .eaf for praat TextGrids or ELAN eaf files, respectively
            TODO: add .txt output
    Returns:
        dict of annotations, with the lists of tuples for keys 'prediction' (referencing phrases), 'words', and 'chars'
            each tuple contains a prediction entry with start time and end time in milliseconds and the prediction as a string
    Output:
        if output is not None, then returns either a TextGrid or eaf file
    """
    name = Path(audio_path).name[:-len(Path(audio_path).suffix)]
    model = AutoModelForCTC.from_pretrained(model_dir).to('cpu')
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    chunks = segment.chunk_audio(path=audio_path, method=chunking_method)
    ts = pympi.Eaf()
    ts.add_linked_file(file_path=audio_path, mimetype='wav')
    ts.remove_tier('default')
    ts.add_tier('prediction')
    ts.add_tier('words')
    ts.add_tier('chars')
    annotations = {'prediction' : [], 'words' : [], 'chars' : []}
    for chunk in chunks:
        pred_st, pred_end = chunk[0] + chunk[2][0], chunk[1] - chunk[2][1]
        #DEBUG: Print start of phrase, end of phrase
        #print(pred_st, pred_end, pred_end-pred_st)
        calign, walign, pred = align_audio(processor, model=model, audio=chunk[3], strides=chunk[2])
        ts.add_annotation('prediction', pred_st, pred_end, def_tok.revert(pred))
        annotations['prediction'].append((pred_st, pred_end, def_tok.revert(pred)))
        if calign != None and walign != None:
            for word in walign:
                ts.add_annotation('words', word['start']+chunk[0], word['end']+chunk[0], 
                def_tok.revert(word['word']))
                annotations['words'].append((word['start']+chunk[0], word['end']+chunk[0], def_tok.revert(word['word'])))
            for char in calign:
                ts.add_annotation('chars', char['start']+chunk[0], char['end']+chunk[0], 
                def_tok.revert(char['char']))
                annotations['chars'].append((char['start']+chunk[0], char['end']+chunk[0], def_tok.revert(char['char'])))
    if output != None: 
        if output == '.TextGrid': ts = ts.to_textgrid()
        ts.to_file(name+"_preds"+output)
    return(annotations)

def generate_alignments_for_phrases(audio_path, src_path, model_dir, src_tier = "prediction",
                            word_tier="words", char_tier="chars", copy_existing=False, output_name=None):
    """
    This function generates word and character transcriptions from an existing transcription file
    Args:
        audio_path : path to audio file (wav or mp3)
        src_path : path to source transcription file (either Praat TextGrid or ELAN eaf)
        model_dir : path to wav2vec 2.0 model
        src_tier (str) : source tier for phrasal transcriptions from source file
        word_tier (str) : name of tier to clear and add word alignments
        char_tier (Str) : name of tier to clear and add character alignments
        copy_existing (bool) : if true, makes output of function a copy of the src_file with the new tiers
            If false, creates entirely new transcript file with only src_tier, word_tier, and char_tier
        output_name (str) : name for the resulting file, defaults to f"{src_path.stem}_realigned" if None
    Outputs:
        Transcript file, either an EAF or TextGrid based on the src_file
    """
    model = AutoModelForCTC.from_pretrained(model_dir).to('cpu')
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    audio_path = Path(audio_path)
    src_path = Path(src_path)
    if audio_path.exists(): lib_aud, sr = librosa.load(audio_path, sr=16000)
    if src_path.suffix == ".TextGrid" : src_file = pympi.TextGrid(src_path).to_eaf()
    elif src_path.suffix == ".eaf" : src_file = pympi.Eaf(src_path)
    if copy_existing: out_file = src_file
    else: 
        out_file = pympi.Eaf()
        out_file.add_tier(src_tier)
    out_file.add_tier(src_tier+"_tokenized")
    out_file.add_linked_file(file_path=audio_path, mimetype=audio_path.suffix[1:])
    # Get annotation data from the source tier and remove tiers being replaced
    src_tier_annotations = src_file.get_annotation_data_for_tier(src_tier)
    if word_tier in out_file.get_tier_names(): 
        out_file.remove_all_annotations_from_tier(word_tier)
    else: out_file.add_tier(word_tier)
    if char_tier in out_file.get_tier_names():
        out_file.remove_all_annotations_from_tier(char_tier)
    else: out_file.add_tier(char_tier)
    for ann in src_tier_annotations:
        # Each corrected transcript entry has to be retokenized using the current tokenization scheme
        transcript = def_tok.apply(ort.remove_special_chars(ann[2]))
        aud_chunk = lib_aud[librosa.time_to_samples(ann[0]/1000, sr=sr): librosa.time_to_samples(ann[1]/1000, sr=sr)]
        calign, walign, tokenized_transcript = align_audio(processor, transcript=transcript, model=model, audio=aud_chunk)
        #Add tokenized phrasal transcripts
        out_file.add_annotation(src_tier, ann[0], ann[1], ann[2])
        out_file.add_annotation(src_tier+"_tokenized", ann[0], ann[1], 
                                def_tok.revert(tokenized_transcript))
        #If character and word alignments were generated, add annotations for both
        if calign != None and walign != None:
            #Add new word alignments
            for word in walign:
                out_file.add_annotation(word_tier, word['start']+ann[0], word['end']+ann[0], 
                    def_tok.revert(word['word']))
            #Add new character alignment annotations
            for char in calign:
                out_file.add_annotation(char_tier, char['start']+ann[0], char['end']+ann[0], 
                    def_tok.revert(char['char']))
        if src_path.suffix == '.TextGrid': out_file = out_file.to_textgrid()
    if output_name==None: output_name = f"{src_path.stem}_realigned"
    out_file.to_file(f"{output_name}{src_path.suffix}")

def correct_alignments(audio_path, old_doc, corrected_doc, model_dir, cor_tier = "prediction", 
                       word_tier="words", char_tier="chars"):
    model = AutoModelForCTC.from_pretrained(model_dir).to('cpu')
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    audio_path = Path(audio_path)
    if audio_path.exists(): lib_aud, sr = librosa.load(audio_path, sr=16000)
    audio_length = librosa.get_duration(y=lib_aud, sr=sr)*1000
    # Check and load corrected transcription as appropriate pympi object
    cor_path = Path(corrected_doc)
    if cor_path.suffix == ".TextGrid" : ts = pympi.TextGrid(cor_path).to_eaf()
    elif cor_path.suffix == ".eaf" : ts = pympi.Eaf(cor_path)
    if old_doc != None:
        old_path = Path(old_doc)
        if old_path.suffix == ".TextGrid" : tso = pympi.TextGrid(old_path).to_eaf()
        elif old_path.suffix == ".eaf" : tso = pympi.Eaf(old_path)
        old_an_dat = tso.get_annotation_data_for_tier(cor_tier)
    ts.add_linked_file(file_path=audio_path, mimetype=audio_path.suffix[1:])
    # Get annotation data for corrected tier and remove tiers being replaced
    cor_an_dat = ts.get_annotation_data_for_tier(cor_tier)
    if old_doc != None:
        print("Comparsion document provided, only producing new alignment for changes to cor_tier")
        if len(cor_an_dat) != len(old_an_dat) : raise Exception("Different number of target tier annotations, not comparable")
    else:
        print("Comparison document not provided, updating all sub-alignments of cor_tier")
    # Iterate through corrected transcription file, creating new alignments based on new transcription
    last_index = len(cor_an_dat)-1
    for d, dat in enumerate(cor_an_dat):
        run = False
        if old_doc != None:
            old_an = tso.get_annotation_data_between_times(cor_tier, dat[0], dat[1])
            if old_an[-1] != dat:
                old_diff = old_an[-1][1]-old_an[-1][0]
                new_diff = dat[1]-dat[0]
                st_diff = new_diff - old_diff
                print(d, dat, old_an[-1], end_diff)
                run = True
        else: run = True
        if run:
            print(dat[0], dat[1], dat[2])
            # Each corrected transcript entry has to be retokenized using the current tokenization scheme
            transcript = def_tok.apply(ort.remove_special_chars(dat[2]))
            if transcript != '':
                print("Aligning |"+transcript+"|")
                if cor_tier != word_tier:
                    #Get audio chunk
                    aud_chunk = lib_aud[librosa.time_to_samples(dat[0]/1000, sr=sr): librosa.time_to_samples(dat[1]/1000, sr=sr)]
                    strides = (0, 0)
                else:
                    #TODO: Implement corrected word alignment, probably by taking either a set window of audio around
                    # the word, or by taking the three words surrounding
                    #Get audio chunk
                    strides = (0, 0)
                    if d == 0 : strides = ((dat[0]-strides[0])*((dat[0]-strides[0])>0) + ((dat[0]-1)*((dat[0]-strides[0])<=0)), strides[1])
                    elif d == last_index: strides = (strides[0], (dat[1]+strides[1])*((dat[1]+strides[1])<audio_length) + ((audio_length-(dat[1]+1))*((dat[1]+strides[1])>=audio_length)))
                    aud_chunk = lib_aud[librosa.time_to_samples((dat[0] - strides[0])/1000 , sr=sr): 
                                        librosa.time_to_samples((dat[1] + strides[1])/1000, sr=sr)]
                #Get alignments for corrected transcript    
                calign, walign, trash = align_audio(processor, transcript=transcript, model=model, audio=aud_chunk,
                                                        strides=strides)
                #If character and word alignments were generated, add annotations for both
                if calign != None and walign != None:
                    #Check if corrected tier is the word tier; if not, clear word annotations from slot and add new annotations
                    if cor_tier != word_tier:
                        #Clear previous word annotations
                        if word_tier in ts.get_tier_names(): 
                            wan_del = ts.get_annotation_data_between_times(word_tier, dat[0], dat[1])
                            [ts.remove_annotation(word_tier, wan[0]+1) for wan in wan_del]
                        else: ts.add_tier(word_tier)
                        #Add new alignments
                        for word in walign:
                            ts.add_annotation(word_tier, word['start']+dat[0], word['end']+dat[0], 
                                def_tok.revert(word['word']))
                    #DEBUG: If the corrected tier is the word tier, for debugging add a new_words tier
                    else:
                        if "new_words" not in ts.get_tier_names(): ts.add_tier("new_words")
                        for word in walign:
                            print((word['start']+dat[0], word['end']+dat[0], def_tok.revert(word['word'])))
                            ts.add_annotation("new_words", word['start']+dat[0], word['end']+dat[0], 
                                def_tok.revert(word['word']))
                    #Clear previous character annotations
                    if char_tier in ts.get_tier_names(): 
                        char_del = ts.get_annotation_data_between_times(char_tier, dat[0], dat[1])
                        [print("removing ", char) for char in char_del]
                        [ts.remove_annotation(char_tier, char[0]+1) for char in char_del]
                    else:
                        ts.add_tier(char_tier)
                    #Add new character alignment annotations
                    for char in calign:
                        print("adding", (char['start']+dat[0], char['end']+dat[0], def_tok.revert(char['char'])))
                        ts.add_annotation(char_tier, char['start']+dat[0]+strides[0], char['end']+dat[0]+strides[0], 
                            def_tok.revert(char['char']))
            else: print("Cannot align blank segment")
        if cor_path.suffix == '.TextGrid': ts = ts.to_textgrid()
    ts.to_file(f"{cor_path.stem}_ac{cor_path.suffix}")


if __name__ == "__main__":
    pass