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
from wav2vec2fasr import segment
from wav2vec2fasr import orthography as ort
from wav2vec2fasr.orthography import def_tok, load_config
from wav2vec2fasr.transcribe import get_logits, transcribe_segment, build_lm_decoder
ort_tokenizer = load_config()[0]

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

def return_alignments(trellis, path, transcript, separator="|"):
    """
    My version of merge_words that also ensures word alignments are flush to each other and to the end of the audio

    Args:
        trellis : the CTC trellis used to generate the path needed here for the time, made by get_trellis()
        path : the CTC path, generated by backtrack()
        transcript : the transcript to be aligned
        separator : the character in the transcript used to separate words
    Returns:
        The following tuple - (chars, words, char_alignments, word_alignments)
        chars : list of character segments
        words : list of word segments
        char_alignments : dict of char alignments
        word_alignments : dict of word alignments
    """
    segments = merge_repeats(path, transcript)
    audio_end = len(trellis)
    word_alignments = []
    char_alignments = []
    words = []
    word_idx = 0
    chars = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                if i2==len(segments): end = audio_end
                else: end = segments[i2].end
                words.append(Segment(word, segments[i1].start, end, score))
                word_alignments.append({"word": word, "start": int(segments[i1].start*20), "end": int(end*20), 
                                        "score": score, "word-idx": word_idx})
                word_idx += 1
            i1 = i2 + 1
            i2 = i1
        else:
            if i2 < len(segments)-1: 
                if segments[i2+1].label == separator:
                    chars.append(Segment(segments[i2].label, segments[i2].start, segments[i2+1].end-1, segments[i2].score))
                    char_alignments.append({"char": segments[i2].label, "start": int(segments[i2].start*20),
                                            "end": int((segments[i2+1].end)*20), "score": segments[i2].score,
                                            "word-idx": word_idx})
                else:
                    chars.append(segments[i2])
                    char_alignments.append({"char": segments[i2].label, "start": int(segments[i2].start*20),
                                            "end": int(segments[i2].end*20), "score": segments[i2].score, 
                                            "word-idx": word_idx})
            else:
                chars.append(Segment(segments[i2].label, segments[i2].start, audio_end, segments[i2].score))
                char_alignments.append({"char": segments[i2].label, "start": int(segments[i2].start*20),
                    "end": int(audio_end*20)-1, "score": segments[i2].score, "word-idx": word_idx})
            i2 += 1
    return(chars, words, char_alignments, word_alignments)

def align_audio(processor: Wav2Vec2Processor, 
                logits = None, 
                transcript : str = None, 
                model : AutoModelForCTC = None, 
                audio = None, 
                strides = (0,0),
                decoder = None) -> tuple:
    """
    Returns millisecond character and word alignments relative to start of audio, as well as transcript if requested
    Structure largely adapted from https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py
    
    Args:
        processor (Wav2Vec2Processor) : Required wav2vec2 processor used for processing the audio
        logits : Optional argument for loading pregenerated logits (if you have them from transcription already)
        transcript (str) : Optional argument for specifying the transcription (if it differs from the logits, for example)
        model (AutoModelForCTC) : Necessary if you do not provide the logits, wav2vec2 model for generating logits
        audio (str or Path or ndarray) : Either the path to an audio file or the audio as an ndarray
        strides (tuple) : Strides on either side of each segment to provide context for prediction
        decoder (ctc decoder) : kenlm language model decoder
    Returns:
        tuple[list, list, str] : character, word, and phrase alignments, respectively
    """
    #If logits are not provided, generate them using the model
    if logits == None: logits = get_logits(processor, model, audio, strides)
    #If transcript is not provided, also generate transcript
    if transcript == None: transcript = transcribe_segment(processor, logits, decoder=decoder)
    align_dictionary = {char: code for char,code in processor.tokenizer.get_vocab().items()}
    #print(logits)
    emission = logits[0].cpu().detach()
    #Replace spaces with vertical pipes symbolizing word boundaries/gaps
    clean_transcript = transcript.replace(' ', '|')
    #Combine multiple adjacent pipes into single pipe
    clean_transcript = re.sub('\|+', '|', clean_transcript)
    tokens = [align_dictionary[c] for c in clean_transcript if c in align_dictionary]
    blank_id = 0
    for char, code in align_dictionary.items():
        if char.lower() == '[pad]' or char.lower() == '<pad>':
            blank_id = code
    trellis = get_trellis(emission, tokens, blank_id)
    path = backtrack(trellis, emission, tokens, blank_id)
    try:
        chars, words, char_alignments, word_alignments = return_alignments(trellis, path, clean_transcript)
        return(char_alignments, word_alignments, transcript)
    except Exception: 
        return(None, None, transcript)
    

def transcribe_audio(audio_path : any, 
                    model_dir,
                    model = None,
                    processor = None,
                    lm_decoder = None,
                    chunking_method='rvad_chunk', 
                    src_path = None,
                    tier_list = None,
                    tier_key = None,
                    utt_tier_name="utterances",
                    word_tier_name="words", 
                    char_tier_name="phones", 
                    output_name=None,
                    output=".TextGrid",
                    eval=False) -> dict:
    """
    Function for (1) chunking a specified audio file by a specified method and
     (2) transcribing and aligning it using a given wav2vec 2.0 model
    
    Args:
        audio_path (str | pathlib.Path) : path to audio file (wav or mp3)
        model_dir (str | pathlib.Path) : path to wav2vec 2.0 model
        model (AutoModelForCTC) : wav2vec2.0 model (optional if model path provided)
        processor (Wav2Vec2Processor) : include your processor here if you've already loaded it
        lm_decoder (BeamSearchDecoderCTC or pathlib.Path) : either a kenlm beam search ctc decoder or a path to one
        chunking_method (str) : method used to chunk audio, check segment.chunk_audio for options
        src_ts : path to source transcription file (either Praat TextGrid or ELAN eaf) for comparison
        tier_list (list of str) : list of specific tiers to segment audio by and transcribe
        tier_key (str) : returns all tiers from src_transcription with the key as a substring of their name
        word_tier_name (str) : name of word tier for word alignments
        char_tier_name (Str) : name of character tier for character alignments
        output_name (str) : name for the resulting file, defaults to f"{src_path.stem}{ts_format}" if None
    Returns:
        dict of annotations, with the lists of tuples for keys 'prediction' (referencing phrases), 'words', and 'chars'
            each tuple contains a prediction entry with start time and end time in milliseconds and the prediction as a string
    Output:
        if output is not None, then returns either a TextGrid or eaf file
    """
    if model == None : model = AutoModelForCTC.from_pretrained(model_dir).to('cpu')
    if processor == None: processor = Wav2Vec2Processor.from_pretrained(model_dir)
    #If language model directory provided, build lm decoder
    if lm_decoder != None and type(lm_decoder) == type(Path()) : decoder, processor = build_lm_decoder(model_dir, lm_decoder, processor)
    elif type(lm_decoder) == "<class 'BeamSearchDecoderCTC'>" : decoder = lm_decoder
    else: decoder = None
    audio_path = Path(audio_path)
    if audio_path.exists(): lib_aud, sr = librosa.load(audio_path, sr=16000)
    chunks = segment.chunk_audio(lib_aud=lib_aud, path=audio_path, method=chunking_method, 
                                 src_path=src_path, tiers=tier_list, tier_key=tier_key)
    ts = pympi.Eaf()
    ts.add_linked_file(file_path=audio_path, mimetype=audio_path.suffix[1:])
    ts.remove_tier('default')
    ts.add_tier(utt_tier_name)
    ts.add_tier(word_tier_name)
    ts.add_tier(char_tier_name)
    annotations = {utt_tier_name : [], word_tier_name : [], char_tier_name : []}
    for chunk in chunks:
        pred_st, pred_end = chunk[0] + chunk[2][0], chunk[1] - chunk[2][1]
        #DEBUG: Print start of phrase, end of phrase
        #print(pred_st, pred_end, pred_end-pred_st)
        calign, walign, pred = align_audio(processor, model=model, audio=chunk[3], strides=chunk[2], decoder=decoder)
        ts.add_annotation(utt_tier_name, pred_st, pred_end, ort_tokenizer.revert(pred))
        annotations[utt_tier_name].append((pred_st, pred_end, ort_tokenizer.revert(pred)))
        if calign != None and walign != None:
            for word in walign:
                ts.add_annotation(word_tier_name, word['start']+chunk[0], word['end']+chunk[0], 
                ort_tokenizer.revert(word['word']))
                annotations[word_tier_name].append((word['start']+chunk[0], word['end']+chunk[0], ort_tokenizer.revert(word['word'])))
            for char in calign:
                ts.add_annotation(char_tier_name, char['start']+chunk[0], char['end']+chunk[0], 
                ort_tokenizer.revert(char['char']))
                annotations[char_tier_name].append((char['start']+chunk[0], char['end']+chunk[0], ort_tokenizer.revert(char['char'])))
    if output != None: 
        if output == '.TextGrid': ts = ts.to_textgrid()
        if output_name == None: output_name=f"{audio_path.stem}{output}"
        ts.to_file(output_name)
    return(annotations)

def transcribe_audio_dir(aud_dir : Path,
                         model_dir,
                         model = None,
                         processor = None,
                         lm_decoder = None,
                         chunking_method='rvad_chunk', 
                         src_dir : Path = None,
                         tier_list = None,
                         tier_key = None,
                         utt_tier_name="utterances",
                         word_tier_name="words", 
                         char_tier_name="phones", 
                         output=".TextGrid"):
    """
    Function for applying transcribe_audio to all audio files in a given directory
    Args:
        aud_dir (pathlib.Path) : path to directory containing audio file (wav or mp3)
        model_dir (str | pathlib.Path) : path to wav2vec 2.0 model
        model (AutoModelForCTC) : wav2vec2.0 model (optional if model path provided)
        processor (Wav2Vec2Processor) : include your processor here if you've already loaded it
        lm_decoder (BeamSearchDecoderCTC or pathlib.Path) : either a kenlm beam search ctc decoder or a path to one
        chunking_method (str) : method used to chunk audio, check segment.chunk_audio for options
        src_dir : path directory of source transcription filse (either Praat TextGrid or ELAN eaf) for comparison
        tier_list (list of str) : list of specific tiers to segment audio by and transcribe
        tier_key (str) : returns all tiers from src_transcription with the key as a substring of their name
        word_tier_name (str) : name of word tier for word alignments
        char_tier_name (Str) : name of character tier for character alignments
        output_name (str) : name for the resulting file, defaults to f"{src_path.stem}{ts_format}" if None
    Output:
        if output is not None, then returns either a TextGrid or eaf file
    """
    if model == None : model = AutoModelForCTC.from_pretrained(model_dir).to('cpu')
    if processor == None: processor = Wav2Vec2Processor.from_pretrained(model_dir)
    #If language model directory provided, build lm decoder
    if lm_decoder != None and type(lm_decoder) == type(Path()) : decoder, processor = build_lm_decoder(model_dir, lm_decoder, processor)
    elif type(lm_decoder) == "<class 'BeamSearchDecoderCTC'>" : decoder = lm_decoder
    else: decoder = None
    wavs = {path.stem : path for path in aud_dir.iterdir() if path.suffix in [".wav", ".mp3"]}
    paired = []
    if src_dir != None and chunking_method=="src_chunk":
        srcs = {path.stem : path for path in src_dir.iterdir() if path.suffix in [".eaf", ".TextGrid"]}
        transcribe_pairs = [(wavs[ts], srcs[ts]) for ts in srcs if ts in wavs]
        paired = [wavs[ts] for ts in wavs in srcs]
        for pair in transcribe_pairs:
            print("Transcribing", pair)
            transcribe_audio(pair[0], model_dir, model=model, processor=processor, lm_decoder=lm_decoder, 
                             chunking_method=chunking_method, src_path=pair[1], tier_list=tier_list, tier_key=tier_key,
                             utt_tier_name=utt_tier_name, word_tier_name=word_tier_name, char_tier_name=char_tier_name, 
                             output=output)
    for wav in wavs not in paired:
        print("Transcribing", wav)
        transcribe_audio(wav, model_dir, model=model, processor=processor, lm_decoder=lm_decoder, 
                            chunking_method=chunking_method, tier_list=tier_list, tier_key=tier_key,
                            utt_tier_name=utt_tier_name, word_tier_name=word_tier_name, char_tier_name=char_tier_name, 
                            output=output)
        

def generate_alignments_for_phrases(audio_path, 
                                    src_path, 
                                    model_dir, 
                                    src_tier = "prediction",
                                    word_tier="words", 
                                    char_tier="chars", 
                                    copy_existing=False, 
                                    output_name=None,
                                    lm_dir = None):
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
        lm_dir (str or pathlib.Path) : path to a kenlm model
    Outputs:
        Transcript file, either an EAF or TextGrid based on the src_file
    """
    model = AutoModelForCTC.from_pretrained(model_dir).to('cpu')
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    #If language model directory provided, build lm decoder
    if lm_dir != None: decoder, processor = build_lm_decoder(model_dir, lm_dir, processor)
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
        transcript = ort_tokenizer.apply(ort.remove_special_chars(ann[2]))
        aud_chunk = lib_aud[librosa.time_to_samples(ann[0]/1000, sr=sr): librosa.time_to_samples(ann[1]/1000, sr=sr)]
        calign, walign, tokenized_transcript = align_audio(processor, transcript=transcript, model=model, audio=aud_chunk,
                                                            decoder = decoder)
        #Add tokenized phrasal transcripts
        out_file.add_annotation(src_tier, ann[0], ann[1], ann[2])
        out_file.add_annotation(src_tier+"_tokenized", ann[0], ann[1], 
                                ort_tokenizer.revert(tokenized_transcript))
        #If character and word alignments were generated, add annotations for both
        if calign != None and walign != None:
            #Add new word alignments
            for word in walign:
                out_file.add_annotation(word_tier, word['start']+ann[0], word['end']+ann[0], 
                    ort_tokenizer.revert(word['word']))
            #Add new character alignment annotations
            for char in calign:
                out_file.add_annotation(char_tier, char['start']+ann[0], char['end']+ann[0], 
                    ort_tokenizer.revert(char['char']))
        if src_path.suffix == '.TextGrid': out_file = out_file.to_textgrid()
    if output_name==None: output_name = f"{src_path.stem}_realigned"
    out_file.to_file(f"{output_name}{src_path.suffix}")

def align_transcriptions(audio_path, 
                        src_path,
                        model_dir,
                        model = None,
                        processor = None,
                        lm_decoder = None,
                        tier_list = None,
                        utt_tier_name=" - utterances",
                        word_tier_name=" - words", 
                        char_tier_name=" - phones", 
                        copy_existing=False, 
                        output_name=None,
                        ts_format=".TextGrid"):

    """
    This function generates word and character alignments from an existing transcription file
    Args:
        audio_path : path to audio file (wav or mp3)
        src_path : path to source transcription file (either Praat TextGrid or ELAN eaf)
        model (AutoModelForCTC or Pathlib.Path) : wav2vec2.0 model or path to wav2vec 2.0 model
        processor (Wav2Vec2Processor) : include your processor here if you've already loaded it
        lm_decoder (BeamSearchDecoderCTC or pathlib.Path) : either a kenlm beam search ctc decoder or a path to one
        tier_list (list of str) : list of specific tiers to transcribe (if none, aligns all tiers)
        utt_tier_name (str) : name of utterance tier for utterance alignments (full name is base_tier+utterance_tier)
        word_tier_name (str) : name of word tier for word alignments (full name is base_tier+word_tier)
        char_tier_name (Str) : name of character tier for character alignments (full name is base_tier+char_tier)
        copy_existing (bool) : if true, makes output of function a copy of the src_file with the new tiers
            If false, creates entirely new transcript file with only utterance_tier, word_tier, and char_tier
        output_name (str) : name for the resulting file, defaults to f"{src_path.stem}_realigned" if None
        lm_dir (str or pathlib.Path) : path to a kenlm model
        ts_format (str) : .TextGrid or .eaf, output of alignments
    Outputs:
        Transcript file, either an EAF or TextGrid based on the src_file
    """
    if model == None : model = AutoModelForCTC.from_pretrained(model_dir).to('cpu')
    if processor == None: processor = Wav2Vec2Processor.from_pretrained(model_dir)
    #If language model directory provided, build lm decoder
    if lm_decoder != None and type(lm_decoder) == type(Path()) : decoder, processor = build_lm_decoder(model_dir, lm_decoder, processor)
    elif type(lm_decoder) == "<class 'BeamSearchDecoderCTC'>" : decoder = lm_decoder
    else: decoder = None
    audio_path = Path(audio_path)
    src_path = Path(src_path)
    if audio_path.exists(): lib_aud, sr = librosa.load(audio_path, sr=16000)
    if src_path.suffix == ".TextGrid" : src_file = pympi.TextGrid(src_path).to_eaf()
    elif src_path.suffix == ".eaf" : src_file = pympi.Eaf(src_path)
    if copy_existing: out_file = src_file
    else: out_file = pympi.Eaf()
    out_file.add_linked_file(file_path=audio_path, mimetype=audio_path.suffix[1:])
    out_file.remove_tier("default")
    #If tier list not provided, get all tiers from source
    if tier_list == None: tier_list = src_file.get_tier_names()
    for src_tier in tier_list:
        utt_tier = src_tier+utt_tier_name
        if not(copy_existing): out_file.add_tier(utt_tier)
        # Get annotation data from the source tier
        src_tier_annotations = src_file.get_annotation_data_for_tier(src_tier)
        word_tier, char_tier = src_tier + word_tier_name, src_tier + char_tier_name
        out_file.add_tier(word_tier)
        out_file.add_tier(char_tier)
        for ann in src_tier_annotations:
            # Each corrected transcript entry has to be retokenized using the current tokenization scheme
            transcript = ort_tokenizer.apply(ort.remove_special_chars(ann[2]))
            aud_chunk = lib_aud[librosa.time_to_samples(ann[0]/1000, sr=sr): librosa.time_to_samples(ann[1]/1000, sr=sr)]
            calign, walign, tokenized_transcript = align_audio(processor, transcript=transcript, model=model, audio=aud_chunk,
                                                                decoder = decoder)
            if not(copy_existing) : out_file.add_annotation(utt_tier, ann[0], ann[1], ann[2])
            #If character and word alignments were generated, add annotations for both
            if calign != None and walign != None:
                #Add new word alignments
                for word in walign:
                    out_file.add_annotation(word_tier, word['start']+ann[0], word['end']+ann[0], 
                        ort_tokenizer.revert(word['word']))
                #Add new character alignment annotations
                for char in calign:
                    out_file.add_annotation(char_tier, char['start']+ann[0], char['end']+ann[0], 
                        ort_tokenizer.revert(char['char']))
    if ts_format == '.TextGrid': out_file = out_file.to_textgrid()
    if output_name==None: output_name = f"{src_path.stem}"
    out_file.to_file(f"{output_name}{ts_format}")

def align_transcription_dirs(aud_dir : Path,
                             src_dir : Path,
                             model_dir : Path,
                             model=None,
                             processor = None,
                             lm_decoder = None,
                             tier_list = None,
                             utt_tier_name=" - utterances",
                             word_tier_name=" - words", 
                             char_tier_name=" - phones",
                             ts_format = ".TextGrid",
                             ):
    if model == None : model = AutoModelForCTC.from_pretrained(model_dir).to('cpu')
    if processor == None: processor = Wav2Vec2Processor.from_pretrained(model_dir)
    #If language model directory provided, build lm decoder
    if lm_decoder != None and type(lm_decoder) == type(Path()) : decoder, processor = build_lm_decoder(model_dir, lm_decoder, processor)
    elif type(lm_decoder) == "<class 'BeamSearchDecoderCTC'>" : decoder = lm_decoder
    else: decoder = None
    wavs = {path.stem : path for path in aud_dir.iterdir() if path.suffix == ".wav"}
    srcs = {path.stem : path for path in src_dir.iterdir() if path.suffix in [".eaf", ".TextGrid"]}
    align_paths = [(wavs[ts], srcs[ts]) for ts in srcs if ts in wavs]
    for pair in align_paths:
        print("Aligning", pair[0].stem)
        align_transcriptions(pair[0], pair[1], model_dir, model=model, processor=processor, lm_decoder=lm_decoder, 
                             tier_list=tier_list, utt_tier_name=utt_tier_name, word_tier_name=word_tier_name, 
                             char_tier_name=char_tier_name, ts_format=ts_format)
    
def correct_alignments(audio_path, old_doc, corrected_doc, model_dir, cor_tier = "prediction", 
                       word_tier="words", char_tier="chars", lm_dir = None):
    model = AutoModelForCTC.from_pretrained(model_dir).to('cpu')
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    #If language model directory provided, build lm decoder
    if lm_dir != None: decoder, processor = build_lm_decoder(model_dir, lm_dir, processor)
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
        print("Comparison document provided, only producing new alignment for changes to cor_tier")
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
                run = True
        else: run = True
        if run:
            print(dat[0], dat[1], dat[2])
            # Each corrected transcript entry has to be retokenized using the current tokenization scheme
            transcript = ort_tokenizer.apply(ort.remove_special_chars(dat[2]))
            if transcript != '':
                print("Aligning |"+transcript+"|")
                aud_chunk = lib_aud[librosa.time_to_samples(dat[0]/1000, sr=sr): librosa.time_to_samples(dat[1]/1000, sr=sr)]
                strides = (0, 0)
                #Get alignments for corrected transcript    
                calign, walign, trash = align_audio(processor, transcript=transcript, model=model, audio=aud_chunk,
                                                        strides=strides, decoder=decoder)
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
                                ort_tokenizer.revert(word['word']))
                    #DEBUG: If the corrected tier is the word tier, for debugging add a new_words tier
                    else:
                        if "new_words" not in ts.get_tier_names(): ts.add_tier("new_words")
                        for word in walign:
                            print((word['start']+dat[0], word['end']+dat[0], ort_tokenizer.revert(word['word'])))
                            ts.add_annotation("new_words", word['start']+dat[0], word['end']+dat[0], 
                                ort_tokenizer.revert(word['word']))
                    #Clear previous character annotations
                    if char_tier in ts.get_tier_names(): 
                        char_del = ts.get_annotation_data_between_times(char_tier, dat[0], dat[1])
                        [print("removing ", char) for char in char_del]
                        [ts.remove_annotation(char_tier, char[0]+1) for char in char_del]
                    else:
                        ts.add_tier(char_tier)
                    #Add new character alignment annotations
                    for char in calign:
                        print("adding", (char['start']+dat[0], char['end']+dat[0], ort_tokenizer.revert(char['char'])))
                        ts.add_annotation(char_tier, char['start']+dat[0]+strides[0], char['end']+dat[0]+strides[0], 
                            ort_tokenizer.revert(char['char']))
            else: print("Cannot align blank segment")
        if cor_path.suffix == '.TextGrid': ts = ts.to_textgrid()
    ts.to_file(f"{cor_path.stem}_ac{cor_path.suffix}")


if __name__ == "__main__":
    pass