"""
wav2vec2 for forced alignment
built from (1) and (2)
(1) https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html
(2) https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py
"""
from transformers import Wav2Vec2Processor, AutoModelForCTC, Wav2Vec2CTCTokenizer
import pympi
import librosa
import segment
import torch
from dataclasses import dataclass
import re
from pathlib import Path
import orthography as ort

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
                return_transcript : bool = False) -> tuple[list, list, str]:
    """
    Returns millisecond character and word alignments relative to start of audio, as well as transcript if requested
    Structure largely adapted from https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py
    
    Args:
        processor (Wav2Vec2Processor) : Required wav2vec2 processor used for processing the audio
        logits : Optional argument for loading pregenerated logits (if you have them from transcription already)
        transcript (str) : Optional argument for specifying the transcription (if it differs from the logits, for example)
        model (AutoModelForCTC) : Necessary if you do not provide the logits, wav2vec2 model for generating logits
        audio (str or Path or ndarray) : Either the path to an audio file or the audio as an ndarray
        return_transcript (bool) : Set to true if you want the returned tuple to output a transcript
    
    Returns:
        tuple[list, list, str] : 
    """
    # If logits are not provided, generate them using the model
    if logits == None:
        # If audio is a path, load from path, otherwise interpret as ndarray
        if type(audio) == type(str()) or type(audio) == type(Path()):
            lib_aud, sr = librosa.load(audio, sr=16000)
        else: 
            lib_aud = audio
        # Process audio and generate logits
        input_values = processor(lib_aud, return_tensors="pt", padding=True, sampling_rate=16000).input_values
        logits = model(input_values.to('cpu')).logits
    # If transcript is not provided, also generate transcript
    if transcript == None : 
        transcript = processor.batch_decode(torch.argmax(logits, dim=-1))[0]
    align_dictionary = {char.lower(): code for char,code in processor.tokenizer.get_vocab().items()}
    emission = logits[0].cpu().detach()
    # Replace spaces with vertical pipes symbolizing word boundaries/gaps
    clean_transcript = transcript.replace(' ', '|').lower()
    # Combine multiple adjacent pipes into single pipe
    clean_transcript = re.sub('\|+', '|', clean_transcript)
    tokens = [align_dictionary[c] for c in clean_transcript]
    blank_id = 0
    for char, code in align_dictionary.items():
        if char == '[pad]' or char == '<pad>':
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
                start = int(char_seg.start*20)#round(char_seg.start * ratio, 3)
                if word_start == None:
                    word_start = start
                end = int(char_seg.end*20)#round(char_seg.end * ratio, 3)
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
        if return_transcript: return(char_alignments, word_alignments, transcript)
        else: return(char_alignments, word_alignments, "")
    else: return(None, None, transcript)

def chunk_and_align(audio_path, model_dir, chunking_method='rvad_chunk', output='.eaf',
                    phon_comb=False, tone_comb=False):
    """
    Function for (1) chunking a specified audio file by a specified method and
     (2) transcribing and aligning it using a given wav2vec 2.0 model
    """
    name = Path(audio_path).name
    model = AutoModelForCTC.from_pretrained(model_dir).to('cpu')
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    chunks = segment.chunk_audio(path=audio_path, method=chunking_method)
    ts = pympi.Eaf()
    ts.add_linked_file(file_path=audio_path, mimetype='wav')
    ts.remove_tier('default')
    ts.add_tier('prediction')
    ts.add_tier('words')
    ts.add_tier('chars')
    for chunk in chunks:
        print(chunk[0], chunk[1])
        calign, walign, pred = align_audio(processor, model=model, audio=chunk[3], return_transcript=True)
        ts.add_annotation('prediction', chunk[0], chunk[1], pred)
        if calign != None and walign != None:
            for word in walign:
                ts.add_annotation('words', word['start']+chunk[0], word['end']+chunk[0], word['word'])
            for char in calign:
                ts.add_annotation('chars', char['start']+chunk[0], char['end']+chunk[0], char['char'])
    if output == '.TextGrid': ts = ts.to_textgrid()
    ts.to_file(name+output)

def correct_alignments(audio_path, corrected_doc, model_dir, cor_tier = "prediction", 
                       word_tier="words", char_tier="chars"):
    model = AutoModelForCTC.from_pretrained(model_dir).to('cpu')
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    if Path(audio_path).is_file(): lib_aud, sr = librosa.load(audio, sr=16000)
    # Check and load corrected transcription as appropriate pympi object
    cor_path = Path(corrected_doc)
    if cor_path.suffix == ".TextGrid" : ts = pympi.TextGrid(cor_path)
    elif cor_path.suffix == ".eaf" : ts = pympi.Eaf(cor_path)
    # Convert to Eaf and relink audio if audio link is broken
    ts = ts.to_eaf()
    ts.add_linked_file(file_path=audio_path, mimetype='wav')
    # Get annotation data for corrected tier and remove tiers being replaced
    cor_an_dat = ts.get_annotation_data_for_tier(cor_tier)
    if cor_tier != word_tier : 
        if word_tier in ts.get_tier_names():
            ts.remove_all_annotations_from_tier(word_tier)
        else: ts.add_tier(word_tier)
    if char_tier in ts.get_tier_names():
        ts.remove_all_annotations_from_tier(char_tier)
    else: ts.add_tier(char_tier)
    # Iterate through corrected transcript, creating new alignments based on new transcription
    for dat in cor_an_dat:
        print(dat[0], dat[1], dat[2])
        aud_chunk = lib_aud[librosa.time_to_samples(dat[0]/1000, sr=sr): librosa.time_to_samples(dat[1]/1000, sr=sr)]
        calign, walign = align_audio(processor, transcript=dat[2], model=model, audio=aud_chunk)
        if calign != None and walign != None:
            if cor_tier != word_tier:
                for word in walign:
                    ts.add_annotation(word_tier, word['start']+dat[0], word['end']+dat[0], word['word'])
            for char in calign:
                ts.add_annotation(char_tier, char['start']+dat[0], char['end']+dat[0], char['char'])
    if cor_path.suffix == '.TextGrid': ts = ts.to_textgrid()
    ts.to_file(f"{cor_path.name}_ac{cor_path.suffix}")


if __name__ == "__main__":
    
    model_dir = "../models/model_6-8-23_xlsr53_nt_nh/"
    audio = "../wav-eaf-meta/wq10_011.wav"
    #chunk_and_align(audio, model_dir, output='.TextGrid')
    correct_alignments(audio, "../wq10_011_nt_nh_cor.TextGrid", model_dir)
    
    
    #align_audio(processor, model=model, audio=audio)