import librosa
import pathlib
import pympi
#import soundfile
from pydub import AudioSegment, silence
import numpy
from math import ceil
import os

#from datasets import Dataset
#from pandas import DataFrame

import re
from jiwer import wer, cer

#IF USING NEWEST VERSION OF TRANSFORMERS (I use 4.11.3 instead): import Wav2Vec2ProcessorWithLM
from transformers import Wav2Vec2Processor, AutoModelForCTC, Wav2Vec2CTCTokenizer#, Wav2Vec2FeatureExtractor,  Wav2Vec2ForCTC, TrainingArguments, Trainer
#Testing pipeline stuff
from transformers import pipeline
import time
import torch
from pyctcdecode import build_ctcdecoder

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
warnings.simplefilter("ignore")

from copy import deepcopy

chars_to_ignore_regex = '[\,\?\.\!\;\:\"\“\%\‘\”\�\。\n\(\/\！\)\）\，]'
tone_regex = '[\¹\²\³\⁴\⁵\-]'
nontone_regex = '[^\¹\²\³\⁴\⁵ \-]'
diacritics = "ʲʷ ʰʷ ̥ ʰ ʲ ʰ ̃ ʷ".split(" ")
trips = ['sʰʷ', 'ʈʰʷ', 'ʂʰʷ', 'tʰʷ', 'qʰʷ', 'nʲʷ', 'kʰʷ', 'lʲʷ', 'ɕʰʷ', 'tʲʷ']
doubs = ['ɕʰ', 'n̥', 'qʷ', 'ɬʷ', 'qʰ', 'xʲ', 'xʷ', 'ɨ̃', 'ʈʷ', 'ʈʰ', 'ŋʷ', 
         'ʑʷ', 'mʲ', 'dʷ', 'ĩ', 'pʰ', 'ɕʷ', 'tʷ', 'rʷ', 'lʲ', 'ɡʷ', 'bʲ', 
         'pʲ', 'tʲ', 'zʷ', 'ɬʲ', 'ʐʷ', 'dʲ', 'ɑ̃', 'lʷ', 'sʷ', 'ə̃', 'kʷ', 
         'æ̃', 'ɖʷ', 'm̥', 'kʰ', 'ʂʷ', 'õ', 'ʂʰ', 'sʰ', 'r̥', 'nʲ', 'tʰ', 
         'jʷ', "õ", "ĩ"]
rep_trips = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
rep_doubs = "ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞⓟⓠⓡⓢⓣⓤⓥⓦⓧⓨⓩⒶⒷⒸⒹⒺⒻⒼⒽⒾⒿⓀⓁⓂⓃⓄⓅⓆⓇⓈⓉⓊⓋⓌⓍⓎⓏ"
tone_chars = "¹ ² ³ ⁵".split(" ")
tones = ["²¹", "²²", "³²", "³⁵", "⁵⁵", "⁵²", "⁵¹"]
rep_tones = "1234567890"

def phone_revert(text):
    for x in range(len(trips)):
        text = re.sub(rep_trips[x], trips[x], text)
    for x in range(len(doubs)):
        text = re.sub(rep_doubs[x], doubs[x], text)
    return text
    
def tone_revert(text):
    for x in range(len(tones)):
        text = re.sub(rep_tones[x], tones[x], text)
    return text

def get_dominant_tier(eaf, tier_type='phrase'):
  candidates = eaf.get_tier_ids_for_linguistic_type(tier_type)
  tiers_w_anns = [(tier, eaf.get_annotation_data_for_tier(tier)) for tier in candidates]
  longest_tier = [(None, [None])]
  for item in tiers_w_anns:
    if len(item[1]) > len(longest_tier[-1][1]): longest_tier.append(item)
  return(longest_tier[-1])

def chunk_audio_by_eaf_into_data(filename, path, aud_ext=".mp3"):
    """Function for chunking an audio file by eaf time stamp values from a given annotation tier into a list of numpy arrays"""
    if pathlib.Path(path+filename+aud_ext).is_file() and pathlib.Path(f"{path}{filename}.eaf").is_file():
        audio, sr = librosa.load(path+filename+aud_ext, sr=16000)
        eaf = pympi.Elan.Eaf(f"{path}{filename}.eaf")
        tar_tier, an_dat = get_dominant_tier(eaf)
        data = []
        for x in range(len(an_dat)):
            start = librosa.time_to_samples(an_dat[x][0]/1000, sr=sr)
            end = librosa.time_to_samples(an_dat[x][1]/1000, sr=sr)
            data.append({'from_file' : filename, 'segment' : x, 'transcript' : an_dat[x][2], 
                'audio' : {'array': audio[start:end], 'sampling_rate' : sr}})
        print(f"{filename} chunked successfully")
        return(data)
    else: 
        print(f"Audio or eaf files not found for {filename}, chunking not possible")

def stride_chunk(max_chunk, stride, length):
    """
    Simple chunking using aribtary divisions based on max chunk length and striding
    """
    num_chunks = ceil((length*1000)/max_chunk)
    nchunks = [[0, max_chunk+stride, (0, stride)]]
    nchunks += [[x*max_chunk-stride, (x+1)*max_chunk+stride, (stride, stride)] for x in range(1, num_chunks-1)]
    nchunks += [[max_chunk*(num_chunks-1), round(length*1000), (stride, 0)]]
    print(nchunks)
    return(nchunks)

def silence_stride_chunk(fullpath, aud_ext, max_chunk, min_chunk, stride, min_sil):
    """
    Slightly more complex chunking using pydub's silence function once and then subdividing chunks that are 
    too long using arbitrary divisions and striding
    """
    aud = AudioSegment.from_file(fullpath, format=aud_ext[1:])
    chunks = silence.detect_nonsilent(aud, min_silence_len=min_sil, silence_thresh=-35)
    nchunks = []
    for chunk in chunks:
        start, stop = chunk[0], chunk[1]
        diff = stop-start
        if diff > max_chunk:
            num_chunks = ceil(diff/max_chunk)
            step = round(diff/num_chunks)
            nchunks.append([start, start+step+stride, (0, stride)])
            for x in range(1, num_chunks):
                nchunks.append([(start+x*step)-stride, (start+(x+1)*step)+stride, (stride, stride)])
            nchunks.append([(start+(num_chunks)*step)-stride, stop, (stride, 0)])
        elif diff > min_chunk:
            nchunks.append([start, stop, (0, 0)])
    return(nchunks)

def og_silence_chunk(filename, path, aud_ext=".mp3", min_sil=1000, min_chunk = 100, max_chunk=10000):
    """
    Uses pydub detect non-silent to chunk audio by silences into speech segments.
    Crude and slow, but functional
    """
    aud = AudioSegment.from_file(path+filename+aud_ext, format=aud_ext[1:])
    chunks = silence.detect_nonsilent(aud, min_silence_len=min_sil, silence_thresh=-35)
    nchunks = []
    for chunk in chunks:
        start, stop = chunk[0], chunk[1]
        diff = stop-start
        if start > 100: start -= 100
        if diff >= min_chunk:
            if diff >= max_chunk:
                solved = False
                for x in range(2, 11):
                    tch = silence.detect_nonsilent(aud[chunk[0]:chunk[1]], min_silence_len=round(min_sil/x), silence_thresh=-35)
                    if len([y for y in tch if y[1]-y[0] > max_chunk]) == 0: 
                        print("solved at", round(min_sil/x))
                        nchunks += [[y[0]+start, y[1]+start] for y in tch]
                        solved=True
                        break
                if not(solved): 
                    print("Couldn't solve using shorter minimum silence lengths, increasing silence threshold instead")
                    tch = silence.detect_nonsilent(aud[chunk[0]:chunk[1]], min_silence_len=round(min_sil/2), silence_thresh=-35)
                    for y in tch:
                        ydiff = y[1]-y[0]
                        if ydiff > max_chunk:
                            print(ydiff)
                            for x in range(1, 6):
                                ntch = silence.detect_nonsilent(aud[start+y[0]:start+y[1]], min_silence_len=round(min_sil/2), silence_thresh=-35+x)
                                if len([z for z in ntch if z[1]-z[0] > max_chunk]) == 0: 
                                    print("solved at silence thresh of", -35+x)
                                    nchunks += [[z[0]+y[0]+start, z[1]+y[0]+start] for z in ntch]
                                    break
                                else:
                                    if x==5:
                                        print("Couldn't divide automatically, instead just chopping up into windows of arbitrary length")
                                        divs = round(ydiff/10000)
                                        step = round(ydiff/divs)
                                        for x in range(divs-1):
                                            nchunks.append([y[0]+start+step*x, y[0]+start+step*(x+1)-1])
                                        nchunks.append([y[0]+start+step*(divs-1), y[1]+start])
                                    pass
                        else: 
                            nchunks += [[y[0]+start, y[1]+start]]
            else: nchunks.append([start, stop])
    chunks = [chunk for chunk in nchunks if chunk[1]-chunk[0] > min_chunk]
    return(chunks)

class prediction:
    def __init__(self, logit, char, start, end):
        self.logit = logit
        self.char = char
        self.out_char = phone_revert(tone_revert(self.char))
        self.start = start
        self.end = end
    
    def update(self, newchar):
        if newchar != self.char:
            self.char = newchar
            self.out_char = phone_revert(tone_revert(self.char))

def merge_breaks(predlst):
    w_predlst = deepcopy(predlst)
    n_predlst = [w_predlst[0]]
    for pred in w_predlst[1:]:
        if pred.char == "|" and n_predlst[-1].char == "|":
            n_predlst[-1].end = pred.end
        else:
            n_predlst.append(pred)
    return(n_predlst)

def merge_pads(predlst):
    w_predlst = deepcopy(predlst)
    n_predlst = [w_predlst[0]]
    np_st = None
    for p in range(1, len(w_predlst[1:])):
        if w_predlst[p].char == "[PAD]" and n_predlst[-1].char != "[PAD]":
            n_predlst[-1].end = w_predlst[p].end
        else:
            #One possible way of removing pads from the beginning, although it leads to some counterintuitive results
            #if w_predlst[p].char != "[PAD]" and np_st == None:
            #    np_st = p
            n_predlst.append(w_predlst[p])
    return(n_predlst[np_st:])

def process_pads(predlst, processor):
    w_predlst = deepcopy(predlst)
    last_np_ind, next_np_ind = 0, 0
    last_wb = 0
    for x in range(len(predlst)):
        if predlst[x].char == "|":
            next_wb = x
            break
    for x in range(len(predlst)):
        if predlst[x].char == "[PAD]":
            #Search for next non-pad prediction if old or not yet made
            if next_np_ind <= x or next_np_ind == last_np_ind:
                prob_next = 0
                for y in range(x, len(predlst)):
                    if predlst[y].char != "[PAD]" and predlst[y].char != "|": 
                        next_np_ind = y
                        break
                    elif predlst[y].char == "|":
                        next_wb = y
            prob_last = predlst[x].logit[processor.tokenizer.convert_tokens_to_ids(predlst[last_np_ind].char)]
            prob_next = predlst[x].logit[processor.tokenizer.convert_tokens_to_ids(predlst[next_np_ind].char)]
            if last_np_ind < last_wb: prob_last = -777
            elif next_np_ind > next_wb: prob_next = -777
            if prob_last >= prob_next:
                w_predlst[x].update(predlst[last_np_ind].char)
            else:
                w_predlst[x].update(predlst[next_np_ind].char)
        elif predlst[x].char != "|":
            last_np_ind = x
            #n_predlst.append(predlst[x])
        else:
            last_wb = x
    return(w_predlst)

def merge_reps(predlst):
    w_predlst = deepcopy(predlst)
    n_predlst = [w_predlst[0]]
    for pred in w_predlst[1:]:
        if pred.char == n_predlst[-1].char:
            n_predlst[-1].end = pred.end
        else:
            n_predlst.append(pred)
    return(n_predlst)

def comb_tones_w_vows(predlst):
    w_predlst = deepcopy(predlst)
    n_predlst = [w_predlst[0]]
    for pred in w_predlst[1:]:
        if pred.char in rep_tones and n_predlst[-1].char != "|":
            n_predlst[-1].end = pred.end
            if pred.char not in n_predlst[-1].char:
                n_predlst[-1].update(n_predlst[-1].char+pred.char)
        else:
            n_predlst.append(pred)
    return(n_predlst)

def comb_words(predlst):
    w_predlst = deepcopy(predlst)
    n_predlst = [w_predlst[0]]
    for pred in w_predlst:
        word = n_predlst[-1]
        if pred.char != "|":
            word.end = pred.end
            word.update(word.char+pred.char)
        else:
            pred.update("")
            word = pred
            n_predlst.append(pred)
    return(n_predlst)
            
def rem_breaks(predlst):
    w_predlst = deepcopy(predlst)
    n_predlst = [w_predlst[0]]
    for x in range(len(w_predlst[:-1])):
        pred = w_predlst[x]
        if pred.char == "|":
            n_predlst.append(w_predlst[x+1])
            n_predlst[-1].start = pred.start
        elif w_predlst[x+1].char != "|":
            n_predlst.append(w_predlst[x+1])
    return(n_predlst)

def ctc_decode(predlst, processor=None, char_align = True, word_align = True):
    predlst_mb = merge_breaks(predlst)
    if processor != None: predlst_pp = process_pads(predlst_mb, processor)
    else: predlst_pp = merge_pads(predlst_mb)
    predlst_mr = merge_reps(predlst_pp)
    predlst_ctwv = comb_tones_w_vows(predlst_mr)
    if word_align: predlst_words = comb_words(predlst_ctwv)
    else: predlst_words = None
    if char_align: predlst_chars = rem_breaks(predlst_ctwv)
    else: predlst_chars = None
    return(predlst_words, predlst_chars)

def transcribe_audio(model_dir, filename, path, aud_ext=".wav", device="cpu", output_path="d:/Northern Prinmi Data/", 
                     has_eaf=False, format=".eaf", min_sil=1000, min_chunk=100, 
                     max_chunk=10000, stride = 1000, char_align = True, word_align = True, lm=None):
    tastt = time.time()
    if os.path.exists(model_dir) and os.path.exists(model_dir+"model/"):
        inner_model_dir = model_dir+"model/"
    else: 
        inner_model_dir = model_dir
    if stride == None: 
        stride = round(max_chunk/6)

    print("***Loading model and processor***")
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model = AutoModelForCTC.from_pretrained(inner_model_dir).to(device)

    if lm!=None:
        #This line necessary to remove <s> and </s> from tokenizer vocab, otherwise prevents integration
        n_tokenizer = Wav2Vec2CTCTokenizer(model_dir+"vocab.json", bos_token=None, eos_token=None)
        vocab_dict = n_tokenizer.get_vocab()
        sorted_vocab_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
        decoder = build_ctcdecoder(labels=list(sorted_vocab_dict.keys()), kenlm_model_path=lm)
        processor = Wav2Vec2Processor(feature_extractor=processor.feature_extractor, tokenizer=n_tokenizer)        

    print("***Chunking audio***")
    if pathlib.Path(path+filename+aud_ext).is_file():
        lib_aud, sr = librosa.load(path+filename+aud_ext, sr=16000)
        length = librosa.get_duration(lib_aud, sr=sr)
        print(f"{filename} is {round(length, 2)}s long")
        #nchunks = stride_chunk(max_chunk, stride, length)
        nchunks = silence_stride_chunk(path+filename+aud_ext, aud_ext, max_chunk, min_chunk, stride, min_sil)
        chunks = [nchunk + [lib_aud[librosa.time_to_samples(nchunk[0]/1000, sr=sr):
                                    librosa.time_to_samples(nchunk[1]/1000, sr=sr)]] for nchunk in nchunks]
        
        print("***Making predictions***")
        eaf = pympi.Eaf(author="transcribe.py")
        eaf.add_linked_file(file_path=path+filename+aud_ext, mimetype=aud_ext[1:])
        eaf.remove_tier('default'), eaf.add_tier("prediction")

        if word_align: 
            eaf.add_tier("words")
            pred_list_words = []
            time_offset = (320 / processor.feature_extractor.sampling_rate)*1000
        if char_align: eaf.add_tier("chars")
        
        phrase_preds = []
        for x in range(len(chunks)):
            st_ms, end_ms = chunks[x][2]
            st_ind, end_ind = round(st_ms/20), round(end_ms/20)
            input_values = processor(chunks[x][3], return_tensors="pt", padding=True, sampling_rate=16000).input_values
            logits = model(input_values.to(device)).logits
            pred_ids = torch.argmax(torch.tensor(logits[0]), dim=-1)
            l_pred_ids = pred_ids[st_ind:len(pred_ids)-end_ind]
            l_logits = logits[0][st_ind:len(logits[0])-end_ind]
            
            if len(l_logits) > 1:
                char_preds = processor.tokenizer.convert_ids_to_tokens(l_pred_ids.tolist())
                pred_list = [prediction(logit=torch.tensor(l_logits[y]), char=char_preds[y], start =chunks[x][0]+y*20+st_ms,
                                        end=chunks[x][0]+(y+1)*20+st_ms) for y in range(len(l_logits))]
                if lm == None: 
                    phrase_preds.append(phone_revert(tone_revert(processor.decode(l_pred_ids))) + " ")
                    pred_list_words, pred_list_chars = ctc_decode(pred_list, char_align=char_align, word_align=word_align)
                else: 
                    dbeam = decoder.decode_beams(l_logits.detach().numpy(), prune_history=True)[0]
                    phrase_preds.append(phone_revert(tone_revert(dbeam[0])))
                    if word_align:
                        pred_list_words += [prediction(logit=dbeam[-2], char=word[0], start=int(round(word[1][0]*time_offset, 2)+chunks[x][0]+st_ms),
                                                    end = int(round(word[1][1]*time_offset, 2)+chunks[x][0]+st_ms)) for word in dbeam[2]]
                    pred_list_chars = ctc_decode(pred_list, char_align=char_align, word_align=word_align)[1]
                
                eaf.add_annotation("prediction", chunks[x][0]+st_ms, chunks[x][1]-end_ms, phrase_preds[-1])
                if word_align: 
                    for word in pred_list_words:
                        eaf.add_annotation("words", word.start, word.end, word.out_char)
                if char_align:
                    for char in pred_list_chars:
                        eaf.add_annotation("chars", char.start, char.end, char.out_char) 
    if has_eaf:
        print("***Fetching previous transcriptions for evaluation***")
        veaf = pympi.Eaf(f"{path}{filename}.eaf")
        tar_tier, anns = get_dominant_tier(veaf)
        tar_txt = " # ".join([ann[2] for ann in anns])
        tar_txt = re.sub(chars_to_ignore_regex, "", tar_txt)
        eaf.add_tier("transcript")
        [eaf.add_annotation("transcript", ann[0], ann[1], re.sub(chars_to_ignore_regex, '', ann[2])) for ann in anns]
        pred_txt = " # ".join(phrase_preds)
        print("WER: ", wer(tar_txt, pred_txt))
        print("CER: ", cer(tar_txt, pred_txt))
    model_name = model_dir[model_dir.rfind("/", 0, -2)+1:-1]
    if lm != None: model_name += "_w_" + lm.split("/")[-1].split(".")[0]
    if format == ".eaf":
        eaf.to_file(f"{output_path}{filename}_{model_name}_preds.eaf")
    elif format == ".TextGrid":
        tg = eaf.to_textgrid()
        tg.to_file(f"{output_path}{filename}_{model_name}_preds.TextGrid")
        
    proc_leng = time.time()-tastt
    print(f"Transcription took {round(proc_leng, 2)}s, or {round(proc_leng/length, 2)}x the length of the recording")  
    print("***Process Complete!***")

def transcribe_dir(model_dir, aud_dir, aud_ext=".wav", device="cpu", output_path="d:/Northern Prinmi Data/Transcripts/", 
                   validate=False, format=".eaf", min_sil=1000, min_chunk=100, max_chunk=10000, 
                   char_align=True, word_align=True, lm=None):
    """Function for automatically chunking all audio files in a given directory by eaf annotation tier time stamps"""
    if not(os.path.exists(output_path)):
        os.mkdir(output_path)
    for path in pathlib.Path(aud_dir).iterdir():
        if path.is_file():
            if str(path).lower().endswith(aud_ext):
                flname = str(path)[len(aud_dir):-len(aud_ext)]
                try:
                    print(f"Attempting to transcribe {flname} using {model_dir}")
                    transcribe_audio(model_dir, flname, aud_dir, aud_ext, device=device, has_eaf=validate, output_path=output_path,
                                     format=format, min_sil=min_sil, min_chunk=min_chunk, max_chunk=max_chunk, char_align=char_align,
                                     word_align=word_align, lm=lm)
                except OSError as error:
                    print(f"Transcribing {flname} failed: {error}")


if __name__ == "__main__":
    tot_start = time.time()
    
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("model_dir", type=str, help="wav2vec 2.0 ASR model to be used")
    parser.add_argument("audio_dir", type=str, help="Directory of audio to be transcribed")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="Run on cpu or gpu")
    parser.add_argument("-t", "--audio_type", type=str, default=".wav", help="Type of audio (.mp3 or .wav)")
    parser.add_argument("-f", "--file_name", type=str, default=None, help="Include file name if you want only one file transcribed")
    parser.add_argument("-o", "--output_dir", type=str, default="./", help="Output directory")
    parser.add_argument("--format", type=str, default=".eaf", help="Output transcription format, either .eaf or .TextGrid")
    parser.add_argument("--has_eaf", action="store_true", help="Audio files have corresponding EAFs for comparison")
    parser.add_argument("--min_sil", type=int, default=1000, help="Minimum decibel threshold for silence detection")
    parser.add_argument("--min_chunk", type=int, default=100, help="Minimum speech chunk length in ms")
    parser.add_argument("--max_chunk", type=int, default=10000, help="Maximum speech chunk length in ms")
    parser.add_argument("--no_char_align", action="store_false", help="Don't align character predictions")
    parser.add_argument("--no_word_align", action="store_false", help="Don't align word predictions")
    parser.add_argument("--lm", default=None, help="Path to kenlm language model")

    args = vars(parser.parse_args())
    if args['file_name'] == None:
        transcribe_dir(model_dir=args['model_dir'], aud_dir=args['audio_dir'], aud_ext=args['audio_type'], device=args['device'], 
                       output_path=args['output_dir'], validate=args['has_eaf'], format=args['format'], min_sil=args['min_sil'], 
                       min_chunk=args['min_chunk'], max_chunk=args['max_chunk'], char_align=args['no_char_align'], 
                       word_align=args['no_word_align'], lm=args['lm'])
    else:
        transcribe_audio(model_dir=args['model_dir'], filename=args['file_name'], path=args['audio_dir'], aud_ext=args['audio_type'], 
                         device=args['device'], output_path=args['output_dir'], has_eaf=args['has_eaf'], format=args['format'], 
                         min_sil=args['min_sil'], min_chunk=args['min_chunk'], max_chunk=args['max_chunk'], 
                         char_align=args['no_char_align'], word_align=args['no_word_align'], lm=args['lm'])
    print('Total time: ', time.time()-tot_start)
