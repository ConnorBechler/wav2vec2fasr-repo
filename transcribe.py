import librosa
import pathlib
import pympi
#import soundfile
from pydub import AudioSegment, silence
import numpy
from math import ceil
import os

from datasets import Dataset
from pandas import DataFrame

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

"""
rep_dict = {}
for x in range(len(trips)):
    rep_dict[trips[x]] = rep_trips[x]
for x in range(len(doubs)):
    rep_dict[doubs[x]] = rep_doubs[x]  
for x in range(len(tones)):
    rep_dict[tones[x]] = rep_tones[x]
print("Encoding scheme:", rep_dict)
"""


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

def check_dominant_tier(eaf, tar_txt=''):
    """Function for returning the tier with the most annotations from an eaf file"""
    pos_tiers = [tier for tier in eaf.tiers if len(tier) > 1 and tar_txt in tier]
    num_chld = 0
    candidate = None
    for tier in pos_tiers:
        chlds = len(eaf.get_child_tiers_for(tier))
        if chlds > num_chld:
            num_chld = chlds
            candidate = tier
    #print(f"Longest tier: {candidate}")
    return(candidate)

def chunk_audio_by_eaf_into_data(filename, path, aud_ext=".mp3"):
    """Function for chunking an audio file by eaf time stamp values from a given annotation tier into a list of numpy arrays"""
    if pathlib.Path(path+filename+aud_ext).is_file() and pathlib.Path(f"{path}{filename}.eaf").is_file():
        audio, sr = librosa.load(path+filename+aud_ext, sr=16000)
        eaf = pympi.Elan.Eaf(f"{path}{filename}.eaf")
        tar_tier = check_dominant_tier(eaf, 'segnum')
        an_dat = eaf.get_annotation_data_for_tier(tar_tier)
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

def chunk_audio_by_silence(filename, path, aud_ext=".mp3", min_sil=1000, min_chunk = 100, max_chunk=10000, stride=1000):
    """
    Uses pydub detect non-silent to chunk audio by silences into speech segments.
    Crude and slow, but functional
    """
    aud = AudioSegment.from_file(path+filename+aud_ext, format=aud_ext[1:])
    chunks = silence.detect_nonsilent(aud, min_silence_len=min_sil, silence_thresh=-35)
    nchunks = []
    print(len(aud))
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

def try_to_align_og_phrases_w_detected_phrases(annotation_data, phrases, eval_win=800):
    """
    Obsolete function trying to pull all words from original transcription that overlap with silence-detection
    generated phrases in order to compare approximately the same time slot between the new transcript and the
    original transcript. Did not work very well, unfortunately.
    """
    anns = annotation_data
    chunks = phrases
    transcripts = ["" for x in range(len(chunks))]
    phrases = [[anns[0][3], 1]]
    new_anns = [[0, 0, 1, 1, anns[0][2]]]
    for a in range(1, len(anns)):
        if anns[a][3] != anns[a-1][3]: phrases.append([anns[a][3], 1])
        else: phrases[-1][1] += 1
        new_anns.append([0, 0, len(phrases), phrases[-1][1], anns[a][2]])
    for a in range(len(anns)):
        new_anns[a][0] = round(((anns[a][1] - anns[a][0])/phrases[new_anns[a][2]-1][1]) * (new_anns[a][3]-1) + anns[a][0])
        new_anns[a][1] = round(((anns[a][1] - anns[a][0])/phrases[new_anns[a][2]-1][1]) * (new_anns[a][3]) + anns[a][0])
    for x in range(len(chunks)):
        for y in range(len(anns)):
            
            if chunks[x][0]-eval_win <= new_anns[y][0] and chunks[x][1]+eval_win >= new_anns[y][1] and anns[y][2][0] not in chars_to_ignore_regex:
                transcripts[x] += anns[y][2] + " "
        if transcripts[x] == '': transcripts[x] += '#'
    return(transcripts)

def silence_chunk_audio_into_data(filename, path, aud_ext=".wav", sr=16000, has_eaf=False, 
                                  min_sil=1000, min_chunk = 100, max_chunk = 10000):
    """Various settings have been tested
    OG: min_sil = 400, max_chunk = 20000 => CER = 0.39
    min_sil = 500, max_chunk = 10000 => CER = 0.39, WER = 0.971
    Best: min_sil = 1000, max_chunk = 10000 => CER = 0.32"""
    if pathlib.Path(path+filename+aud_ext).is_file():
        lib_aud, sr = librosa.load(path+filename+aud_ext, sr=sr)
        chunks = chunk_audio_by_silence(filename, path, aud_ext, min_sil, min_chunk, max_chunk)
        transcripts = ["" for x in range(len(chunks))]
        tar_txt, anns = None, None
        if has_eaf==True: 
            veaf = pympi.Eaf(f"{path}{filename}.eaf")
            tar_tier = veaf.get_tier_ids_for_linguistic_type('phrase')[0]
            anns = veaf.get_annotation_data_for_tier(tar_tier)
            tar_txt = " # ".join([ann[2] for ann in anns])
            tar_txt = re.sub(chars_to_ignore_regex, "", tar_txt)
        data = []
        for x in range(len(chunks)):
            start = librosa.time_to_samples(chunks[x][0]/1000, sr=sr)
            end = librosa.time_to_samples(chunks[x][1]/1000, sr=sr)
            data.append({'from_file' : filename, 'segment' : x, 'transcript' : transcripts[x], 
                'audio' : {'array': lib_aud[start:end], 'sampling_rate' : sr}})
        print(f"{filename} chunked successfully")
        return(chunks, data, tar_txt, anns)

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
    n_predlst = []
    for pred in w_predlst:
        if pred.char == "|" and n_predlst[-1].char == "|":
            n_predlst[-1].end = pred.end
        else:
            n_predlst.append(pred)
    return(n_predlst)

def merge_pads(predlst):
    w_predlst = deepcopy(predlst)
    n_predlst = [w_predlst[0]]
    for pred in w_predlst[1:]:
        if pred.char == "[PAD]" and n_predlst[-1].char != "[PAD]":
            n_predlst[-1].end = pred.end
        else:
            n_predlst.append(pred)
    return(n_predlst)

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
                     max_chunk=10000, char_align = True, word_align = True, lm=None):
    if os.path.exists(model_dir) and os.path.exists(model_dir+"model/"):
        inner_model_dir = model_dir+"model/"
    else: 
        inner_model_dir = model_dir
    stride = round(1000/20)
    #Pipeline testing stuff
    #pipe = pipeline('automatic-speech-recognition', model_dir, decoder=lm)
    #END pipeline testing stuff
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model = AutoModelForCTC.from_pretrained(inner_model_dir).to(device)
    chunks, target, tar_txt, og_anns = silence_chunk_audio_into_data(filename, path, aud_ext, has_eaf=has_eaf, 
                                                   min_sil=min_sil, min_chunk=min_chunk, max_chunk=max_chunk)
    target_ds = Dataset.from_pandas(DataFrame(target))

    if lm!=None:
        #This line necessary to remove <s> and </s> from tokenizer vocab, otherwise prevents integration
        n_tokenizer = Wav2Vec2CTCTokenizer(model_dir+"vocab.json", bos_token=None, eos_token=None)
        vocab_dict = n_tokenizer.get_vocab()
        sorted_vocab_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
        decoder = build_ctcdecoder(labels=list(sorted_vocab_dict.keys()), kenlm_model_path=lm)
        processor = Wav2Vec2Processor(feature_extractor=processor.feature_extractor, tokenizer=n_tokenizer)

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(audio["array"], 
                                        sampling_rate=audio["sampling_rate"]).input_values[0]
        with processor.as_target_processor():
            batch["labels"] = processor(batch["transcript"]).input_ids
        return batch

    print("***Preparing Dataset***")
    target_prepped_ds = target_ds.map(prepare_dataset, remove_columns=target_ds.column_names, num_proc=1)
    print("***Dataset Prepared***")

    print("***Making Predictions***")
    eaf = pympi.Eaf(author="transcribe.py")
    eaf.add_linked_file(file_path=path+filename+aud_ext, mimetype=aud_ext[1:])
    eaf.remove_tier('default'), eaf.add_tier("prediction")
    if word_align: eaf.add_tier("words")
    if char_align: eaf.add_tier("chars")
    phrase_preds, pipe_preds = [], []
    if word_align: 
        pred_list_words = []
        time_offset = (320 / processor.feature_extractor.sampling_rate)*1000
    for x in range(len(chunks)):
        stt = time.time()
        st_ms, end_ms = chunks[x][2]
        st_ind, end_ind = round(st_ms/20), round(end_ms/20)
        input_dict = processor(target_prepped_ds[x]["input_values"],
                               return_tensors="pt", padding=True, sampling_rate=16000)
        logits = model(input_dict.input_values.to(device)).logits
        pred_ids = torch.argmax(torch.tensor(logits[0]), dim=-1)
        l_pred_ids = pred_ids[st_ind:len(pred_ids)-end_ind]
        l_logits = logits[0][st_ind:len(logits[0])-end_ind]
        if len(l_logits) > 1:
            if lm == None: 
                phrase_preds.append(phone_revert(tone_revert(processor.decode(l_pred_ids))) + " ")
            else: 
                #with open('test_old.txt', 'a') as f:
                #    f.write("\n" + str(l_logits))
                dbeam = decoder.decode_beams(l_logits.detach().numpy(), prune_history=True)[0]
                phrase_preds.append(phone_revert(tone_revert(dbeam[0])))
                print(dbeam[0])
                if word_align:
                    pred_list_words += [prediction(logit=dbeam[-2], char=word[0], start=int(round(word[1][0]*time_offset, 2)+chunks[x][0]+st_ms),
                                                end = int(round(word[1][1]*time_offset, 2)+chunks[x][0]+st_ms)) for word in dbeam[2]]
            char_preds = processor.tokenizer.convert_ids_to_tokens(pred_ids.tolist())
            eaf.add_annotation("prediction", chunks[x][0]+st_ms, chunks[x][1]-end_ms, phrase_preds[-1])
            pred_list = [prediction(logit=torch.tensor(l_logits[y]), char=char_preds[y], start =chunks[x][0]+y*20+st_ms,
                                    end=chunks[x][0]+(y+1)*20+st_ms) for y in range(len(l_logits))]
            eaf.add_tier(f"Debug Chunk {str(x)}")
            for pred in pred_list:
                eaf.add_annotation(f"Debug Chunk {str(x)}", pred.start, pred.end, pred.out_char)
            if lm == None: pred_list_words, pred_list_chars = ctc_decode(pred_list, char_align=char_align, word_align=word_align)
            else: pred_list_chars = ctc_decode(pred_list, char_align=char_align, word_align=word_align)[1]
            print("Base performance:", time.time()-stt)
            if word_align: 
                for word in pred_list_words:
                    eaf.add_annotation("words", word.start, word.end, word.out_char)
            if char_align:
                for char in pred_list_chars:
                    eaf.add_annotation("chars", char.start, char.end, char.out_char) 
    if has_eaf:
        eaf.add_tier("transcript")
        [eaf.add_annotation("transcript", ann[0], ann[1], re.sub(chars_to_ignore_regex, '', ann[2])) for ann in og_anns]
        pred_txt = " # ".join(phrase_preds)
        #pipe_pred_txt = " # ".join(pipe_preds)
        print("WER: ", wer(tar_txt, pred_txt))
        print("CER: ", cer(tar_txt, pred_txt))
        #print("Pipe WER: ", wer(tar_txt, pipe_pred_txt))
        #print("PipeCER: ", cer(tar_txt, pipe_pred_txt))
    model_name = model_dir[model_dir.rfind("/", 0, -2)+1:-1]
    if lm != None: model_name += "_w_" + lm.split("/")[-1].split(".")[0]
    if format == ".eaf":
        eaf.to_file(f"{output_path}{filename}_{model_name}_preds.eaf")
    elif format == ".TextGrid":
        tg = eaf.to_textgrid()
        tg.to_file(f"{output_path}{filename}_{model_name}_preds.TextGrid")
        
    print("***Process Complete!***")

def new_transcribe_audio(model_dir, filename, path, aud_ext=".wav", device="cpu", output_path="d:/Northern Prinmi Data/", 
                     has_eaf=False, format=".eaf", min_sil=1000, min_chunk=100, 
                     max_chunk=10000, char_align = True, word_align = True, lm=None):
    if os.path.exists(model_dir) and os.path.exists(model_dir+"model/"):
        inner_model_dir = model_dir+"model/"
    else: 
        inner_model_dir = model_dir
    stride = round(1000/20)
    
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
        aud = AudioSegment.from_file(path+filename+aud_ext, format=aud_ext[1:])
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
                    #with open('test_new.txt', 'a') as f:
                    #    f.write("\n" + str(l_logits))
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
        tar_tier = veaf.get_tier_ids_for_linguistic_type('phrase')[0]
        anns = veaf.get_annotation_data_for_tier(tar_tier)
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
        
        print(f"{filename} chunked successfully")
        
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
        transcribe_dir(args['model_dir'], args['audio_dir'], args['audio_type'], args['device'], args['output_dir'], 
        args['has_eaf'], args['format'], args['min_sil'], args['min_chunk'], args['max_chunk'], args['no_char_align'], 
        args['no_word_align'], args['lm'])
    else:
        transcribe_audio(args['model_dir'], args['file_name'], args['audio_dir'], args['audio_type'], args['device'], 
        args['output_dir'], args['has_eaf'], args['format'], args['min_sil'], args['min_chunk'], args['max_chunk'], 
        args['no_char_align'], args['no_word_align'], args['lm'])
    print('Total time: ', time.time()-tot_start)
