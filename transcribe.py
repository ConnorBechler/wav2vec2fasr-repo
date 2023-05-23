import librosa
import pathlib
import pympi
import soundfile
from pydub import AudioSegment, silence
import numpy

from datasets import Dataset
from pandas import DataFrame

import re
from jiwer import wer, cer

from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer, AutoModelForCTC
import torch

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


def chunk_audio_by_silence(filename, path, aud_ext=".mp3", min_sil=1000, min_chunk = 100, max_chunk=10000):
    """Uses librosa.effects.split described at https://librosa.org/doc/main/generated/librosa.effects.split.html"""
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
                for x in range(2, 10):
                    tch = silence.detect_nonsilent(aud[chunk[0]:chunk[1]], min_silence_len=round(min_sil/x), silence_thresh=-35)
                    if len([y for y in tch if y[1]-y[0] > max_chunk]) == 0: 
                        print("solved at", round(min_sil/x))
                        nchunks += [[y[0]+start, y[1]+start] for y in tch]
                        solved=True
                        break
                    else:
                        pass
                if not(solved): 
                    print("Couldn't divide automatically, instead just chopping up into windows of arbitrary length")
                    divs = round(diff/10000)
                    step = round(diff/divs)
                    for x in range(divs-1):
                        nchunks.append([start+step*x, start+step*(x+1)-1])
                    nchunks.append([start+step*(divs-1), stop])
            else: nchunks.append([start, stop])
    chunks = nchunks
    return(chunks)

def silence_chunk_audio_into_data(filename, path, aud_ext=".wav", sr=16000, has_eaf=False, 
                                  min_sil=1000, min_chunk = 100, max_chunk = 10000):
    """Various settings have been tested
    OG: min_sil = 400, max_chunk = 20000 => CER = 0.39
    min_sil = 500, max_chunk = 10000 => CER = 0.39, WER = 0.971
    Best: min_sil = 1000, max_chunk = 10000 => CER = 0.32"""
    if pathlib.Path(path+filename+aud_ext).is_file():
        aud = AudioSegment.from_file(path+filename+aud_ext, format=aud_ext[1:])
        lib_aud, sr = librosa.load(path+filename+aud_ext, sr=sr)
        chunks = chunk_audio_by_silence(filename, path, aud_ext, min_sil, min_chunk, max_chunk)
        transcripts = ["" for x in range(len(chunks))]
        if has_eaf==True: 
            eval_win = 800
            veaf = pympi.Eaf(f"{path}{filename}.eaf")
            tar_tier = veaf.get_tier_ids_for_linguistic_type('word')[0]
            anns = veaf.get_annotation_data_for_tier(tar_tier)
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
        data = []
        for x in range(len(chunks)):
            start = librosa.time_to_samples(chunks[x][0]/1000, sr=sr)
            end = librosa.time_to_samples(chunks[x][1]/1000, sr=sr)
            data.append({'from_file' : filename, 'segment' : x, 'transcript' : transcripts[x], 
                'audio' : {'array': lib_aud[start:end], 'sampling_rate' : sr}})
        print(f"{filename} chunked successfully")
        return(chunks, data)

def apply_model_to_audio(dir, filename, path, aud_ext=".wav", device="cpu", has_eaf=False, export=".eaf", 
                         min_sil=1000, min_chunk=100, max_chunk=10000):
    model_dir = dir+"model/"
    processor = Wav2Vec2Processor.from_pretrained(dir)
    model = AutoModelForCTC.from_pretrained(model_dir).to(device)
    chunks, target = silence_chunk_audio_into_data(filename, path, aud_ext, has_eaf=has_eaf, 
                                                   min_sil=min_sil, min_chunk=min_chunk, max_chunk=max_chunk)
    target_ds = Dataset.from_pandas(DataFrame(target))

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

    def get_predictions(ind):
        input_dict = processor(target_prepped_ds[ind]["input_values"], return_tensors="pt", padding=True, sampling_rate=16000)
        logits = model(input_dict.input_values.to(device)).logits
        pred_ids = torch.argmax(logits, dim=-1)[0]
        if has_eaf: label = target_ds[ind]["transcript"]
        else: label = None
        pred = phone_revert(tone_revert(processor.decode(pred_ids))) + " "
        return label, pred
    
    print("***Making Predictions***")
    eaf = pympi.Eaf(author="transcribe.py")
    eaf.add_linked_file(file_path=path+filename+aud_ext, mimetype=aud_ext[1:])
    eaf.remove_tier('default'), eaf.add_tier("prediction")
    if has_eaf: 
        labels, preds = [], []
        eaf.add_tier("transcript")
    for x in range(len(chunks)):
        label, pred = get_predictions(x)
        if has_eaf: 
            labels.append(label), preds.append(pred)
            print(str(x), label + "|| "+ pred)
            eaf.add_annotation("transcript", chunks[x][0], chunks[x][1], label)
        else: print(pred)
        eaf.add_annotation("prediction", chunks[x][0], chunks[x][1], pred)
    if has_eaf:
        print("WER: ", wer(labels, preds))
        print("CER: ", cer(labels, preds))
    if export == ".eaf":
        eaf.to_file(f"d:/Northern Prinmi Data/{filename}_test.eaf")
    elif export == ".TextGrid":
        tg = eaf.to_textgrid()
        tg.to_file(f"d:/Northern Prinmi Data/{filename}_test.TextGrid")
    print("***Process Complete!***")



t_path = "d:/Northern Prinmi Data/wav-eaf-meta/"
t_file = "wq15_069"
t2_path = "d:/Northern Prinmi Data/"
t2_file = "jz18_040"
model = "D:/Northern Prinmi Data/models/model_5-11-23_combboth_1e-4/"

"""
eaf = pympi.Eaf(t_path+t_file+".eaf")
print(eaf.get_tier_names())
tar = eaf.get_tier_ids_for_linguistic_type('word')[0]
for x in eaf.get_annotation_data_between_times(tar, 13000, 14000):
    print(x[2], end=" ")
"""

#apply_model_to_audio(model, t_file, t_path, aud_ext=".wav", has_eaf=True)
apply_model_to_audio(model, t2_file, t2_path, aud_ext=".wav")#, has_eaf=True)

#chunk_audio_by_silence_into_eaf(t_file, t_path, aud_ext=".wav", has_eaf=True)
#t_data = Dataset.from_pandas(DataFrame(silence_chunk_audio_into_data(t_file, t_path, has_eaf=True)))