import librosa
import pathlib
import pympi
import soundfile
from pydub import AudioSegment, silence
import numpy
import os

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
    """Uses pydub detect non-silent"""
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
                        #print([y[1]-y[0] for y in tch])
                        nchunks += [[y[0]+start, y[1]+start] for y in tch]
                        solved=True
                        break
                    else:
                        #if x == 10:
                        #    print([y[1]-y[0] for y in tch])
                        pass
                if not(solved): 
                    print("Couldn't solve using shorter minimum silence lengths, increasing silence threshold instead")
                    tch = silence.detect_nonsilent(aud[chunk[0]:chunk[1]], min_silence_len=round(min_sil/2), silence_thresh=-35)
                    for y in tch:
                        ydiff = y[1]-y[0]
                        if ydiff > max_chunk:
                            print(ydiff)
                            for x in range(1, 6):
                                ntch = silence.detect_nonsilent(aud[start+y[0]:start+y[1]], min_silence_len=round(min_sil/2), silence_thresh=-35+x)
                                #print(ntch, -35+x)
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
                    """
                    print("Couldn't divide automatically, instead just chopping up into windows of arbitrary length")
                    divs = round(diff/10000)
                    step = round(diff/divs)
                    for x in range(divs-1):
                        nchunks.append([start+step*x, start+step*(x+1)-1])
                    nchunks.append([start+step*(divs-1), stop])
                    """
            else: nchunks.append([start, stop])
    chunks = [chunk for chunk in nchunks if chunk[1]-chunk[0] > min_chunk]
    return(chunks)

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
        aud = AudioSegment.from_file(path+filename+aud_ext, format=aud_ext[1:])
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
            #Removed the following
            #transcripts = try_to_align_og_phrases_w_detected_phrases(anns, chunks)
        data = []
        for x in range(len(chunks)):
            start = librosa.time_to_samples(chunks[x][0]/1000, sr=sr)
            end = librosa.time_to_samples(chunks[x][1]/1000, sr=sr)
            data.append({'from_file' : filename, 'segment' : x, 'transcript' : transcripts[x], 
                'audio' : {'array': lib_aud[start:end], 'sampling_rate' : sr}})
        print(f"{filename} chunked successfully")
        return(chunks, data, tar_txt, anns)

def transcribe_audio(model_dir, filename, path, aud_ext=".wav", device="cpu", has_eaf=False, 
                     output_path="d:/Northern Prinmi Data/", export=".eaf", min_sil=1000, min_chunk=100, 
                     max_chunk=10000):
    inner_model_dir = model_dir+"model/"
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model = AutoModelForCTC.from_pretrained(inner_model_dir).to(device)
    chunks, target, tar_txt, og_anns = silence_chunk_audio_into_data(filename, path, aud_ext, has_eaf=has_eaf, 
                                                   min_sil=min_sil, min_chunk=min_chunk, max_chunk=max_chunk)
    target_ds = Dataset.from_pandas(DataFrame(target))
    
    #deb_eaf = pympi.Eaf(author="transcriber.py")
    #deb_eaf.add_linked_file(file_path=path+filename+aud_ext, mimetype=aud_ext[1:])
    #[deb_eaf.add_annotation("default", chunk[0], chunk[1], "speaking") for chunk in chunks]
    #deb_eaf.to_file(f"{output_path}DEBUG_{filename}.eaf")

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
        phn_list = processor.tokenizer.convert_ids_to_tokens(pred_ids.tolist())
        phn_preds = [phone_revert(tone_revert(phn)) for phn in phn_list]
        pred = phone_revert(tone_revert(processor.decode(pred_ids))) + " "
        return pred, phn_preds
    
    print("***Making Predictions***")
    eaf = pympi.Eaf(author="transcribe.py")
    eaf.add_linked_file(file_path=path+filename+aud_ext, mimetype=aud_ext[1:])
    eaf.remove_tier('default'), eaf.add_tier("prediction"), eaf.add_tier("phn_preds")
    preds = []
    for x in range(len(chunks)):
        #print(x, chunks[x][1]-chunks[x][0], end=" ")
        pred, phn_preds = get_predictions(x)
        preds.append(pred)
        print(pred)
        eaf.add_annotation("prediction", chunks[x][0], chunks[x][1], pred)
        for y in range(len(phn_preds)):
            eaf.add_annotation("phn_preds", chunks[x][0]+y*20, chunks[x][0]+(y+1)*20, phn_preds[y])
    if has_eaf:
        eaf.add_tier("transcript")
        [eaf.add_annotation("transcript", ann[0], ann[1], ann[2]) for ann in og_anns]
        pred_txt = " # ".join(preds)
        #print(tar_txt)
        #print(pred_txt)
        print("WER: ", wer(tar_txt, pred_txt))
        print("CER: ", cer(tar_txt, pred_txt))
    if export == ".eaf":
        eaf.to_file(f"{output_path}{filename}_predicted_transcription.eaf")
    elif export == ".TextGrid":
        tg = eaf.to_textgrid()
        tg.to_file(f"{output_path}{filename}_predicted_transcription.TextGrid")
    print("***Process Complete!***")

def transcribe_dir(aud_dir, model_dir, aud_ext=".wav", device="cpu", output_path="d:/Northern Prinmi Data/Transcripts/", 
                   validate=False, export=".eaf", min_sil=1000, min_chunk=100, max_chunk=10000):
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
                                     export=export, min_sil=min_sil, min_chunk=min_chunk, max_chunk=max_chunk)
                except OSError as error:
                    print(f"Transcribing {flname} failed: {error}")

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
#transcribe_dir("d:/Northern Prinmi Data/wav-eaf-meta/testing/", model, validate=True)

transcribe_audio(model, t_file, t_path, aud_ext=".wav", has_eaf=True, export='.TextGrid')
#transcribe_audio(model, t2_file, t2_path, aud_ext=".wav")#, has_eaf=True)
#transcribe_audio(model, "wq09_075", t_path, has_eaf=True)
#transcribe_audio(model, "sl05_000", t_path, has_eaf=True)


#chunk_audio_by_silence_into_eaf(t_file, t_path, aud_ext=".wav", has_eaf=True)
#t_data = Dataset.from_pandas(DataFrame(silence_chunk_audio_into_data(t_file, t_path, has_eaf=True)))