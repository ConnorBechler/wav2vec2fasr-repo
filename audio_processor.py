"""Audio Processor

Set of functions for processing Elan-annotated data for corpus and NLP purposes

@author: Connor Bechler
@date: Fall 2022
"""

import pathlib
import re
import pympi
import librosa
import soundfile
import os
from datasets import Dataset
from pandas import DataFrame

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

def chunk_audio_by_eaf(filename, path, aud_ext=".mp3"):
    """Function for chunking an audio file by eaf time stamp values from a given annotation tier"""
    if pathlib.Path(path+filename+aud_ext).is_file() and pathlib.Path(f"{path}{filename}.eaf").is_file():
        audio, sr = librosa.load(path+filename+aud_ext, sr=16000)
        eaf = pympi.Elan.Eaf(f"{path}{filename}.eaf")
        tar_tier = check_dominant_tier(eaf, 'segnum')
        an_dat = eaf.get_annotation_data_for_tier(tar_tier)
        foldername = "chunks"
        if not(os.path.isdir(path+foldername)): os.mkdir(path+foldername)     
        for x in range(len(an_dat)):
            start = librosa.time_to_samples(an_dat[x][0]/1000, sr=sr)
            end = librosa.time_to_samples(an_dat[x][1]/1000, sr=sr)
            soundfile.write(f"{path}{foldername}/{filename}_{str(x)}.mp3", audio[start:end], sr)
            with open(f"{path}{foldername}/{filename}_{str(x)}.txt", 'w', encoding='utf-8') as f: f.write(an_dat[x][2])
        print(f"{filename} chunked successfully")
    else: 
        print(f"Audio or eaf files not found for {filename}, chunking not possible")

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

def chunk_dir(directory, aud_ext=".mp3"):
    """Function for automatically chunking all audio files in a given directory by eaf annotation tier time stamps"""
    for path in pathlib.Path(directory).iterdir():
        if path.is_file():
            if str(path).lower().endswith(aud_ext):
                flname = str(path)[len(directory):-len(aud_ext)]
                try:
                    chunk_audio_by_eaf(flname, directory, aud_ext)
                except OSError as error:
                    print(f"{flname} chunking failed: {error}")

def chunk_dir_into_dataset(directory, aud_ext=".mp3"):
    """Function for automatically chunking all audio files in a given directory by eaf annotation tier time stamps"""
    dataset = []
    for path in pathlib.Path(directory).iterdir():
        if path.is_file():
            if str(path).lower().endswith(aud_ext):
                flname = str(path)[len(directory):-len(aud_ext)]
                try:
                    dataset += chunk_audio_by_eaf_into_data(flname, directory, aud_ext)
                except OSError as error:
                    print(f"{flname} chunking failed: {error}")
    return(dataset)


flname = "jl33_007"
p_testing = "d:/Northern Prinmi Data/wav-eaf-meta/testing"
p_training = "d:/Northern Prinmi Data/wav-eaf-meta/training"
#p = "C:/Users/cbech/Desktop/Northern Prinmi Project/Northern-Prinmi-Project/preprocessed/audio/"

#chunk_audio_by_eaf(flname, p)
#chunk_dir(p, aud_ext='.wav')


#dataset_list = chunk_audio_by_eaf_into_data(flname, p)
#dataset_list = chunk_dir_into_dataset(p, aud_ext='.wav')
testing_dl = DataFrame(chunk_dir_into_dataset(p_testing, aud_ext=".wav"))
training_dl = DataFrame(chunk_dir_into_dataset(p_training, aud_ext=".wav"))
print(testing_dl)
#print(dataset_list)
hgf_testing = Dataset.from_pandas(testing_dl)
hgf_training = Dataset.from_pandas(training_dl)

dir = "/dataset"

#test = [{"a":"1", "b":"2", "c":"3"} for x in range(5)]
#print(DataFrame(test))


#if not(os.path.isdir(p_testing+dir)): os.mkdir(p_testing+dir)
#hgf_testing.save_to_disk(p_testing+dir)

#if not(os.path.isdir(p_training+dir)): os.mkdir(p_training+dir)
#hgf_training.save_to_disk(p_training+dir)

