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

def return_tiers(eaf, tar_txt='segnum', find_dominant=False):
    """Function for returning tiers with specified text in name from EAF file 
    Can also just return tier with most children"""
    pos_tiers = [tier for tier in eaf.tiers if len(tier) > 1 and tar_txt in tier]
    if find_dominant:
      num_chld = 0
      candidate = None
      for tier in pos_tiers:
          chlds = len(eaf.get_child_tiers_for(tier))
          if chlds > num_chld:
              num_chld = chlds
              candidate = tier
      pos_tiers = [candidate]
    return(pos_tiers)

def chunk_audio_by_eaf_into_audio(path, out_path=None, out_aud=".wav", tar_tier_type="segnum", find_dominant=False, exclude_regex=u'[\u4e00-\u9fff]'):
    """Function for chunking an audio file by eaf time stamp values from a given annotation tier into audio files"""
    if path.is_file() and pathlib.Path(str(path).rstrip(path.suffix)+".eaf").is_file():
        audio, sr = librosa.load(path, sr=16000)
        print(str(path).rstrip(path.suffix)+".eaf")
        eaf = pympi.Elan.Eaf(str(path).rstrip(path.suffix)+".eaf")
        pos_tiers = return_tiers(eaf, tar_tier_type, find_dominant)
        if out_path == None: out_path = path.parent+f"/{path.stem}_chunks/"
        if not(os.path.isdir(out_path)): os.mkdir(out_path)
        for tier in pos_tiers:
          an_dat = eaf.get_annotation_data_for_tier(tier)
          for x in range(len(an_dat)):
              start = librosa.time_to_samples(an_dat[x][0]/1000, sr=sr)
              end = librosa.time_to_samples(an_dat[x][1]/1000, sr=sr)
              if not(re.search(exclude_regex, an_dat[x][2])):
                soundfile.write(f"{out_path}/{path.stem}_{tier}_#{str(x)}{out_aud}", audio[start:end], sr)
                with open(f"{out_path}/{path.stem}_{tier}_#{str(x)}.txt", 'w', encoding='utf-8') as f: f.write(an_dat[x][2])
        print(f"{path.stem} chunked successfully")
    else:
        print(f"Audio or eaf files not found for {path.stem}, chunking not possible")

def chunk_audio_by_eaf_into_data(path, tar_tier_type="segnum", find_dominant=False, exclude_regex=u'[\u4e00-\u9fff]'):
    """Function for chunking an audio file by eaf time stamp values from a given type of annotation tier into a list of numpy arrays"""
    if path.is_file() and pathlib.Path(str(path).rstrip(path.suffix)+".eaf").is_file():
        audio, sr = librosa.load(path, sr=16000)
        print(str(path).rstrip(path.suffix)+".eaf")
        eaf = pympi.Elan.Eaf(str(path).rstrip(path.suffix)+".eaf")
        pos_tiers = return_tiers(eaf, tar_tier_type, find_dominant)
        data = []
        for tier in pos_tiers:
          an_dat = eaf.get_annotation_data_for_tier(tier)
          for x in range(len(an_dat)):
              start = librosa.time_to_samples(an_dat[x][0]/1000, sr=sr)
              end = librosa.time_to_samples(an_dat[x][1]/1000, sr=sr)
              if not(re.search(exclude_regex, an_dat[x][2])):
                data.append({'from_file' : path.name, 'tier': tier, 'segment' : x, 'transcript' : an_dat[x][2],
                    'audio' : {'array': audio[start:end], 'sampling_rate' : sr}})
        print(f"{path.name} chunked successfully")
        return(data)
    else:
        print(f"Audio or eaf files not found for {path.stem}, chunking not possible")

def chunk_dir_into_audio(directory, out_aud=".wav", name_tar=".wav", file_list=[]):
    """Function for automatically chunking all audio files in a given directory by eaf annotation tier time stamps into audio files"""
    for path in pathlib.Path(directory).iterdir():
        if path.is_file():
            if name_tar in str(path) or name_tar=="":
                if path.stem in file_list or file_list==[]:
                  try:
                      chunk_audio_by_eaf_into_audio(path, out_aud=out_aud)
                  except OSError as error:
                      print(f"{path.name} chunking failed: {error}") 

def chunk_dir_into_dataset(directory, name_tar=".wav", file_list=[]):
    """Function for automatically chunking all audio files in a given directory by eaf annotation tier time stamps"""
    dataset = []
    for path in pathlib.Path(directory).iterdir():
        if path.is_file():
            if name_tar in str(path) or name_tar=="":
                if path.stem in file_list or file_list==[]:
                  try:
                      dataset += chunk_audio_by_eaf_into_data(path)
                  except OSError as error:
                      print(f"{path.name} chunking failed: {error}") 
    return(dataset)

def save_dataset_to_dsk(dataset, path):
  """Function for saving chunked dataset to disk as huggingface dataset"""
  hgf_dataset = Dataset.from_pandas(DataFrame(dataset))
  if not(os.path.isdir(path)): os.mkdir(path)
  hgf_dataset.save_to_disk(path)


if __name__ == "__main__":
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
    
    
    if not(os.path.isdir(p_testing+dir)): os.mkdir(p_testing+dir)
    hgf_testing.save_to_disk(p_testing+dir)
    
    if not(os.path.isdir(p_training+dir)): os.mkdir(p_training+dir)
    hgf_training.save_to_disk(p_training+dir)

