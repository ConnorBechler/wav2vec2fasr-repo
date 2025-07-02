"""Audio Processor

Set of functions for processing Elan-annotated data for corpus and NLP purposes

@author: Connor Bechler
@date: Fall 2022
"""

from pathlib import Path
import re
from pympi import Eaf, TextGrid
import librosa
import soundfile
import os
from datasets import Dataset
from pandas import DataFrame
from wav2vec2fasr.orthography import hanzi_reg

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
#warnings.simplefilter("ignore")


def return_tiers(eaf, tar_txt='segnum', find_dominant=False):
    """Function for returning tiers with specified text in name from EAF file 
    Can also just return tier with most children"""
    pos_tiers = [tier for tier in eaf.tiers if len(tier) > 1 and tar_txt in tier]
    #print(pos_tiers)
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

def chunk_audio_by_transcript_into_data(path, 
                                 tar_tier_type : str = "segnum", 
                                 find_dominant : bool = False, 
                                 exclude_regex=hanzi_reg):
    """Function for chunking an audio file by eaf or textgrid time stamp values from a given type of annotation tier into 
    a list of dictionary entries with specific metadata (file, tier, segment, transcript, audio)

    ASSUMES A TRANSCRIPT (.eaf or .TextGrid) EXISTS FOR THE AUDIO FILE IN THE DIRECTORY, WITH THE SAME NAME

    Args:
        path (str | pathlib.Path) : path to .wav or .mp3 audio file
        tar_tier_type (str) : only tiers with this string in their name will be included
        find_dominant (bool) : if True, returns only the tier with the most annotations
        exclude_regex (str | None) : if set, excludes any annotations who match with regex
    Return:
        list of dictionaries containing annotations from transcripts and audio arrays
    """
    path = Path(path)
    if path.is_file() and path.suffix in [".wav", ".mp3"] :
        if path.parent.joinpath(Path(path.stem+".eaf")).is_file(): eaf = Eaf(path.parent.joinpath(Path(path.stem+".eaf")))
        elif path.parent.joinpath(Path(path.stem+".TextGrid")).is_file() : eaf = TextGrid(path.parent.joinpath(Path(path.stem+".TextGrid"))).to_eaf()
        else: raise Exception("Missing transcript (eaf or TextGrid) for this audio file in this directory")
        audio, sr = librosa.load(path, sr=16000)
        pos_tiers = return_tiers(eaf, tar_tier_type, find_dominant)
        data = []
        if exclude_regex == None: exclude_regex = "(?!)"
        for tier in pos_tiers:
            an_dat = eaf.get_annotation_data_for_tier(tier)
            for x in range(len(an_dat)):
                start = librosa.time_to_samples(an_dat[x][0]/1000, sr=sr)
                end = librosa.time_to_samples(an_dat[x][1]/1000, sr=sr)
                #print(an_dat[x][1]/1000 - an_dat[x][0]/1000)
                if not(re.search(exclude_regex, an_dat[x][2])):
                    data.append({'from_file' : path.name, 'tier': tier, 'segment' : x, 'transcript' : an_dat[x][2],
                        'audio' : {'array': audio[start:end], 'sampling_rate' : sr}})
        print(f"{path.name} chunked successfully")
        return(data)
    else:
        print(f"Audio file not found at {path}, chunking not possible")

def chunk_dir_into_dataset(directory, name_tar="", file_list=[], tar_tier_type="segnum"):
    """Function for automatically chunking all .wav and .mp3 files in a given directory by eaf annotation tier time stamps
    Args:
        directory (str | Path) : Path to directory containing audio files and transcript files
        name_tar (str) : Filter string, only files whose names contain the string will be chunked (unless blank)
        file_list (list) : list of file names to be chunked (without extension)
    Returns:
        list of dictionaries containing annotations from transcripts and audio arrays, all individual files transcripts/data
            are concatenated
    """
    dataset = []
    directory = Path(directory)
    for path in directory.iterdir():
        if path.is_file() and path.suffix in [".wav", ".mp3"]:
            if name_tar in path.name or name_tar=="":
                if path.stem in file_list or file_list==[]:
                  try:
                      dataset += chunk_audio_by_transcript_into_data(path, tar_tier_type=tar_tier_type)
                  except OSError as error:
                      print(f"{path.name} chunking failed: {error}") 
    return(dataset)

def chunk_dir_into_audio(directory, out_dir="chunks", out_aud=".wav", name_tar="", file_list=[]):
    """Function for automatically chunking all audio files in a given directory by eaf annotation tier time stamps into audio files
    Args:
        directory (str | Path) : Path to directory containing audio files and transcript files
        name_tar (str) : Filter string, only files whose names contain the string will be chunked (unless blank)
        file_list (list) : list of file names to be chunked (without extension)
    Output:
        directory full of audio files created from transcription chunks
    """
    directory = Path(directory)
    if not(os.path.isdir(out_dir)): os.mkdir(out_dir)
    for path in directory.iterdir():
        if path.is_file() and path.suffix in [".wav", ".mp3"]:
            if name_tar in path.name or name_tar=="":
                if path.stem in file_list or file_list==[]:
                    try:
                        data = chunk_audio_by_transcript_into_data(path, out_dir=out_dir, out_aud=out_aud)
                        for ann in data:
                            soundfile.write(f"{out_dir}/{path.stem}_{ann['tier']}_#{ann['segment']}{out_aud}", 
                                            ann["audio"]["array"], ann["audio"]["sampling_rate"])
                            with open(f"{out_dir}/{path.stem}_{ann['tier']}_#{ann['segment']}.txt", 'w', encoding='utf-8') as f: 
                                f.write(ann["transcript"])
                    except OSError as error:
                        print(f"{path.name} chunking failed: {error}") 

def save_dataset_to_dsk(dataset, path) -> Dataset:
    """Function for saving chunked dataset to disk as huggingface dataset
    args:
        dataset (list) : dataset created by chunk_audio_by_transcript_into_data
        path (str | path) : path to directory to save dataset to (will be created if it doesn't exist)
    Output:
        huggingface dataset saved to path
    """
    hgf_dataset = Dataset.from_pandas(DataFrame(dataset))
    if not(os.path.isdir(path)): os.mkdir(path)
    hgf_dataset.save_to_disk(path)
    return(hgf_dataset)

def create_dataset_from_dir(directory, name : str, out_path, name_tar="", file_list=[], tar_tier_type="segnum") -> Dataset:
    """Function for automatically chunking all .wav and .mp3 files in a given directory by eaf annotation tier time stamps
    Args:
        directory (str | Path) : Path to directory containing audio files and transcript files
        name (str) : Name of dataset directory
        out_path (str | path) : output path where dataset directory should be created
        name_tar (str) : Filter string, only files whose names contain the string will be chunked (unless blank)
        file_list (list) : list of file names to be chunked (without extension)
    Out:
        hgf_dataset directory saved to output location
    Returns:
        Dataset
    """
    directory, out_path = Path(directory), Path(out_path)
    dataset = chunk_dir_into_dataset(directory, name_tar, file_list, tar_tier_type)
    dataset = save_dataset_to_dsk(dataset, out_path.joinpath(name))
    return(dataset)

if __name__ == "__main__":
    
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--data_dir", help="Path to a folder with two subfolders named training and testing containing paired audio (.mp3 or .wav) and transcript (.eaf or .TextGrid) files")
    parser.add_argument("-o", "--output_dir", help="Path where training and testing directories containing huggingface datasets will be created")
    parser.add_argument("-t", "--tier", default="sentence", help="Name or part of name of the tier in each transcript from which the training labels should be drawn")
    args = args = vars(parser.parse_args())

    create_dataset_from_dir(Path(args['data_dir']).joinpath("training/"), "training", Path(args['output_dir']), tar_tier_type=args['tier'])
    create_dataset_from_dir(Path(args['data_dir']).joinpath("testing/"), "testing", Path(args['output_dir']), tar_tier_type=args['tier'])

    
    """TESTING FUNCTIONS
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
    hgf_testing = Dataset.from_pandas(testing_dl)
    hgf_training = Dataset.from_pandas(training_dl)
    
    if not(os.path.isdir(p_testing)): os.mkdir(p_testing)
    hgf_testing.save_to_disk(p_testing)
    
    if not(os.path.isdir(p_training)): os.mkdir(p_training)
    hgf_training.save_to_disk(p_training)
    """

