"""
MFA Tools Interface
"""

import pathlib
import pympi
from datasets import load_from_disk
from wav2vec2fasr.audio_processor import return_tiers
from wav2vec2fasr.orthography import load_config, remove_special_chars, hanzi_reg
import os
import re
from itertools import combinations

npp_diacritics = [c for c in ' ̃ʰʲʷ ̥ ']
full_diacritics = [chr(i) for i in range(688,880)]
sups_and_subs = [chr(i) for i in range (8304, 8349)]
ignore_chars = full_diacritics+sups_and_subs
#g and ɡ are different unicode characters, ɡ is ipa

def search_ts_corpus(directory, query, tar_tier="phrase-segnum"):
    for path in pathlib.Path(directory).iterdir():
        if path.is_file():
            if path.suffix in ['.eaf', '.TextGrid']:
                if path.suffix == '.eaf':
                    eaf = pympi.Eaf(path)
                    tar_tiers = [tier for tier in eaf.get_tier_names() if tar_tier in tier]
                    tar_tier_ans = {tier : eaf.get_annotation_data_for_tier(tier) for tier in tar_tiers}
                elif path.suffix == '.TextGrid':
                    tg = pympi.TextGrid(path)
                    tar_tiers = [tier.name for tier in tg.get_tiers() if tar_tier in tier.name]
                    tar_tier_ans = {tier : tg.get_tier(tier).get_all_intervals() for tier in tar_tiers}
                for tier_name in tar_tiers:
                        for annotation in tar_tier_ans[tier_name]:
                            if re.match(query, annotation[2]):
                                print(path, annotation)

def preprocess_ts_for_mfa(audio_path, 
                        src_path, 
                        preserve_tiers = None,
                        exclude_tiers = None,
                        tier_search_key = "phrase-segnum",
                        rem_special_chars = True,
                        apply_tokenization = True,
                        revert_tokenziation = True,
                        output_dir = None,
                        ts_format = ".TextGrid",
                        exclude_regex=hanzi_reg):
    """
    Function for preprocessing transcript files (EAF/TG) for reading by MFA

    Args:
        audio_path : path to audio file (wav or mp3)
        src_path : path to source transcription file (either Praat TextGrid or ELAN eaf)
        model_dir : path to wav2vec 2.0 model
        preserve_tiers (list of str) : list of specific tiers to preserve (use only this OR exclude_tiers OR tier_search_key)
        exclude_tiers (list of str) : list of specific tiers to remove (use only this OR preserve_tiers OR tier_search_key)
        tier_search_key (str) : preserve tiers with this string in their tier name (use only this OR preserve_tiers OR exclude_tiers)
        rem_special_chars (bool) :
        apply_tokenization (bool) : apply your orthographic tokenization scheme to output
        revert_tokenization (bool) : revert your orthographic tokenization scheme
        output_dir (str or pathlib.Path) :
        ts_format (str) : .TextGrid by default or .eaf
    Outputs:
        Transcript file, either an EAF or TextGrid based on the src_file
    """
    audio_path = pathlib.Path(audio_path)
    src_path = pathlib.Path(src_path)
    ort_tokenizer = load_config()[0]
    if src_path.suffix == ".TextGrid" : src_file = pympi.TextGrid(src_path).to_eaf()
    elif src_path.suffix == ".eaf" : src_file = pympi.Eaf(src_path)
    if preserve_tiers != None: tier_list = preserve_tiers
    elif exclude_tiers != None: tier_list = [tier for tier in src_file.get_tier_names() if tier not in exclude_tiers]
    elif tier_search_key != None: tier_list = return_tiers(src_file, tier_search_key)
    out_file = pympi.Eaf()
    if audio_path.exists(): out_file.add_linked_file(file_path=audio_path, mimetype=audio_path.suffix[1:])
    for src_tier in tier_list:
        out_file.add_tier(src_tier)
        # Get annotation data from the source tier
        src_tier_annotations = src_file.get_annotation_data_for_tier(src_tier)
        for ann in src_tier_annotations:
            transcript = ann[2]
            if not(re.search(exclude_regex, ann[2])):
                if rem_special_chars : transcript = remove_special_chars(transcript) 
                if apply_tokenization : transcript = ort_tokenizer.apply(transcript)
                if revert_tokenziation : transcript = ort_tokenizer.revert(transcript)
                out_file.add_annotation(src_tier, ann[0], ann[1], transcript)
    if ts_format == ".TextGrid": out_file = out_file.to_textgrid()
    if output_dir == None:
        output_dir = str(src_path.parent)+"/preprocessed_for_mfa/"
    if not(pathlib.Path(output_dir).exists()): os.mkdir(output_dir)
    out_file.to_file(f"{output_dir}/{src_path.stem}{ts_format}")

def load_vocab_from_ts_directory(directory, tar_tier="phrase-segnum"):
    """Function for loading all words and characters from a corpus of transcripts (either EAF or TG)"""
    words = []
    for path in pathlib.Path(directory).iterdir():
        if path.is_file():
            if path.suffix == '.eaf':
                eaf = pympi.Eaf(path)
                tar_tiers = [tier for tier in eaf.get_tier_names() if tar_tier in tier]
                for tier_name in tar_tiers:
                    for annotation in eaf.get_annotation_data_for_tier(tier_name):
                        words += annotation[2].split(' ')
            elif path.suffix == '.TextGrid':
                tg = pympi.TextGrid(path)
                tar_tiers = [tier.name for tier in tg.get_tiers() if tar_tier in tier.name]
                for tier_name in tar_tiers:
                    for annotation in tg.get_tier(tier_name).get_all_intervals():
                        words += annotation[2].split(' ')
    vocab = [word for word in list(set(words)) if len(word) > 0]
    chars = [chr for chr in set("".join(vocab))]
    return(vocab, chars)

def generate_simple_pron_dict(vocab, exclude = ignore_chars):
    text = ""
    for word in vocab:
      pronunciation = [letter for letter in word if letter not in full_diacritics]#letter not in exclude]
      if len(pronunciation) < 1:
        pronunciation = ["sil"]
        print(word + '\t' + " ".join(pronunciation))
      text += word + '\t' + " ".join(pronunciation) + "\n"
    return(text)

def generate_pron_dict_w_phonemap(vocab, phoneMapping, convert=True, out_dir = pathlib.Path("./")):
    text = ""
    # Get list of phones from phone map (text file with phones separated by tabs)
    with open(phoneMapping, "r", encoding="utf-8") as f:
        pmtxt = f.read()
        phones = [l.split("\t") for l in pmtxt.split("\n")]
        phones.reverse()
    vocab_txt = "\n".join(vocab)
    for p in range(len(phones)):
        vocab_txt = re.sub(phones[p][0], f" {str(p)} ", vocab_txt)
    # Remove characters that are not in the phone map
    vocab_txt = re.sub("[^0-9| |\n]+", "", vocab_txt)
    phoned_vocab = vocab_txt.split("\n")
    for w in range(len(phoned_vocab)):
        if convert : i = 1
        else: i = 0
        pronunciation = [phones[int(p)][i] for p in phoned_vocab[w].split(" ") if p != ""]#letter not in exclude]
        #print(vocab[w], phoned_vocab[w], pronunciation)
        if len(pronunciation) < 1:
          pronunciation = ["sil"]
          #print(word + '\t' + " ".join(pronunciation))
        text += vocab[w] + '\t' + " ".join(pronunciation) + "\n"
    with open(str(out_dir)+"/pron_dict_pm.txt", "w", encoding="utf-8") as f:
        f.write(text) 
    return(text)

def describe_audio_corpus(corpus_dir : pathlib.Path, tier_list = None, speech_tier_key : str = None):
    """Method for loading and describing a directory containing a corpus of eaf or TextGrid transcription files.
    Description includes total duration of transcribed audio in the file, duration of tier overlap, and proportion of
    each tier's duration which is overlapping.
    Arguments:
        corpus_dir (pathlib.Path) : path to directory containing transcription files
        tier_list (list) : list of tiers to describe, if None and no speech_tier_key provided, all tiers are loaded
        speech_tier_key (str) : if provided w/o a tier_list, returns all tiers in the file with the key as a substring of their name
    """
    srcs = {path.stem : path for path in corpus_dir.iterdir() if path.suffix in [".eaf", ".TextGrid"]}
    for src in srcs:
        print("Describing", src)
        if ".eaf" in src: file = pympi.Eaf(srcs[src])
        else : file = pympi.TextGrid(srcs[src]).to_eaf()
        if tier_list != None: tiers = tier_list
        elif speech_tier_key != None: tiers = [tier for tier in file.get_tier_names() if speech_tier_key in tier]
        else: tiers = [tier for tier in file.get_tier_names()]
        tier_durs = {tier : 0 for tier in tiers}
        tier_utterances = {}
        tier_speech = {tier : set() for tier in tiers}
        total_speech = set()
        for tier in tiers:
            an_dat = file.get_annotation_data_for_tier(tier)
            tier_utterances[tier] = len(an_dat)
            for dat in an_dat: 
                tier_durs[tier] += dat[1] - dat[0]
                #[total_speech.add(i) for i in list(range(dat[0],dat[1]))]
                [(total_speech.add(i), tier_speech[tier].add(i)) for i in list(range(dat[0],dat[1]))]
        #speech_frames = set(range(min(total_speech), max(total_speech)))
        tier_ave_utterance = {tier : tier_durs[tier]/tier_utterances[tier] for tier in tiers}
        print("Total milliseconds of speech:", len(total_speech))
        print("Tier durations in MS:", tier_durs)
        print("Number of utterances by tier:", tier_utterances)
        print("Average utterance duration by tier:", tier_ave_utterance)
        # If there are multiple tiers, return duration of overlapping speech
        overlapping_speech = set()
        pair_overlaps = {}
        if len(tiers) > 1:
            pairs = list(combinations(tiers, 2))
            #print(pairs)
            for pair in pairs: 
                pair_overlaps[pair] = tier_speech[pair[0]].intersection(tier_speech[pair[1]])
                overlapping_speech |= pair_overlaps[pair]
            print("Overlapping speech duration:", len(overlapping_speech))
        # If there is overlapping speech, return what proportion of each tier is overlapping
        if len(overlapping_speech) > 0:
            tier_overlapping = {}
            for tier in tiers:
                tier_overlapping[tier] = set()
                for pair in pair_overlaps:
                    if tier in pair: tier_overlapping[tier] |= pair_overlaps[pair]
            tier_prop_ovlp = {tier : len(tier_overlapping[tier])/tier_durs[tier] for tier in tiers}
            print("Proportion of tier which is overlapped with another:", tier_prop_ovlp)
            #[print(tier, "proportion overlap: ", len(tier_overlapping[tier])/tier_durs[tier]) for tier in tiers]