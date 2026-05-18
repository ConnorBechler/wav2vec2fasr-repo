"""
MFA Tools Interface
"""

import pathlib
import pympi
#from datasets import load_from_disk
from wav2vec2fasr.audio_processor import return_tiers
from wav2vec2fasr.orthography import load_config, remove_special_chars, hanzi_reg
import os
import re
from itertools import combinations
import polars as pl
from Levenshtein import editops, matching_blocks

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
    out_file.remove_tier("default")
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

def describe_ts_corpus(corpus_dir : pathlib.Path, tier_list = None, speech_tier_key : str = None, 
                       collate_data : pathlib.Path = None, save_dir : pathlib.Path = False):
    """Method for loading and describing a directory containing a corpus of eaf or TextGrid transcription files.
    Description includes total duration of transcribed audio in the file, duration of tier overlap, and proportion of
    each tier's duration which is overlapping.
    Arguments:
        corpus_dir (pathlib.Path) : path to directory containing transcription files
        tier_list (list) : list of tiers to describe, if None and no speech_tier_key provided, all tiers are loaded
        speech_tier_key (str) : if provided w/o a tier_list, returns all tiers in the file with the key as a substring of their name
        collate_data (pathlib.Path) : if provided, left join the the data table to be collated to the generated description
        save_dir (pathlib.Path) : if provided, save output tables to this directory
    """
    srcs = {path.stem : path for path in corpus_dir.iterdir() if path.suffix in [".eaf", ".TextGrid"]}
    rec_table_col_names = ["name","dur","overlap_dur"]
    rec_table = []#{col : [] for col in rec_table_col_names}
    rec_tier_tables = {}
    if collate_data != None: collate_table = pl.read_csv(collate_data)
    for s, src in enumerate(srcs):
        print("Describing", src)
        rec_table.append([src])
        if ".eaf" in src: file = pympi.Eaf(srcs[src])
        else : file = pympi.TextGrid(srcs[src]).to_eaf()
        if tier_list != None: tiers = tier_list
        elif speech_tier_key != None: tiers = [tier for tier in file.get_tier_names() if speech_tier_key in tier]
        else: tiers = [tier for tier in file.get_tier_names()]
        tier_table_col_names = ["tier","dur","utt_num","wrd_num","chr_num","overlap_dur"]
        tier_table = []
        #Need to collect these separately because they are used to calculate overlap
        tier_durs = {tier : 0 for tier in tiers}
        tier_speech = {}
        total_speech = set()
        for tier in tiers:
            tier_chars = ""
            tier_words = []
            tier_speech[tier] = set()
            an_dat = file.get_annotation_data_for_tier(tier)
            for dat in an_dat: 
                tier_chars += dat[2].replace(" ", "")
                tier_words += dat[2].split(" ")
                tier_durs[tier] += dat[1] - dat[0]
                [(total_speech.add(i), tier_speech[tier].add(i)) for i in list(range(dat[0],dat[1]))]
            tier_table.append([tier, tier_durs[tier], len(an_dat), len(tier_words), len(tier_chars)])
        rec_table[s].append(len(total_speech))
        #speech_frames = set(range(min(total_speech), max(total_speech)))
        # If there are multiple tiers, return duration of overlapping speech
        overlapping_speech = set()
        pair_overlaps = {}
        if len(tiers) > 1:
            pairs = list(combinations(tiers, 2))
            for pair in pairs: 
                pair_overlaps[pair] = tier_speech[pair[0]].intersection(tier_speech[pair[1]])
                overlapping_speech |= pair_overlaps[pair]
        rec_table[s].append(len(overlapping_speech))
        # Return what proportion of each tier is overlapping
        tier_overlapping = {}
        for tier in tiers:
            tier_overlapping[tier] = set()
            for pair in pair_overlaps:
                if tier in pair: tier_overlapping[tier] |= pair_overlaps[pair]
        for t, tier in enumerate(tiers): tier_table[t].append(len(tier_overlapping[tier]))
        tier_table = pl.DataFrame(tier_table,tier_table_col_names,orient="row")
        rec_tier_tables[src] = tier_table
        print(rec_tier_tables[src])
    #rec_table = pl.DataFrame(rec_table).with_columns((pl.col("overlap_dur")/pl.col("dur")).alias("overlap_prop"))
    ave_cols = ["ave_utt_dur","ave_wrd_s", "ave_chr_s", "overlap_prop"]
    ave_table = [[rec_tier_tables[src]["dur"].sum()/rec_tier_tables[src]["utt_num"].sum(), 
                   rec_tier_tables[src]["wrd_num"].sum()/rec_tier_tables[src]["dur"].sum(),
                   rec_tier_tables[src]["chr_num"].sum()/rec_tier_tables[src]["dur"].sum(), 
                   rec_tier_tables[src]["overlap_dur"].sum()/rec_tier_tables[src]["dur"].sum()] for src in srcs]
    rec_table = pl.DataFrame(rec_table, rec_table_col_names, orient="row").with_columns(pl.DataFrame(ave_table, ave_cols, orient="row"))
    if collate_data != None: rec_table = rec_table.join(collate_table, how="left", on="name")
    rec_table.write_csv(save_dir.joinpath("corpus_description.csv"))
    print(rec_table)

def editops_words(seq1, seq2, return_matches=False):
    """Converts list to string of unique characters per word to enable editops on lists
    Arguments:
        seq1 (list) : source list for comparison
        seq2 (list) : destination list for comparison
        return_matches (bool) : set true to return matching blocks instead of editops
    Returns:
        edits (_EditopsList) : returned if return_matches is false
        matches (_MatchingBlocks) : returned if return_matches is true
    """
    encode, decode = {}, {}
    for w, word in enumerate(set(seq1 + seq2)):
        encode[word] = chr(w)
        decode[chr(w)] = word
    new_seq1 = "".join([encode[word] for word in seq1])
    new_seq2 = "".join([encode[word] for word in seq2])
    edits = editops(new_seq1,new_seq2)
    if return_matches:
        matches = matching_blocks(edits, new_seq1, new_seq2)
        return(matches)
    else: return(edits)

def compare_tss(ts1, ts2, tier_list = None, comp_tier_key : str = "words", phrase_tier_key="utterances"):
    """Function for comparing alignments contained in .eaf or .TextGrid format
    Arguments:
        ts1 (Path) : path to transcription in .eaf or .TextGrid alignment
        ts2 (Path) : path to transcription in .eaf or .TextGrid alignment
        tier_list (list) : list of tiers to describe, if None and no speech_tier_key provided, all tiers are loaded
        comp_tier_key (str) : if provided w/o a tier_list, returns all tiers in the file with the key as a substring of their name
    """
    if ts1.suffix == ".eaf": file1 = pympi.Eaf(ts1)
    else : file1 = pympi.TextGrid(ts1).to_eaf()
    if ts2.suffix == ".eaf": file2 = pympi.Eaf(ts2)
    else : file2 = pympi.TextGrid(ts2).to_eaf()
    # Remove unnecessary tiers
    tiers_to_be_removed = [tier for tier in file1.get_tier_names() if "default" in tier]
    file1.remove_tiers(tiers_to_be_removed)
    tiers_to_be_removed = [tier for tier in file2.get_tier_names() if "default" in tier]
    file2.remove_tiers(tiers_to_be_removed)
    # Change naming conventions to match
    tiers_to_be_changed = [tier for tier in file1.get_tier_names() if "segnum-en" not in tier]
    for tier in tiers_to_be_changed: 
        an_dat = file1.get_annotation_data_for_tier(tier)
        file1.add_tier("A_phrase-segnum-en"+" - "+tier)
        for an in an_dat: file1.add_annotation("A_phrase-segnum-en"+" - "+tier, an[0], an[1], an[2])
        file1.remove_tier(tier)
        tiers_to_be_changed = [tier for tier in file1.get_tier_names() if "segnum-en" not in tier]
    tiers_to_be_changed = [tier for tier in file2.get_tier_names() if "pmi_Qaaa-fonipa-x-Pumi-etic" in tier]
    for tier in tiers_to_be_changed: 
        an_dat = file2.get_annotation_data_for_tier(tier)
        new_tier = str(tier).replace("pmi_Qaaa-fonipa-x-Pumi-etic", "en")
        file2.add_tier(new_tier)
        for an in an_dat: file2.add_annotation(new_tier, an[0], an[1], an[2])
        file2.remove_tier(tier)
    #Debug
    #print(file1.get_tier_names())
    #print(file2.get_tier_names())
    if tier_list != None: comp_tiers = tier_list
    elif comp_tier_key != None: comp_tiers = [tier for tier in file1.get_tier_names() if comp_tier_key in tier]
    else: comp_tiers = [tier for tier in file1.get_tier_names()]
    phrase_tiers = [tier for tier in file1.get_tier_names() if phrase_tier_key in tier]
    #Narrow comp_tiers only to tiers included in both
    comp_tiers = [tier for tier in comp_tiers if tier in file2.get_tier_names()]
    tier_table_cols = ["recording","tier","text","start1","end1", "text2","start2","end2"]
    data = []
    if True:
        for tier in phrase_tiers:
            phrase_dat = file1.get_annotation_data_for_tier(tier)
            #Fetch only the word tier with the same speaker letter as the phrase tier
            word_tier = [comp_tier for comp_tier in comp_tiers if comp_tier[0] == tier[0]][0]
            #if len(phrases1) == len(phrases2):
            word_errors = {}
            for phrase in phrase_dat:
                #TODO: Add word error collection
                word_ans1 = file1.get_annotation_data_between_times(word_tier, phrase[0],phrase[1]-1)
                word_ans2 = file2.get_annotation_data_between_times(word_tier, phrase[0],phrase[1]-1)
                words_actual = phrase[2].split()#[word for word in phrase[2].split()]
                words1 = [word[2] for word in word_ans1]
                words2 = [word[2] for word in word_ans2]
                #print("real->mfa",editops_words(words_actual, words1))
                #print("real->wv2",editops_words(words_actual, words2))
                matches = editops_words(words1, words2, return_matches=True)
                aligned_ans1 = [word_ans1[x[0]:x[0]+x[2]] for x in matches][0]
                aligned_ans2 = [word_ans2[x[1]:x[1]+x[2]] for x in matches][0]
                for d, dat in enumerate(aligned_ans1):
                    data.append([ts1.stem, word_tier, dat[2], dat[0], dat[1], 
                                 aligned_ans2[d][2], aligned_ans2[d][0], aligned_ans2[d][1]])
    output = data
    output = pl.DataFrame(data, schema=tier_table_cols, orient="row")
    output = output.with_columns((abs(pl.col("start1")-pl.col("start2"))).alias("start_diff"))
    output = output.with_columns((abs(pl.col("end1")-pl.col("end2"))).alias("end_diff"))
    output = output.with_columns((pl.col("start_diff")+pl.col("end_diff")).alias("total_offset"))
    return(output)
    #input("..wait")

def compare_ts_dirs(ts_dir1 : pathlib.Path, ts_dir2 : pathlib.Path, tier_list : list = None, 
                    comp_tier_key : str = "words", phrase_tier_key="utterances"):
    """Function for comparing alignments contained in .eaf or .TextGrid format from a given directory with matching names
    ts_dir1 (Path) : path to transcription directory containing .eaf or .TextGrid alignments
    ts_dir2 (Path) : path to transcription directory containing .eaf or .TextGrid alignments
    """
    dir1_paths = {path.stem : path for path in ts_dir1.iterdir() if path.suffix in [".eaf", ".TextGrid"]}
    dir2_paths = {path.stem : path for path in ts_dir2.iterdir() if path.suffix in [".eaf", ".TextGrid"]}
    align_paths = [(dir1_paths[ts], dir2_paths[ts]) for ts in dir1_paths if ts in dir2_paths]
    words_df = pl.DataFrame()
    for pair in align_paths:
        print("Comparing", pair[0].stem, "and", pair[1].stem)
        comparison_df = compare_tss(pair[0], pair[1], tier_list=tier_list, comp_tier_key=comp_tier_key, phrase_tier_key=phrase_tier_key)
        if words_df.is_empty(): 
            words_df = comparison_df
        else: words_df = words_df.vstack(comparison_df)
    print(words_df)
    #words_df.write_csv(ts_dir1.stem+"_vs_"+ts_dir2.stem+"_wordcomp.csv")


