from datasets import load_from_disk, load_metric, Audio
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)

import re
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import random

from jiwer import wer, cer
from Levenshtein import editops

from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer, AutoModelForCTC

#Arguments stuff added with help from https://machinelearningmastery.com/command-line-arguments-for-your-python-script/
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
warnings.simplefilter("ignore")

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

rep_dict = {}
for x in range(len(trips)):
    rep_dict[trips[x]] = rep_trips[x]
for x in range(len(doubs)):
    rep_dict[doubs[x]] = rep_doubs[x]  
for x in range(len(tones)):
    rep_dict[tones[x]] = rep_tones[x]
print("Encoding scheme:", rep_dict)


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

import os

def main_program(eval_dir="output", data_dir=None, checkpoint=None, cpu=False):
    project_dir = "npp_asr"
    output_dir = "output"
    full_project = os.path.join(os.environ["HOME"], project_dir)
    output_path = os.path.join(full_project, output_dir)
    eval_dir = os.path.join(output_path, eval_dir)
    if data_dir == None:
        data_dir = os.path.join(eval_dir, "data/")
    data_train = os.path.join(data_dir, "training/")
    data_test = os.path.join(data_dir, "testing/")
    vocab_dir = os.path.join(eval_dir, "vocab.json")
    if checkpoint == None:
        model_dir = os.path.join(eval_dir, "model/")
    else:
        model_dir = os.path.join(eval_dir, f"checkpoint-{str(checkpoint)}")
    if cpu: 
        device = 'cpu'
    else: 
        device = 'cuda'


    logging.debug(f"Loading training data from {data_train}")
    np_train_ds = load_from_disk(data_train)

    logging.debug(f"Loading test data from {data_test}")
    np_test_ds = load_from_disk(data_test)
    

    try:
        logging.debug("Loading finetuned processor")
        processor = Wav2Vec2Processor.from_pretrained(eval_dir)
    except:
        logging.debug("No finetuned processor found, generating from vocab")
        logging.debug("tokenizer setup")
        tokenizer = Wav2Vec2CTCTokenizer(vocab_dir, 
                                        unk_token="[UNK]", 
                                        pad_token="[PAD]", 
                                        word_delimiter_token="|")
    
        logging.debug("extractor setup")
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, 
                                                    sampling_rate=16000, 
                                                    padding_value=0.0, 
                                                    do_normalize=True, 
                                                    return_attention_mask=True)
    
        logging.debug("processor setup")
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, 
                                    tokenizer=tokenizer)

    
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(audio["array"], 
                                        sampling_rate=audio["sampling_rate"]).input_values[0]
        
        with processor.as_target_processor():
            batch["labels"] = processor(batch["transcript"]).input_ids
        return batch

    logging.debug("training prep")
    np_train_prepped_ds = np_train_ds.map(prepare_dataset, remove_columns=np_train_ds.column_names, num_proc=4)

    logging.debug("test prep")
    np_test_prepped_ds = np_test_ds.map(prepare_dataset, remove_columns=np_test_ds.column_names, num_proc=4)
    
    #Load fine-tuned model
    logging.debug("Loading finetuned model")
    model = AutoModelForCTC.from_pretrained(model_dir).to(device)
    
    
    #logging.debug("Loading wer metric")
    #wer_metric = load_metric("wer")
    
    logging.debug("Loading logits for test values")
    
    full = list(range(len(np_test_prepped_ds["input_values"])))
    #Genre
    songs = list(range(0,5+1))+ list(range(76, 97+1)) + list(range(170,202+1))
    rituals = list(range(6, 75+1)) + list(range(98, 169+1)) + list(range(203, 278+1))
    #Location
    sichuan = list(range(0,5+1)) + list(range(217,278+1))
    yunnan = list(range(6, 216+1))
    #Recording names
    recordings = {
        "sl05_000": list(range(0,6)),
        "td21-22_020": list(range(6,26)),
        "wq09_075": list(range(26, 76)),
        "wq10_11": list(range(76, 98)),
        "wq12_017": list(range(98, 170)),
        "wq14_42": list(range(170, 203)),
        "wq15_069": list(range(203, 217)),
        "yy29_000": list(range(217, 278))
    }
    
    def get_predictions(ind):
        input_dict = processor(np_test_prepped_ds[ind]["input_values"], return_tensors="pt", padding=True, sampling_rate=16000)
        logits = model(input_dict.input_values.to(device)).logits
        pred_ids = torch.argmax(logits, dim=-1)[0]
        label = phone_revert(tone_revert(np_test_ds[ind]["transcript"]))
        pred = phone_revert(tone_revert(processor.decode(pred_ids)))
        return label, pred
    
    def compute_wer(ind_list):
        labels, preds = [], []
        for ind in ind_list:
            label, pred = get_predictions(ind)
            labels.append(label)
            preds.append(pred)
        return(wer(labels, preds))
        
    def compute_cer(ind_list):
        labels, preds = [], []
        for ind in ind_list:
            label, pred = get_predictions(ind)
            labels.append(label)
            preds.append(pred)
        return(cer(labels, preds))
    
    #Function to count the number of substitutions per expected character 
    def count_replacements(ind_list):
        replacements = []
        for ind in ind_list:
            label, pred = get_predictions(ind)
            edits = editops(label, pred)
            replacements += [(label[x[1]], pred[x[2]]) for x in edits if x[0] == 'replace']
        rep_counts = {}
        rep_types = {}
        for x in replacements:
          if x[0] in rep_types:
            rep_counts[x[0]] += 1
            if x[1] in rep_types[x[0]]:
              rep_types[x[0]][x[1]] += 1
            else:
              rep_types[x[0]][x[1]] = 1
          else:
            rep_counts[x[0]] = 1
            rep_types[x[0]] = {x[1]: 1}
        return rep_counts, rep_types
    
    def save_replacements_table(name, ind_list):
        replacements = []
        for ind in ind_list:
            label, pred = get_predictions(ind)
            edits = editops(label, pred)
            replacements += [(label[x[1]], pred[x[2]]) for x in edits if x[0] == 'replace']
        targets = set()
        errors = set()
        for x in replacements:
            targets.add(x[0])
            errors.add(x[1])
        targets = sorted(list(targets))
        errors = sorted(list(errors))
        table = [[0 for y in range(len(errors))] for x in range(len(targets))]
        for x in replacements: table[targets.index(x[0])][errors.index(x[1])] += 1
        csv = "\t"+"\t".join(errors)
        for r in range(len(table)): csv += f"\n{targets[r]}\t"+"\t".join([str(i) for i in table[r]])
        with open(name+'.tsv', 'w') as f:
            f.write(csv)
        return csv

    def count_errors(ind_list):
        edits, err_counts = [], {}#, rep_types = [], {}, {}
        for ind in ind_list:
            label, pred = get_predictions(ind)
            #Include padding to avoid indexing errors with final deletions/insertions
            label, pred = label + "#", pred + "#"
            ops = editops(label, pred)
            edits += [(x[0], label[x[1]], pred[x[2]]) for x in ops if x[0] != 'insert']
            edits += [(x[0], pred[x[2]], label[x[1]]) for x in ops if x[0] == 'insert']
        for e in edits:
            err_counts[e[1]] = err_counts.setdefault(e[1], {y : 0 for y in ['replace', 'insert', 'delete']})
            err_counts[e[1]][e[0]] += 1
            #if e[0] == 'replace': 
            #    rep_types[e[1]] = rep_types.setdefault(e[1], {})
            #    rep_types[e[1]][e[2]] = rep_types[e[1]].setdefault(e[2], 0) + 1
        return err_counts#, rep_types
    
    def save_error_table(name, ind_list):
        err_counts = count_errors(ind_list)
        errs = ['replace', 'insert', 'delete']
        header = "\t"+ "\t".join(list(err_counts.keys())) + "\n"
        body = "\n".join([f"{err[:3]}:\t" + "\t".join([str(err_counts[k][err]) for k in err_counts.keys()]) for err in errs])
        sum = "\nsum:\t" + "\t".join([str(err_counts[k][errs[0]] + err_counts[k][errs[1]] + err_counts[k][errs[2]]) for k in err_counts.keys()])
        csv = header + body + sum
        with open(name+'.tsv', 'w') as f:
            f.write(csv)
        return csv
    

    print(eval_dir)
    #Lines that collect and print the encoded vocab (preprocessed prediction labels) of the training dataset
    vocab_count = {}
    for entry in np_train_ds:
        for char in entry["transcript"]:
            if char in vocab_count: vocab_count[char] += 1
            else: vocab_count[char] = 1
    print("Encoded training vocab:", vocab_count)
    
    #Lines that collect and print the reconverted vocab (expected output) of the training dataset
    d_vocab_count = {}
    for entry in np_train_ds:
        for char in phone_revert(tone_revert(entry["transcript"])):
            if char in d_vocab_count: d_vocab_count[char] += 1
            else: d_vocab_count[char] = 1
    print("Decoded training vocab: ", d_vocab_count)
    
    #Lines that collect and print the encoded vocab (preprocessed prediction labels) of the testing dataset
    vocab_count = {}
    for entry in np_test_ds:
        for char in entry["transcript"]:
            if char in vocab_count: vocab_count[char] += 1
            else: vocab_count[char] = 1
    print("Encoded testing vocab:", vocab_count)
    
    #Lines that collect and print the reconverted vocab (expected output) of the testing dataset
    d_vocab_count = {}
    for entry in np_test_ds:
        for char in phone_revert(tone_revert(entry["transcript"])):
            if char in d_vocab_count: d_vocab_count[char] += 1
            else: d_vocab_count[char] = 1
    print("Decoded testing vocab: ", d_vocab_count)
    
    text = [phone_revert(tone_revert(np_test_ds[ind]["transcript"])) for ind in range(279)]
    with open(eval_dir+'_transcript.txt', 'w') as f:
        f.write("\n".join(text))
    

      
    #Block used to calculate WER on each subsection of the testing set
    print(f"{eval_dir} WER on full testing set: {compute_wer(full)}")
    print(f"{eval_dir} WER on songs: {compute_wer(songs)}")
    print(f"{eval_dir} WER on rituals: {compute_wer(rituals)}")
    print(f"{eval_dir} WER on Sichuan recordings: {compute_wer(sichuan)}")
    print(f"{eval_dir} WER on Yunnan recordings: {compute_wer(yunnan)}")
    for key in list(recordings):
        print(f"{eval_dir} WER on {key}: {compute_wer(recordings[key])}")
    
    #Block used to calculate CER on each subsection of the testing set
    print(f"{eval_dir} CER on full testing set: {compute_cer(full)}")
    print(f"{eval_dir} CER on songs: {compute_cer(songs)}")
    print(f"{eval_dir} CER on rituals: {compute_cer(rituals)}")
    print(f"{eval_dir} CER on Sichuan recordings: {compute_cer(sichuan)}")
    print(f"{eval_dir} CER on Yunnan recordings: {compute_cer(yunnan)}")
    for key in list(recordings):
        print(f"{eval_dir} CER on {key}: {compute_cer(recordings[key])}")
    
    #Save error and replacements tables to eval directory
    print(save_error_table(eval_dir+'_errors', full))
    print(save_replacements_table(eval_dir+'_replacements', full))
    
    

    """#OLD DEBUGGING: Block to take WER of each line from the testing set and average it
    ex_inds = list(range(len(np_test_prepped_ds["input_values"])))
    
    sum_wer, num = 0, 0
    for ind in ex_inds:
        input_dict = processor(np_test_prepped_ds[ind]["input_values"], return_tensors="pt", padding=True, sampling_rate=16000)
        logits = model(input_dict.input_values.to(device)).logits
        pred_ids = torch.argmax(logits, dim=-1)[0]
        pred_str = phone_revert(tone_revert(processor.decode(pred_ids)))
        label_str = phone_revert(tone_revert(np_test_ds[ind]["transcript"]))
        
        
        sum_wer += wer(label_str, pred_str)
        num += 1
    print(f"{eval_dir} average WER: {sum_wer/num}")
    """

    #Block which prints out random sample predictions
    logging.debug("Sample Predictions")
    random.seed("test")
    inds = list(range(0,279))
    random.shuffle(inds)
    ex_inds = inds[:20]
    #ex_inds = list(range(0, 279, 4))
    for ind in ex_inds:
        input_dict = processor(np_test_prepped_ds[ind]["input_values"], return_tensors="pt", padding=True, sampling_rate=16000)
        logits = model(input_dict.input_values.to(device)).logits
        pred_ids = torch.argmax(logits, dim=-1)[0]
        
        print(f"Index: {ind}")
        print("Actual:")
        print(phone_revert(tone_revert(np_test_ds[ind]["transcript"])))
        print("Prediction:")
        print(phone_revert(tone_revert(processor.decode(pred_ids))))
    

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("eval_dir", help="Directory of model to be evaluated")
    parser.add_argument("-d", "--data_dir", default=None, help="Directory of data to evaluate model with")
    parser.add_argument("-c", "--checkpoint", default=None, help="Checkpoint of model to evaluate")
    parser.add_argument("--cpu", action="store_true", help="Run without mixed precision")
    args = vars(parser.parse_args())
    
    logging.debug("***Evaluating model***")
    main_program(eval_dir=args['eval_dir'], 
        data_dir=args['data_dir'], 
        checkpoint=args['checkpoint'], 
        cpu=args['cpu'])
    """if torch.cuda.is_available():
        #main_program(eval_dir="test_nochanges_2-9-23")
        #main_program(eval_dir="test_nochanges_12-21-22")
        #main_program(eval_dir="test_combdiac_12-21-22")
        #main_program(eval_dir="test_combtones_12-21-22")
        #main_program(eval_dir="test_combboth_12-21-22")
        #main_program(eval_dir="test_combdiac_12-21-22")
        #main_program(eval_dir="test_combboth_12-21-22")
        #main_program(eval_dir="test_notones_12-21-22")
    else:
        print("no cuda")"""