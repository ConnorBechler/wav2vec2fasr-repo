from datasets import load_from_disk#, load_metric, Audio
#import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)

import re
import torch
#from dataclasses import dataclass, field
#from typing import Any, Dict, List, Optional, Union
import random
import os

from jiwer import wer, cer
from Levenshtein import editops

from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, AutoModelForCTC#, Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC, TrainingArguments, Trainer
from pyctcdecode import build_ctcdecoder

#Arguments stuff added with help from https://machinelearningmastery.com/command-line-arguments-for-your-python-script/
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
warnings.simplefilter("ignore")

from wav2vec2fasr.prinmitext import phone_convert, tone_convert, phone_revert, tone_revert
from wav2vec2fasr import orthography
#TODO: Actually implement the new orthography module's methods

def main_program(home=None, 
                eval_dir="output", 
                data_dir=None, 
                checkpoint=None, 
                cpu=False, 
                lm=None,
                rules_tsv=None):
    """Function for evaluating the performance of a wav2vec2 model on a dataset
    Generates a multitude of outputs, printing most to the console but also creating
    error tables and replacement tables as csvs"""

    if home == None: home = os.environ["HOME"]
    project_dir = "npp_asr"
    output_dir = "output"
    full_project = os.path.join(home, project_dir)
    output_path = os.path.join(full_project, output_dir)
    eval_dir = os.path.join(output_path, eval_dir)
    if data_dir == None:
        data_dir = os.path.join(eval_dir, "data/")
    data_train = os.path.join(data_dir, "training/")
    data_test = os.path.join(data_dir, "testing/")
    vocab_dir = os.path.join(eval_dir, "vocab.json")
    if checkpoint == None:
        if os.path.exists(os.path.join(eval_dir, "model/")):
            model_dir = os.path.join(eval_dir, "model/")
        else: model_dir = eval_dir
    else:
        model_dir = os.path.join(eval_dir, f"checkpoint-{str(checkpoint)}")
    if cpu: 
        device = 'cpu'
    else: 
        device = 'cuda'
    if lm== None: eval_name = eval_dir.split('/')[-1]
    else: eval_name = eval_dir.split('/')[-1]+"_w_"+lm.split("/")[-1].split(".")[0]
    print("Evaluating", eval_name)

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
        tokenizer = Wav2Vec2CTCTokenizer(vocab_dir, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    
        logging.debug("extractor setup")
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, 
                                                    sampling_rate=16000, 
                                                    padding_value=0.0, 
                                                    do_normalize=True, 
                                                    return_attention_mask=True)
    
        logging.debug("processor setup")
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, 
                                    tokenizer=tokenizer)
        
    if lm !=None:
        logging.debug("Setting up language model decoder")
        n_tokenizer = Wav2Vec2CTCTokenizer(vocab_dir, bos_token=None, eos_token=None)
        vocab_dict = n_tokenizer.get_vocab()
        sorted_vocab_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
        decoder = build_ctcdecoder(labels=list(sorted_vocab_dict.keys()), kenlm_model_path=lm)
    
    #Load fine-tuned model
    logging.debug("Loading finetuned model")
    model = AutoModelForCTC.from_pretrained(model_dir).to(device)
    
    full = list(range(len(np_test_ds["transcript"])))
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
    
    def get_predictions(ind, return_comb=False):
        input_dict = processor(np_test_ds[ind]["audio"]['array'], return_tensors="pt", padding=True, sampling_rate=16000)
        logits = model(input_dict.input_values.to(device)).logits
        if lm == None: 
            pred_ids = torch.argmax(logits, dim=-1)[0]
            if return_comb: comb_pred = phone_convert(tone_convert(processor.decode(pred_ids)))
            pred = phone_revert(tone_revert(processor.decode(pred_ids)))
        else: 
            if return_comb: comb_pred = phone_convert(tone_convert(decoder.decode(logits[0].detach().cpu().numpy())))
            pred = phone_revert(tone_revert(decoder.decode(logits[0].detach().cpu().numpy())))
        if return_comb: comb_label = phone_convert(tone_convert(np_test_ds[ind]["transcript"]))
        label = phone_revert(tone_revert(np_test_ds[ind]["transcript"]))
        if return_comb:
            return(label, pred, comb_label, comb_pred)
        else: return(label, pred)

    def compute_wer(ind_list, in_preds=None):
        labels, preds = [], []
        for ind in ind_list:
            if in_preds == None: label, pred = get_predictions(ind)
            else: label, pred = in_preds[0][ind], in_preds[1][ind]
            labels.append(label)
            preds.append(pred)
        return(wer(labels, preds))
        
    def compute_cer(ind_list, in_preds=None):
        labels, preds = [], []
        for ind in ind_list:
            if in_preds == None: label, pred = get_predictions(ind)
            else: label, pred = in_preds[0][ind], in_preds[1][ind]
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
    
    def save_replacements_table(name, ind_list, in_preds=None):
        replacements = []
        comb_replacements = []
        for ind in ind_list:
            if in_preds == None: label, pred, comb_label, comb_pred = get_predictions(ind, return_comb=True)
            else: label, pred, comb_label, comb_pred = in_preds[0][ind], in_preds[1][ind], in_preds[2][ind], in_preds[3][ind]
            edits = editops(label, pred)
            comb_edits = editops(comb_label, comb_pred)
            replacements += [(label[x[1]], pred[x[2]]) for x in edits if x[0] == 'replace']
            comb_replacements += [(comb_label[x[1]], comb_pred[x[2]]) for x in comb_edits if x[0] == 'replace']
        targets, errors = set(), set()
        for x in replacements:
            targets.add(x[0])
            errors.add(x[1])
        targets = sorted(list(targets))
        errors = sorted(list(errors))
        table = [[0 for y in range(len(errors))] for x in range(len(targets))]
        for x in replacements: table[targets.index(x[0])][errors.index(x[1])] += 1
        csv = "\t"+"\t".join(errors)
        for r in range(len(table)): csv += f"\n{targets[r]}\t"+"\t".join([str(i) for i in table[r]])
        with open(name+'.tsv', 'w', encoding='utf-8') as f:
            f.write(csv)
        comb_targets, comb_errors = set(), set()
        for x in comb_replacements:
            comb_targets.add(x[0])
            comb_errors.add(x[1])
        comb_targets = sorted(list(comb_targets))
        comb_errors = sorted(list(comb_errors))
        table = [[0 for y in range(len(comb_errors))] for x in range(len(comb_targets))]
        for x in comb_replacements: table[comb_targets.index(x[0])][comb_errors.index(x[1])] += 1
        comb_csv = "\t"+"\t".join(comb_errors)
        for r in range(len(table)): comb_csv += f"\n{comb_targets[r]}\t"+"\t".join([str(i) for i in table[r]])
        with open(name+'_comb.tsv', 'w', encoding='utf-8') as f:
            f.write(comb_csv)
        return csv, comb_csv

    def count_errors(ind_list, in_preds=None):
        edits, err_counts = [], {}#, rep_types = [], {}, {}
        comb_edits, comb_err_counts = [], {}
        for ind in ind_list:
            if in_preds == None: label, pred, comb_label, comb_pred = get_predictions(ind, return_comb=True)
            else: label, pred, comb_label, comb_pred = in_preds[0][ind], in_preds[1][ind], in_preds[2][ind], in_preds[3][ind]
            #Include padding to avoid indexing errors with final deletions/insertions
            label, pred = label + "#", pred + "#"
            comb_label, comb_pred = comb_label + "#", comb_pred + "#"
            ops, comb_ops = editops(label, pred), editops(comb_label, comb_pred)
            edits += [(x[0], label[x[1]], pred[x[2]]) for x in ops if x[0] != 'insert']
            edits += [(x[0], pred[x[2]], label[x[1]]) for x in ops if x[0] == 'insert']
            comb_edits += [(x[0], comb_label[x[1]], comb_pred[x[2]]) for x in comb_ops if x[0] != 'insert']
            comb_edits += [(x[0], comb_pred[x[2]], comb_label[x[1]]) for x in comb_ops if x[0] == 'insert']
        for e in edits:
            err_counts[e[1]] = err_counts.setdefault(e[1], {y : 0 for y in ['replace', 'insert', 'delete']})
            err_counts[e[1]][e[0]] += 1
        for ce in comb_edits:
            comb_err_counts[ce[1]] = comb_err_counts.setdefault(ce[1], {y : 0 for y in ['replace', 'insert', 'delete']})
            comb_err_counts[ce[1]][ce[0]] += 1
        return err_counts, comb_err_counts
    
    def save_error_table(name, ind_list, in_preds=None):
        err_counts, comb_err_counts = count_errors(ind_list, in_preds)
        errs = ['replace', 'insert', 'delete']
        header = "\t"+ "\t".join(list(err_counts.keys())) + "\n"
        body = "\n".join([f"{err[:3]}:\t" + "\t".join([str(err_counts[k][err]) for k in err_counts.keys()]) for err in errs])
        sum = "\nsum:\t" + "\t".join([str(err_counts[k][errs[0]] + err_counts[k][errs[1]] + err_counts[k][errs[2]]) for k in err_counts.keys()])
        csv = header + body + sum
        with open(name+'.tsv', 'w', encoding='utf-8') as f:
            f.write(csv)
        header = "\t"+ "\t".join(list(comb_err_counts.keys())) + "\n"
        body = "\n".join([f"{err[:3]}:\t" + "\t".join([str(comb_err_counts[k][err]) for k in comb_err_counts.keys()]) for err in errs])
        sum = "\nsum:\t" + "\t".join([str(comb_err_counts[k][errs[0]] + comb_err_counts[k][errs[1]] + comb_err_counts[k][errs[2]]) for k in comb_err_counts.keys()])
        comb_csv = header + body + sum
        with open(name+'_comb.tsv', 'w', encoding='utf-8') as f:
            f.write(comb_csv)
        return csv, comb_csv
    
    logging.debug("Loading logits for test values")
    labels, preds, comb_labels, comb_preds = [], [], [], []
    for ind in full:
        label, pred, comb_label, comb_pred = get_predictions(ind, return_comb=True)
        labels.append(label), preds.append(pred)
        comb_labels.append(comb_label), comb_preds.append(comb_pred)

    print('Printing vocab of', eval_name)
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
    with open(eval_name+'_transcript.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(text))
    
    in_preds = (labels, preds)
    #Block used to calculate WER on each subsection of the testing set
    print(f"{eval_name} WER on full testing set: {compute_wer(full, in_preds)}")
    print(f"{eval_name} WER on songs: {compute_wer(songs, in_preds)}")
    print(f"{eval_name} WER on rituals: {compute_wer(rituals, in_preds)}")
    print(f"{eval_name} WER on Sichuan recordings: {compute_wer(sichuan, in_preds)}")
    print(f"{eval_name} WER on Yunnan recordings: {compute_wer(yunnan, in_preds)}")
    for key in list(recordings):
        print(f"{eval_name} WER on {key}: {compute_wer(recordings[key], in_preds)}")
    
    #Block used to calculate CER on each subsection of the testing set
    print(f"{eval_name} CER on full testing set: {compute_cer(full, in_preds)}")
    print(f"{eval_name} CER on songs: {compute_cer(songs, in_preds)}")
    print(f"{eval_name} CER on rituals: {compute_cer(rituals, in_preds)}")
    print(f"{eval_name} CER on Sichuan recordings: {compute_cer(sichuan, in_preds)}")
    print(f"{eval_name} CER on Yunnan recordings: {compute_cer(yunnan, in_preds)}")
    for key in list(recordings):
        print(f"{eval_name} CER on {key}: {compute_cer(recordings[key], in_preds)}")
    
    in_preds = (labels, preds, comb_labels, comb_preds)
    #Save error and replacements tables to eval directory
    err_table, comb_err_table = save_error_table(eval_name+'_errors', full, in_preds)
    print("Error Table: \n" + err_table)
    print("Combined Error Table: \n" + comb_err_table)
    rep_table, comb_rep_table = save_replacements_table(eval_name+'_replacements', full, in_preds)
    print("Replacements Table: \n" + rep_table)
    print("Combined Replacements Table: \n" + comb_rep_table)

    #Block which prints out random sample predictions
    logging.debug("Sample Predictions")
    random.seed("test")
    inds = list(range(0,279))
    random.shuffle(inds)
    ex_inds = inds[:20]
    #ex_inds = list(range(0, 279, 4))
    for ind in ex_inds:
        label, pred = labels[ind], preds[ind]
        print(f"Index: {ind}")
        print("Actual:")
        print(label)
        print("Prediction:")
        print(pred)
    

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("eval_dir", help="Directory of model to be evaluated")
    parser.add_argument("-d", "--data_dir", default=None, help="Directory of data to evaluate model with")
    parser.add_argument("-c", "--checkpoint", default=None, help="Checkpoint of model to evaluate")
    parser.add_argument("--cpu", action="store_true", help="Run without mixed precision")
    parser.add_argument("--lm", default=None, help="Path to kenlm language model")
    args = vars(parser.parse_args())
    
    logging.debug("***Evaluating model***")
    main_program(eval_dir=args['eval_dir'], 
        data_dir=args['data_dir'], 
        checkpoint=args['checkpoint'], 
        cpu=args['cpu'],
        lm=args['lm'])
