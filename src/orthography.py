#!/usr/local/bin/python
# -*- coding: utf-8 -*-
"""
Orthography manager

"""
import pathlib
import os
import re
from pympi import Eaf, TextGrid
from importlib import resources
from src import tokenizations

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
warnings.simplefilter("ignore")

rep_chrs = [chr(x) for x in list(range(9312, 9912))]
chars_to_ignore_regex = '[\,\?\.\!\;\:\"\“\%\‘\”\�\。\n\(\/\！\)\）\，]'

def remove_special_chars(text):
    text = re.sub(chars_to_ignore_regex, '', text.lower())
    return(text)

def batch_remove_special_chars(batch, key="transcript"):
    batch[key] = remove_special_chars(batch[key])
    return batch

def load_directory(directory, 
                   ext=".txt", 
                   tier_target = None, 
                   report=True) -> list:
    """
    Function for loading all files with a particular extension within a specific directory
    Args:
        directory (str | pathlib.Path) : directory to load files from
        ext (str) : file extension of the file types to be loaded
            Excepts txt, eaf, and TextGrid
        tier_target (str) : name of tiers from eaf and TextGrid files to be loaded
    Return:
        txts (list) : A list of tuples with each files' name and contents
    """
    txts = []
    for path in pathlib.Path(directory).iterdir():
        if path.is_file() and path.suffix == ext:
                txt = ""
                if ext == '.txt':
                    with path.open(encoding='utf8', errors='replace') as f:
                        txt = f.read()#.replace("None", "ɴone")
                    txts.append((path.name, txt))
                if ext == '.eaf':
                    eaf = Eaf(path)
                    tar_tiers = [tier for tier in eaf.get_tier_names() if tier_target in tier]
                    for tier_name in tar_tiers:
                        txt += " ".join([annotation[2] for annotation in eaf.get_annotation_data_for_tier(tier_name)])
                if ext == '.TextGrid':
                    tg = TextGrid(path)
                    tar_tiers = [tier.name for tier in tg.get_tiers() if tier_target in tier.name]
                    for tier_name in tar_tiers:
                        txt += " ".join([annotation[2] for annotation in tg.get_tier(tier_name).get_all_intervals()])
                if txt != "":
                    txts.append((path.name,txt))
    if report: print("Loaded ", len(txts), " texts")
    return txts

def get_full_vocab(txts : list) -> set:
    mega_txt = "\n".join([txt[1] for txt in txts])
    vocab = set(mega_txt)
    return(vocab)

class Tokenization_Scheme:
    """A class of objects for applying orthographic transformations for tokenization.
Initialization loads from a path to a tsv with the following structure:
"repl\tnasals\tn~\tN\ncomb\ttri\txxx\ncomb\tdi\txx"
Where the first column is the type of operation, the second column is the label for the 
particular rule, the third column is the set of characters to be operated on, and the 
fourth column is used to specify replacement characters. The order of rows DETERMINES 
rule order, so be sure to have the combinations you want to happen first higher up in the table!

Note: If you only want a rule to run on application and not on reversion, label it "CLEAN" """
    path : str
    name : str
    _rules : dict
    _rule_order : list

    def __init__(self, path : str):
        if path != None:
            self.name = pathlib.Path(path).stem
            tsv = pathlib.Path(path).read_text(encoding="utf-8")
            rows = [line.split("\t") for line in tsv.split("\n")]
            rules, rule_order = {}, []
            for x in range(len(rows)):
                if rows[x][1] == "comb": rep = rep_chrs[x]
                elif rows[x][1] == "repl": rep = rows[x][3]
                if rows[x][0] in rule_order:
                    rules[rows[x][0]].append((rows[x][2], rep))
                else:
                    rule_order.append(rows[x][0])
                    rules[rows[x][0]] = [(rows[x][2], rep)]
        self._rules, self._rule_order = rules, rule_order

    def apply(self, txt, ignore_rules=[], only_rule=None) -> str:
        """Applies orthographic combination rules to a string"""
        output = txt
        if only_rule == None:
            for rule in self._rule_order: 
                if rule not in ignore_rules and only_rule == None:
                    reps = self._rules[rule]
                    for rep in reps: 
                        output = re.sub(rep[0], rep[1], output)
        elif only_rule in self._rule_order:
            reps = self._rules[only_rule]
            for rep in reps: 
                output = re.sub(rep[0], rep[1], output)
        return(output)

    def batch_apply(self, batch, key = "transcript", ignore_rules=[], only_rule=None):
        """Applies orthographic combination rules to a batch"""
        batch[key] = self.apply(batch[key], ignore_rules=ignore_rules, only_rule=only_rule)
        return batch
        
    def revert(self, txt, ignore_rules=[]) -> str:
        """Reverts orthographic combination rules applied to a string"""
        output = txt
        ignore_rules.append("CLEAN")
        for rule in self._rule_order: 
            if rule not in ignore_rules:
                reps = self._rules[rule]
                for rep in reps: 
                    output = re.sub(rep[1], rep[0], output)
        return(output)

    def batch_revert(self, batch, key = "transcript", ignore_rules=[]):
        """Reverts orthographic combination rules applied to a batch"""
        batch[key] = self.revert(batch[key], ignore_rules=ignore_rules)
        return batch

    def apply_to_files(self, 
                                files = [], 
                                tar_tiers = [], 
                                new_name = None,
                                revert_op=False):
        """Applies orthographic combination rules to a list of eaf or textgrid files
        Args:
            files (str | list) : either a single path to a TextGrid or eaf file or a list of such paths
            tar_tiers (str | list) : either a single tier name or a list of tier names
            new_name (str) : a new name for the resulting file, defaults to original name plus tokenization scheme name
            revert (bool) : if set to true, reverses application of the tokenization scheme rather than applying it
        """
        if type(files) == type("string"): files = [files]
        if type(tar_tiers) == type("string") : tar_tiers = [tar_tiers]
        for file in files:
            path = pathlib.Path(file)
            name = path.stem
            if path.suffix == ".eaf" : ts = Eaf(path)
            elif path.suffix == ".TextGrid" : ts = TextGrid(path).to_eaf()
            else : raise Exception("files must be .eaf or .TextGrid")
            tiers = [tier for tier in ts.tiers if len(tier) > 1 and tier in tar_tiers]
            for tier in tiers:
                an_dat = ts.get_annotation_data_for_tier(tier)
                if revert_op : 
                    new_an_dat = [(an[0], an[1], self.revert(an[2])) for an in an_dat]
                elif not(revert_op) : 
                    new_an_dat = [(an[0], an[1], self.apply(an[2])) for an in an_dat]
                ts.remove_all_annotations_from_tier(tier)
                for an in new_an_dat:
                    ts.add_annotation(tier, an[0], an[1], an[2])
            if path.suffix == ".TextGrid": ts = ts.to_textgrid()
            if revert_op: operation = "_revert"
            else: operation = "_apply"
            if new_name == None : new_name = name + "_" + self.name + operation
            ts.to_file(os.path.join(path.parent, new_name+path.suffix))
                    
        
#Load default tokenization scheme
with resources.path(tokenizations, "default_tokenization.tsv") as def_path:
    def_tok_path = str(def_path)
def_tok = Tokenization_Scheme(def_tok_path)

def load_tokenization(path, backup=True):
    """Copy contents of new tsv into default tokenization tsv
    Args:
        path (str | pathlib.Path) : path to a tokenization tsv, either full path or tsv must be in tokenizations folder
        backup (bool) : if true, copy previous tokenization scheme to backup_toks.tsv
    """
    path = pathlib.Path(path)
    old_path = pathlib.Path(def_tok_path)
    old_tsv = old_path.read_text(encoding="utf-8")
    if path != None and path.exists():
        print(str(path))
        new_tsv = path.read_text(encoding="utf-8")
    elif pathlib.Path(old_path.parent.joinpath(path)).exists():
        new_tsv = pathlib.Path(old_path.parent.joinpath(path)).read_text(encoding="utf-8")
    else: raise Exception("Bad path provided for update: "+path+" not valid tsv")
    if old_tsv != new_tsv:
        print("Previous tokenization scheme saved to backup_toks.tsv")
        with open(old_path.parent.joinpath("backup_toks.tsv"), "w", encoding="utf-8") as f:
            f.write(old_tsv)
        with open(def_tok_path, "w", encoding="utf-8") as f:
            f.write(new_tsv)
    else: 
        print("Same tokenization as previous, backup_toks.tsv not updated")
    global def_tok
    def_tok = Tokenization_Scheme(def_tok_path) 


TESTING = True
if __name__ == "__main__":
    
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--tokenization", default="default_tokenization.tsv", help="Name of tokenization tsv to load tokenization from")
    args = vars(parser.parse_args())
    if not(TESTING):
        load_tokenization(args['tokenization'])

    if TESTING:
        load_tokenization("pumi_cb.tsv")
        #txts = load_directory("C:/Users/cbech/Desktop/Northern Prinmi Project/wq12_017/", ".eaf", "phrase-seg", report=False)
        #txt = remove_special_chars(txts[0][1])
        #txt = def_tok.apply(remove_special_chars(txts[0][1]), only_rule="CLEAN")
        #print(def_tok.apply(txt)[:100])
        #new = def_tok.revert(def_tok.apply(txt))
        #if new != txt : raise Exception("TEST FAILED")

        #files = "C:/Users/cbech/Desktop/Northern Prinmi Project/td21-22_020/td21-22_020_preds.eaf"
        #tar_tiers = ["prediction", "words", "chars"]
        #def_tok.apply_to_files(files, tar_tiers)
        #def_tok.apply_to_files(files, tar_tiers, revert_op=True)