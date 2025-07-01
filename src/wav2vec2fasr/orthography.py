#!/usr/local/bin/python
# -*- coding: utf-8 -*-
"""
Orthography manager

"""
import pathlib
import os
import re
from pympi import Eaf, TextGrid
from importlib import resources as il_resources
from wav2vec2fasr import resources
import json, unicodedata

with il_resources.path(resources, "config.json") as config_path:
    with open(str(config_path), "r") as f:
        config = json.loads(f.read())

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
warnings.simplefilter("ignore")

rep_chrs = [chr(x) for x in list(range(9312, 9912))]
chars_to_ignore_regex = '[\,\?\.\!\;\:\"\“\%\‘\”\�\。\n\(\/\！\)\）\，\？\[\]\#\@'+"\']"
hanzi_reg = u'[\u4e00-\u9fff]'
pinyin_tones_reg = [chr(i) for i in []]
all_diac_reg = u'[\u0300-\u036f]'
#all_diac_reg = "[" + '\\'.join([chr(i) for i in list(range(768, 879))]) + "]"

def unicode_normalize_chars(text, norm="NFKD", exclude=[]):
    out = ""
    for c in text:
        if c not in exclude: out += unicodedata.normalize(norm, c)
        else: out += c   
    return(out)

def unicode_normalize_batch(batch, key="transcript", norm="NFKD", exclude=[]):
    batch[key] = unicode_normalize_chars(batch[key], norm, exclude)
    return batch

def remove_special_chars(text):
    text = re.sub(chars_to_ignore_regex, '', text.lower())
    return(text)

def batch_remove_special_chars(batch, key="transcript"):
    batch[key] = remove_special_chars(batch[key])
    return batch

def remove_special_chars_from_files(files = [], 
                    tar_tiers = [],
                    new_name = None):
        """Applies orthographic combination rules to a list of eaf or textgrid files
        Args:
            files (str | list) : either a single path to a TextGrid or eaf file or a list of such paths
            tar_tiers (str | list) : either a single tier name or a list of tier names; if empty, defaults to all
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
            if tar_tiers == []: tar_tiers = ts.get_tier_names()
            tiers = [tier for tier in ts.tiers if len(tier) > 1 and tier in tar_tiers]
            for tier in tiers:
                an_dat = ts.get_annotation_data_for_tier(tier)
                new_an_dat = [(an[0], an[1], remove_special_chars(an[2])) for an in an_dat]
                ts.remove_all_annotations_from_tier(tier)
                for an in new_an_dat:
                    ts.add_annotation(tier, an[0], an[1], an[2])
            if path.suffix == ".TextGrid": ts = ts.to_textgrid()
            if new_name == None : new_name = name + "_special_chars_removed"
            ts.to_file(os.path.join(path.parent, new_name+path.suffix))

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
    """A class of objects for applying orthographic transformations for tokenization."""
    path : str
    name : str
    _rules : dict
    _rule_order : list
    _tokens : dict = {}
    _normalize : bool = False
    _preserve : list = []
    _strict : bool = False

    def __init__(self, path : str):
        """Loads tokenization scheme from json or TSV
        Initialization loads from a path to a tsv with the following structure:
        "repl\tnasals\tn~\tN\ncomb\ttri\txxx\ncomb\tdi\txx"
        Where the first column is the label for the particular rule, the second column is  the type 
        of operation, the third column is the set of characters to be operated on, and the 
        fourth column is used to specify replacement characters. The order of rows DETERMINES 
        rule order, so be sure to have the combinations you want to happen first higher up in the table!

        Note: If you only want a rule to run on application and not on reversion, label it "CLEAN" """
        if path != None:
            self.path = path
            self.name = pathlib.Path(path).stem
            suffix = pathlib.Path(path).suffix
            if suffix == ".tsv":
                doc = pathlib.Path(path).read_text(encoding="utf-8")
                rows = [line.split("\t") for line in doc.split("\n")]
                rules, rule_order = {}, []
                for x in range(len(rows)):
                    if rows[x][1] == "comb": rep = rep_chrs[x]
                    elif rows[x][1] == "repl": rep = rows[x][3]
                    if rows[x][0] in rule_order:
                        rules[rows[x][0]].append((rows[x][2], rep))
                    else:
                        rule_order.append(rows[x][0])
                        rules[rows[x][0]] = [(rows[x][2], rep)]
            elif suffix == ".json": 
                doc = json.loads(pathlib.Path(path).read_text("utf-8"))
                rule_order = ["replace", "tokens"]
                rules = {rule : [] for rule in rule_order}
                for item in doc["replace"].items():
                    rules["replace"].append((item[0], item[1]))
                self._tokens, x = [], 0
                for cat in doc["tokens"].keys(): self._tokens += doc["tokens"][cat]
                for phone in self._tokens:
                    if len(phone) > 1 : 
                        rules["tokens"].append((phone, rep_chrs[x]))
                        x += 1
                if "options" in doc:
                    self._normalize = bool(doc["options"]["unicode-normalize"])
                    if "unicode-preserve" in doc["options"]: self._preserve = doc["options"]["unicode-preserve"]
                    self._strict = bool(doc["options"]["strict"])
            self._rules, self._rule_order = rules, rule_order

    def __str__(self):
        return(f"Rules: {self._rules}\n Rule Order: {self._rule_order}")

    def apply(self, txt, ignore_rules=[], only_rule=None) -> str:
        """Applies orthographic combination rules to a string"""
        output = txt
        if self._normalize : output = unicode_normalize_chars(output, exclude=self._preserve)
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
        if self._strict:
            toks = list(set(output)-set(" "))
            exceptions = [self.revert(tok) for tok in toks if self.revert(tok) not in self._tokens]
            for exception in exceptions: output = re.sub(exception, "", output)
        return(output)

    def batch_apply(self, batch, key = "transcript", ignore_rules=[], only_rule=None):
        """Applies orthographic combination rules to a batch"""
        batch[key] = self.apply(batch[key], ignore_rules=ignore_rules, only_rule=only_rule)
        return batch
        
    def revert(self, txt, ignore_rules=[]) -> str:
        """Reverts orthographic combination rules applied to a string"""
        output = txt
        ignore_rules.append("CLEAN")
        ignore_rules.append("replace")
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
        """Applies orthographic combination rules to a list of eaf, textgrid, or txt files
        Args:
            files (str | list) : either a single path to a TextGrid, eaf, or txt file, or a list of such paths
            tar_tiers (str | list) : either a single tier name or a list of tier names; if empty, defaults to all
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
            if path.suffix in [".eaf",".TextGrid"]:
                if tar_tiers == []: tar_tiers = ts.get_tier_names()
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
            elif path.suffix == ".txt":
                ts = False
                with open(path, "r", encoding="utf-8") as f:
                    body = f.read()
                if revert_op: new_body = self.revert(body)
                else : new_body = self.apply(body)
            else : raise Exception("files must be .eaf, .TextGrid, or .txt")
            if revert_op: operation = "_revert"
            else: operation = "_apply"
            if new_name == None : new_name = name + "_" + self.name + operation
            if ts == False: 
                with open(os.path.join(path.parent, new_name+path.suffix), "w", encoding="utf-8") as f:
                    f.write(new_body)
            else: ts.to_file(os.path.join(path.parent, new_name+path.suffix))

    def check_tokenization(self, text) -> list:
        """Check to see if there are any tokens present in the text not present in the tokenization scheme"""
        # Take set of tokenized text (as all tokens are single characters after scheme application)
        toks = list(set(self.apply(text)))
        toks = [self.revert(tok) for tok in toks]
        toks_not_in_scheme = [tok for tok in toks if tok not in self._tokens]
        if " " in toks_not_in_scheme: toks_not_in_scheme.remove(" ")
        return(toks_not_in_scheme)
    
    def save(self, out_path, name=None):
        out_path = pathlib.Path(out_path)
        if name == None: name = self.name
        with open(self.path, 'r', encoding="utf-8") as f:
            text = f.read()
        if not(os.path.exists(out_path)):
            os.mkdir(out_path)
        with open(out_path.joinpath(name +".json"), 'w', encoding="utf-8") as w:
            w.write(text)

                    
#Load default tokenization scheme
with il_resources.path(resources, "default_tokenization.tsv") as def_path:
    def_tok_path = str(def_path)
def_tok = Tokenization_Scheme(def_tok_path)

def load_tokenization(path, backup=True):
    """Copy contents of new tsv into default tokenization tsv
    Args:
        path (str | pathlib.Path) : path to a tokenization tsv, either full path or tsv must be in resources folder
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

def load_config():
    """Function for loading tokenizer and path to evaluation set from config
    Returns:
        tokenizer (Tokenization_Scheme) : loaded from the tsv pathed to in the config file
        eval_set_path (pathlib.Path) : path to evaluation settings json provided in the config file
    """
    with il_resources.path(resources, "config.json") as config_path:
        config_path = pathlib.Path(config_path)
        with open(config_path, "r") as f:
            config = json.loads(f.read())
            eval_set_path = pathlib.Path(config["evaluation_set"])
            tokenization_path = pathlib.Path(config["tokenization"])
    # Check if evaluation_set is just name of file in resources or an outside file path
    # If it is just a file name and is present in resources, reassign as a full path to file in resources
    if len(str(eval_set_path.parent)) < 2 and pathlib.Path(config_path.parent.joinpath(eval_set_path)).exists():
        eval_set_path = config_path.parent.joinpath(eval_set_path)
    # Check if tokenization is just name of file in resources or an outside file path
    # If it is just a file name and is present in resources, reassign as a full path to file in resources
    if len(str(tokenization_path.parent)) < 2 and pathlib.Path(config_path.parent.joinpath(tokenization_path)).exists():
        tokenization_path = config_path.parent.joinpath(tokenization_path)
    tokenizer = Tokenization_Scheme(tokenization_path)
    return(tokenizer, eval_set_path)

def set_tokenization_path(path):
    """Function for setting config tokenization path
    WARNING: MODIFIES INTERNAL PROJECT PATH
    Args:
        path (pathlib.Path | str) : path to tokenization tsv
    """
    if pathlib.Path(path).suffix == ".tsv":
        with il_resources.path(resources, "config.json") as config_path:
            config_path = pathlib.Path(config_path)
            with open(config_path, "r+") as f:
                config = json.loads(f.read())
                config["tokenization"] = path
                f.seek(0)
                f.write(json.dumps(config))
                f.truncate()

    else: raise Exception(f"{path} not path to .tsv")

def explore_corpus(path, ext, tier_target = None):
    """Function for opening set of texts to see characters and character combinations
    Args:
        path (pathlib.Path | str) : path to directory containing transcript files
    """
    tss = load_directory(path, ext, tier_target)
    out = []
    for t in range(len(tss)):
        out.append([tss[t][0], remove_special_chars(tss[t][1])])
    vocab = get_full_vocab(out)
    vocab -= {" ", "\n"}
    vdct = {c : v for c, v in enumerate(vocab)}
    print(vdct)
    inp = ""
    while inp != "QUIT":
        inp = input("Combine characters by index: ")
        command = inp.split(" ")
        if len(command) :
            print(command)
            if len(command) > 1:
                if type(int(command[0])) == type(int(command[1])) == type(5):
                    print(vdct[int(command[0])], vdct[int(command[1])])
                

if __name__ == "__main__":
    #explore_corpus("C:/Users/bechl\code/npp_asr/unaligned-wavs/combined-corpus-w-flex", ".TextGrid", "words")
    
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--change_path", default=None, help="Name of tokenization tsv to set as default in config.json")
    args = vars(parser.parse_args())
    if args["change_path"] != None:
        set_tokenization_path(args['change_path'])