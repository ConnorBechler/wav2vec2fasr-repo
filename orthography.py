"""
Orthography manager

"""
import pathlib
import re
from pympi import Eaf, TextGrid

rep_chrs = [chr(x) for x in list(range(9312, 9912))]
chars_to_ignore_regex = '[\,\?\.\!\;\:\"\“\%\‘\”\�\。\n\(\/\！\)\）\，]'

def remove_special_chars(text):
    text = re.sub(chars_to_ignore_regex, '', text.lower())
    return(text)

def load_directory(directory, ext=".txt", tier_target = None) -> list[tuple[str, str], ...]:
    """Function for loading all files with a particular extension within a specific directory as text files"""
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
    print("Loaded ", len(txts), " texts")
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
    _rules : dict
    _rule_order : list

    def __init__(self, path : str):
        if path != None:
            tsv = pathlib.Path(path).read_text(encoding="utf-8")
            rows = [line.split("\t") for line in tsv.split("\n")]
            rules, rule_order = {}, []
            print(rows)
            for x in range(len(rows)):
                if rows[x][1] == "comb": rep = rep_chrs[x]
                elif rows[x][1] == "repl": rep = rows[x][3]
                if rows[x][0] in rule_order:
                    rules[rows[x][0]].append((rows[x][2], rep))
                else:
                    rule_order.append(rows[x][0])
                    rules[rows[x][0]] = [(rows[x][2], rep)]
        self._rules, self._rule_order = rules, rule_order

    def apply(self, txt, ignore_rules=[]):
        """Applies orthographic combination rules"""
        output = txt
        for rule in self._rule_order: 
            if rule not in ignore_rules:
                reps = self._rules[rule]
                for rep in reps: 
                    output = re.sub(rep[0], rep[1], output)
        return(output)
    
    def revert(self, txt, ignore_rules=[]):
        """Reverts orthographic combination rules"""
        output = txt
        ignore_rules.append("CLEAN")
        for rule in self._rule_order: 
            if rule not in ignore_rules:
                reps = self._rules[rule]
                for rep in reps: 
                    output = re.sub(rep[1], rep[0], output)
        return(output)

def load_orthographic_scheme_from_tsv(path : str) -> dict:
    """
    Load orthographic combinations from tsv file
    Args:
        path (str) : A string path to a tsv with the following format
            ex. "xxxx\t4graph\nxxx\ttrigraph\nxx\tdigraph"
            Where the first column is the set of characters to be combined into one token and the second
            column is the label for the ruleset
    Returns:
        rules (dict) : A dictionary where each key is the string name of a rule and each value is a list 
            of strings of the orthographic characters to be combined according to the rule
            ex. { "tri": ["xxx", "yyy", "zzz"], "di": ["xx", "yy", "zz"]}
        rule_order (list) : A list of the rule names in the order they should be enacted
            ex. ["tri", "di"]
        combs (list) : A list of specific orthographic transformations to be applied to the text 
            for tokenization by rule order
    """
    tsv = pathlib.Path(path).read_text(encoding="utf-8")
    entries = [line.split("\t") for line in tsv.split("\n")]
    rules, rule_order, combs = {}, [], []
    for x in range(len(entries)):
        combs.append((entries[x][0], rep_chrs[x]))
        if entries[x][1] not in rules:
            rules[entries[x][1]] = [entries[x][0]]
            rule_order.append(entries[x][1])
        else:
            rules[entries[x][1]].append(entries[x][0])
    return(Tokenization_Scheme(rules, rule_order, combs))

def _apply_rules(txt : str, rules : dict, rule_order : list) -> str:
    output = txt
    num_reps = 0
    for rule in rule_order:
        for comb in rules[rule]:
            output = re.sub(comb, rep_chrs[num_reps], output)
            num_reps+=1
    return(output)

def _get_combs_from_rules(rules : list, rule_order : list) -> list:
    num_reps = 0
    encoding = []
    for rule in rule_order:
        for comb in rules[rule]:
            encoding.append((comb, rep_chrs[num_reps]))
            num_reps+=1
    return(encoding)

def apply_combs(txt : str, combs : list) -> str :
    output = txt
    for entry in combs:
        output = re.sub(entry[0], entry[1], output)
    return(output)

def revert_combs(txt : str, combs : list) -> str :
    output = txt
    for entry in combs:
        output = re.sub(entry[1], entry[0], output)
    return(output)

if __name__ == "__main__":
    txts = load_directory("C:/Users/cbech/Desktop/Northern Prinmi Project/wq12_017/", ".eaf", "phrase-seg")
    #txt = remove_special_chars(txts[0][1])
    txt = remove_special_chars(txts[0][1])
    pumi = Tokenization_Scheme("c:/Users/cbech/Desktop/Northern Prinmi Project/Northern-Prinmi-Project-Cluster/pumi_tok.tsv")
    new = pumi.revert(pumi.apply(txt))
    print(new)
    #rules, rule_order, combs = load_combs_from_tsv("c:/Users/cbech/comb_tokens.tsv")
    #tfd = apply_combs(txts[0][1], combs)
    #dtfd = revert_combs(tfd, combs)
    #print(tfd[:100])
    #print(dtfd[:100])
    #print(txts[0][1] == dtfd)