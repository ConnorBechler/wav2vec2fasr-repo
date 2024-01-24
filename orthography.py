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
        """Applies orthographic combination rules to a string"""
        output = txt
        for rule in self._rule_order: 
            if rule not in ignore_rules:
                reps = self._rules[rule]
                for rep in reps: 
                    output = re.sub(rep[0], rep[1], output)
        return(output)

    def batch_apply(self, batch, key = "transcript", ignore_rules=[]):
        """Applies orthographic combination rules to a batch"""
        batch[key] = self.apply(batch[key], ignore_rules=ignore_rules)
        return batch
    
    def revert(self, txt, ignore_rules=[]):
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
        

if __name__ == "__main__":
    txts = load_directory("C:/Users/cbech/Desktop/Northern Prinmi Project/wq12_017/", ".eaf", "phrase-seg")
    #txt = remove_special_chars(txts[0][1])
    txt = remove_special_chars(txts[0][1])
    pumi = Tokenization_Scheme("c:/Users/cbech/Desktop/Northern Prinmi Project/Northern-Prinmi-Project-Cluster/pumi_tok.tsv")
    new = pumi.revert(pumi.apply(txt))
    print(new)