"""
Module for basic Prinmi transcription preprocessing
"""

import re

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
rep_combs = {"õ": "õ", "ĩ": "ĩ"}


rep_dict = {}
for x in range(len(trips)):
    rep_dict[trips[x]] = rep_trips[x]
for x in range(len(doubs)):
    rep_dict[doubs[x]] = rep_doubs[x]  
for x in range(len(tones)):
    rep_dict[tones[x]] = rep_tones[x]
print("Encoding scheme:", rep_dict)


def phone_convert(text):
    for x in range(len(trips)):
        text = re.sub(trips[x], rep_trips[x], text)
    for x in range(len(doubs)):
        text = re.sub(doubs[x], rep_doubs[x], text)
    return text

def tone_convert(text):
    for x in range(len(tones)):
        text = re.sub(tones[x], rep_tones[x], text)
    return text

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

def decomb_nasals(text):
    for k in rep_combs:
        text = re.sub(k, rep_combs[k], text)