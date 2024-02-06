"""Text Processor

Set of functions for parsing PDF interlinear glosses converted to text documents and using them as a corpus. 
Crude but functional, not super useful with current ELAR documents.

TODO: Actually update to pull text from ELAN transcripts 

@Author: Connor Bechler
@Date: Summer 2022
"""
import pathlib
import re
import pympi

def load_directory(directory, ext=".txt"):
    """Function for loading all files with a particular extension within a specific directory as text files"""
    txts = []
    for path in pathlib.Path(directory).iterdir():
        if path.is_file():
            if str(path).lower().endswith(ext):
                with open(path, encoding='utf8', errors='replace') as f:
                    txt = f.read()#.replace("None", "ɴone")
                docname = str(path).split('\\')[-1]
                txts.append((docname, txt))
    print("Loaded ", len(txts), " texts")
    return txts

def items_in(item_list, target_list):
    """Cross checks if any items in a list are in a second target list"""
    for item in item_list:
        if item in target_list:
            return(True)
            break
    return(False)

def process_interlinear(txt):
    """Function for reorganizing an interlinear txt file"""
    split_txt = txt.split("\n\n")
    sentences = []
    sentence = []
    sent_num = 0
    chk = ["Free", "Note", "Chn", "."]
    for b in range(len(split_txt)):
        block = split_txt[b].split("\n")
        if "Word" in block[0]:
            if sent_num != 0:
                sentence.append(["。","。","。","。","。","。"])
                sentences.append([sent_num, sentence, tran])
            sentence = []
            tran = ""
            sent_num += 1
        elif items_in(chk, block[0]): tran += "\n".join(block) + "\n"
        else:
            word = None
            block_leng = len(block)
            if  block_leng == 3: 
                word = [""] + block + ["", ""]
            elif block_leng == 4:
                nn_block = split_txt[b+2].split("\n")
                if len(nn_block) == 2 and not(items_in(chk, nn_block[0])):
                    word = block + split_txt[b+2].split("\n")
                else: word = block + ["",""]
            elif block_leng == 6: word = block
            if word != None: sentence.append(word)
    #print(sentences)
    return(sentences)

def create_raw_prinmi_files(txt_list, new_file_name_add = "JUST_PRINMI"):
    """Function for creating raw prinmi text files from converted interlinears"""
    for text in txt_list:
        procd_txt = process_interlinear(text[1])
        new_file_txt = ""
        for s in procd_txt:
            for w in s[1]:
                new_file_txt += w[0]
                #Add space between words if not already built into transcript
                if len(w[0]) > 0:
                    if w[0][-1] != " ": new_file_txt += " "
        new_file_name = text[0][:-4] + new_file_name_add + ".txt"
        with open(new_file_name, "w", encoding="utf-8") as f:
            f.write(new_file_txt)
    print("Wrote all files")

def create_raw_prinmi_txt(glossed):
    """Function for creating raw prinmi txt from glossed data"""
    text = ""
    for s in glossed:
        for w in s[1]:
            text += w[0]
            #Add space between words if not already built into transcript
            if len(w[0]) > 0:
                if w[0][-1] != " ": text += " "
    return text

def create_glossed_prinmi_files(txt_list, new_file_name_add = "GLOSSED_PRINMI"):
    """Function for creating raw prinmi text files from converted interlinears"""
    for text in txt_list:
        procd_txt = process_interlinear(text[1])
        new_file_txt = ""
        for s in procd_txt:
            for w in s[1]:
                add = " "
                if len(w[0]) == 0: new_file_txt = new_file_txt[:-1]
                #Add space between words if not already built into transcript
                elif w[0][-1] == " ": add = ""
                new_file_txt += w[0]+"<"+w[2]+">" + add
        new_file_name = text[0][:-4] + new_file_name_add + ".txt"
        with open(new_file_name, "w", encoding="utf-8") as f:
            f.write(new_file_txt)
    print("Wrote all files")

def create_data_structure(text_dir, meta_dir):
    """Function for creating data structure of interlinear glossed texts and metadata"""
    texts = load_directory(text_dir)
    metas = load_directory(meta_dir)
    data_structure = []
    for t in range(len(texts)):
        name = texts[t][0]
        int_txt = process_interlinear(texts[t][1])
        meta = process_metadata(metas[t][1])
        data_structure.append((name, meta, int_txt))
    return(data_structure)

def create_count_table_old(input_dir, delim="\t"):
    """Function for counting all words in an interlinear glossed text list"""
    text_list = load_directory(input_dir)
    dim = len(text_list)
    words = {}
    punc = ["(", ")", ",", ".", " "]
    for t in range(len(text_list)):
        text = text_list[t]
        procd_txt = process_interlinear(text[1])
        new_file_txt = ""
        for s in procd_txt:
            for w in s[1]:
                testw = "".join([x for x in w[0] if x not in punc])
                if testw in words:
                    words[testw][t] += 1
                else:
                    words[testw] = [0 for x in range(dim)]
                    words[testw][t] = 1
    table_text = delim + delim.join([n[0] for n in text_list]) + "\n"
    head_words = words.keys()
    for w in head_words:
        table_text += w
        for x in words[w]:
            table_text += delim +str(x)
        table_text += "\n"
    
    if delim == "\t": ext = ".tsv"
    elif delim == ",": ext = ".csv"
    with open("COUNT_TABLE" + ext, "w", encoding="utf-8") as f:
            f.write(table_text)

def process_metadata(txt, tabs=False):
    """Function for processing metadata from ELAR"""
    txt = txt.replace("\u200e","")
    mkeys = "Title,Description,Region,Address,Topic,Genre,Keywords,Languages,Actors".split(",")
    searches = "Title: ?,Title:?.*\n\nDescription:? ,Region:? ,Address:? ,Topic:? ,Genre:? ,Keyword:? ,Language\nName:? ,Full Name:? ".split(",")
    data = {}
    for s in range(len(searches)):
        if tabs: data[mkeys[s]] = "\t".join([x[1] for x in re.findall('('+searches[s]+')(.*)', txt)])
        else: data[mkeys[s]] = [x[1] for x in re.findall('('+searches[s]+')(.*)', txt)]
    return(data)

def count_words(data_strc):
    """Function for counting all words in an interlinear glossed data structure"""
    dim = len(data_strc)
    words = {}
    punc = ["(", ")", ",", ".", " "]
    for t in range(dim):
        procd_txt = data_strc[t][2]
        new_file_txt = ""
        for s in procd_txt:
            for w in s[1]:
                testw = "".join([x for x in w[0] if x not in punc])
                if testw in words:
                    words[testw][t] += 1
                else:
                    words[testw] = [0 for x in range(dim)]
                    words[testw][t] = 1
    return(words)

def calc_word_density(data_strc):
    """Function for calculating word density in an interlinear glossed data structure"""
    dim = len(data_strc)
    words = {}
    punc = ["(", ")", ",", ".", " "]
    doc_sents = [len(data_strc[t][2]) for t in range(dim)]
    for t in range(dim):
        procd_txt = data_strc[t][2]
        new_file_txt = ""
        for s in procd_txt:
            doc_sents[t] += 1
            for w in s[1]:
                testw = "".join([x for x in w[0] if x not in punc])
                if testw in words:
                    words[testw][t] += 1/doc_sents[t]
                else:
                    words[testw] = [0 for x in range(dim)]
                    words[testw][t] = 1/doc_sents[t]
    return(words)

def create_count_table(data_strc, delim="\t", incl_headers=True, v=True):
    """Function for creating text table of word counts"""
    words = count_words(data_strc)
    table_text =""
    if incl_headers: table_text += delim + delim.join([n[0] for n in data_strc]) + "\n"
    head_words = words.keys()
    for w in head_words:
        table_text += w
        for x in words[w]:
            table_text += delim +str(x)
        table_text += "\n"
    return(table_text)

def create_metadata_table(data_strc, delim="\t", delim2=",", incl_headers=True):
    """Function for creating table of metadata"""
    table_text = ""
    if incl_headers: table_text += delim + delim.join([n[0] for n in data_strc]) + "\n"
    metadata = data_strc[0][1]
    data_labels = list(metadata.keys())
    for label in data_labels:
        table_text += label
        for x in data_strc:           
            table_text += delim + delim2.join(x[1][label])
        table_text += "\n"
    return(table_text)

def create_horiz_table(data_strc, delim="\t", delim2=",", tok_keys=True, write2file=True):
    """Function for creating horizontal table (w/ recordings as rows, vars as cols) of metadata and counts"""
    row_heads = [n[0] for n in data_strc]
    metadata = data_strc[0][1]
    data_labels = list(metadata.keys())
    words = calc_word_density(data_strc)
    head_words = words.keys()
    tok_key_dict = None
    if tok_keys: head_tokens = [f"tok{str(x)}" for x in range(len(head_words))]
    else: head_tokens = head_words
    table_text = f"Recordings{delim}" + delim.join(data_labels) + delim + delim.join(head_tokens) + "\n"
    for rownum in range(len(row_heads)):
        doc_meta = data_strc[rownum][1]
        table_text += f"{row_heads[rownum]}{delim}"
        table_text += delim.join([delim2.join(doc_meta[label]) for label in data_labels]) + delim
        table_text += delim.join([str(words[word][rownum]) for word in head_words])
        #for word in head_words:
        #    table_text += str(words[word][rownum]) + delim
        table_text += "\n"
    #Write table to file
    if delim == "\t": ext = ".tsv"
    elif delim == ",": ext = ".csv"
    if write2file:
        with open("TABLE_H" + ext, "w", encoding="utf-8") as f:
            f.write(table_text)
    #Write token keys to file if using
    if tok_keys:
        head_words = list(head_words)
        tok_key_dict = dict([(head_tokens[x], head_words[x]) for x in range(len(head_words))])
        if write2file:
            tok_key_text = "Key\tOriginal\n"
            tok_key_text += "\n".join([f"{head_tokens[x]}\t{head_words[x]}" for x in range(len(head_words))])
            with open(f"TOKEN_KEYS{ext}", "w", encoding="utf-8") as f:
                f.write(tok_key_text)
    return(tok_key_dict)

def create_horiz_meta_table(dir, delim="\t", delim2=",", w2f=True):
    """Function for creating horizontal table (w/ recordings as rows, vars as cols) of metadata"""
    metas = load_directory(dir)
    eafs = load_directory(dir, ext=".eaf")
    times = [re.findall('TIME_VALUE="(.+)"', eaf[1])[-1] for eaf in eafs]
    table_text = "Recording" + delim + delim.join(list(process_metadata(metas[0][1]).keys())) + delim+ "LastTslot"+"\n"
    table_text += "\n".join(
        [meta[0] + delim + delim.join(
            [delim2.join(process_metadata(meta[1])[k]) for k in list(process_metadata(meta[1]).keys())]
            ) + delim + times[metas.index(meta)] for meta in metas]
        )
    if w2f:
        if delim == "\t": ext = ".tsv"
        elif delim == ",": ext = ".csv"
        with open("META_TABLE" + ext, "w", encoding="utf-8") as f:
            f.write(table_text)
    else: print(table_text)

def create_count_table_file(data_strc, delim="\t"):
    """Function for counting words in data structure and outputting count table"""
    meta_table = create_metadata_table(data_strc, delim=delim)
    count_table = create_count_table(data_strc, delim=delim, incl_headers=False)
    table_text = meta_table + "\n" + count_table    
    if delim == "\t": ext = ".tsv"
    elif delim == ",": ext = ".csv"
    with open("COUNT_TABLE" + ext, "w", encoding="utf-8") as f:
            f.write(table_text)

def word_search_loop(data_strc):
    """Function allowing the data structure to be queried for quick word gloss lookups"""
    inp = ""
    quit_words = "exit,quit,escape".split(",")
    win = 5
    while inp.lower() not in quit_words:
        inp = input("-->")
        search = None
        output = ""
        gloss = re.match("(gloss )(.*)", inp)
        kwic = re.match("(kwic )(.*)", inp)
        tok = re.match("(tok)(.*)", inp)
        if gloss: search = gloss.string[6:]
        elif kwic: search = kwic.string[5:]
        elif tok: search = inp
        #Loop through recordings
        if search != None:
            if 'tok' in search and tokdict != None: search = tokdict[search]
            #Loop through documents
            for x in range(len(data_strc)):
                if gloss:
                    #Loop through sentences
                    for y in range(len(data_strc[x][2])):
                        #Loop through words
                        for z in range(len(data_strc[x][2][y][1])):
                            if search == data_strc[x][2][y][1][z][0]:
                                output += data_strc[x][0] + " Line " + str(y+1)+ " "
                                output += data_strc[x][2][y][1][z][2] + "\n"
                elif kwic:
                    txt = create_raw_prinmi_txt(data_strc[x][2])
                    results = re.findall('(.{,50})('+search+')(.{,50})',txt)
                    if len(results)>0:
                        output += "\n".join([data_strc[x][0]+ " "+" ".join(y) for y in results]) + "\n"
            output = output[:-1]
            if tok: output = search
            print(output)

ds = create_data_structure('C:/Users/cbech/Desktop/Northern Prinmi Project/Northern-Prinmi-Project/preprocessed/txt', 'C:/Users/cbech/Desktop/Northern Prinmi Project/Northern-Prinmi-Project/meta')
tokdict = create_horiz_table(ds, write2file=False)
word_search_loop(ds)

#p = "C:/Users/cbech/Desktop/Northern Prinmi Project/Northern-Prinmi-Project/preprocessed/eaf-meta"
#p = "D:/Northern Prinmi Data/wav-eaf-meta"
#create_horiz_meta_table(p)

#print(ds[0])
#create_count_table_file(ds)
#print(calc_word_density(ds))