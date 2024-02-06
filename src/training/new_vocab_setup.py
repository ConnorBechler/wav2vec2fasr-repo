from datasets import load_from_disk
import logging
import os
import json

def setup_vocab(
        home = None,
        project_dir = "npp_asr",
        output_dir = "",
        data_dir=None):
    """
    Function for setting up vocab for wav2vec 2.0 training

    Args:
        home (str) : root system directory
        project_dir (str) : name of project directory
        output_dir (str) : output directory for vocab file
        data_dir (str) : directory with named training and testing directories
    
    Output:
        vocab.json file with full wav2vec 2.0 vocabulary
    """
    
    if home == None: home = os.environ["HOME"]
    full_project = os.path.join(home, project_dir)
    if output_dir == "": output_dir = os.path.join(full_project, output_dir)
    if data_dir == None: data_dir = os.path.join(output_dir, "data/")
    data_train = os.path.join(data_dir, "training/")
    data_test = os.path.join(data_dir, "testing/")
    
    if os.path.exists(data_train):
        logging.debug("Training directory exists")
    if os.path.exists(data_test):
        logging.debug("Testing directory exists")
    
    
    logging.debug(f"Loading training data from {data_train}")
    np_train_ds = load_from_disk(data_train)
    
    logging.debug(f"Loading test data from {data_test}")
    np_test_ds = load_from_disk(data_test)
    
    def extract_all_chars(batch):
      all_text = " ".join(batch["transcript"])
      vocab = list(set(all_text))
      return {"vocab": [vocab], "all_text": [all_text]}
    
    logging.debug("Extracting training chars")
    vocab_train = np_train_ds.map(extract_all_chars, 
                                  batched=True, 
                                  batch_size=-1, 
                                  keep_in_memory=True, 
                                  remove_columns=np_train_ds.column_names)
    
    logging.debug("Extracting testing chars")
    vocab_test = np_test_ds.map(extract_all_chars, 
                                batched=True, 
                                batch_size=-1,
                                keep_in_memory=True, 
                                remove_columns=np_test_ds.column_names)
    
    
    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    
    out_json = os.path.join(output_dir, "vocab.json")
    
    with open(out_json, 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)
