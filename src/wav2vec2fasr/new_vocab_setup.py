from datasets import load_from_disk
import logging
import os
from pathlib import Path
import json

def setup_vocab(
        data_dir,
        output_dir
        ):
    """
    Function for setting up vocab for wav2vec 2.0 training

    Args:
        data_dir (str) : directory with named training and testing directories
        run_dir (str) : output directory for vocab file
    
    Output:
        vocab.json file with full wav2vec 2.0 vocabulary
    """
    
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    data_train = data_dir.joinpath("training/")
    data_test = data_dir.joinpath("testing/")
    
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
    
    if not(os.path.exists(output_dir)):
        os.mkdir(output_dir)
    out_json = output_dir.joinpath("vocab.json")
    
    with open(out_json, 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)
