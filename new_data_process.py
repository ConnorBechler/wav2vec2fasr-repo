import os
import logging
import re

from datasets import load_from_disk, Audio
from prinmitext import chars_to_ignore_regex, tone_regex, nontone_regex, trips, doubs
from prinmitext import rep_trips, rep_doubs, tones, rep_tones, rep_combs

def process_data(
        project_dir = "npp_asr", 
        data_dir="data", 
        output_dir="data_processed",
        remove_tones=False,
        remove_nontones=False,
        combine_diac=False,
        combine_tones=False,
        remove_hyphens=False):
            
    logging.basicConfig(level=logging.DEBUG)
    
    project_dir = project_dir #"npp_asr"
    full_project = os.path.join(os.environ["HOME"], project_dir)
    data_dir = os.path.join(full_project, data_dir)
    data_train = os.path.join(data_dir, "training/dataset/")
    data_test = os.path.join(data_dir, "testing/dataset/")
    data_out = os.path.join(output_dir, "data/")
    dtr_out = os.path.join(data_out, "training/")
    dte_out = os.path.join(data_out, "testing/")
    
    if os.path.exists(data_train):
        logging.debug("Training directory exists")
    if os.path.exists(data_test):
        logging.debug("Testing directory exists")
    
    
    logging.debug(f"Loading training data from {data_train}")
    np_train_full_ds = load_from_disk(data_train)
    
    logging.debug(f"Loading test data from {data_test}")
    np_test_full_ds = load_from_disk(data_test)
    
    selector = [i for i in range(len(np_train_full_ds['segment'])) if i != 243]
    
    np_train_full_ds = np_train_full_ds.select(selector)
    
    np_train_full_ds = np_train_full_ds.remove_columns(["from_file", "segment"])
    np_test_full_ds = np_test_full_ds.remove_columns(["from_file", "segment"])
    
    #np_train_full_ds = np_train_full_ds.select(range(81))
    #np_test_full_ds = np_test_full_ds.select(range(21))
    
    def remove_special_characters(batch):
        batch["transcript"] = re.sub(
            chars_to_ignore_regex, '', batch["transcript"]).lower() + " "
        for k in rep_combs:
            batch['transcript'] = batch['transcript'].replace(k, rep_combs[k])
        # There are errors in the transcripts, single and triple superscript tones (with the triples lacking movement, i.e. 555)
        # The following lines fix these errors by: 
        # A) coverting the 555 to high level tone 55 as these appear to show the same tone contour
        # B) Removing sinle tone superscripts, as leaving a tone off is an immediately visible error, easier to see than an incorrect tone guess
        if "⁵⁵⁵" in batch["transcript"]:
            batch["transcript"] = re.sub("⁵⁵⁵", "⁵⁵", batch["transcript"])
        batch["transcript"] = re.sub("(?<!¹|²|³|⁵)[¹²³⁵] ", " ", batch["transcript"])
        return batch
    
    def remove_hyphen_char(batch):
        batch["transcript"] = re.sub('-', '', batch['transcript'])
        return batch
    
    def remove_tone_chars(batch):
        batch["transcript"] = re.sub(tone_regex, '', batch["transcript"])
        return batch
    
    def remove_nontone_chars(batch):
        batch["transcript"] = re.sub(nontone_regex, '', batch["transcript"])
        batch["transcript"] = re.sub('\-', ' ', batch["transcript"])
        batch["transcript"] = re.sub(
            '[\¹\²\³\⁴\⁵][\¹\²\³\⁴\⁵]()[\¹\²\³\⁴\⁵][\¹\²\³\⁴\⁵]', ' ', batch['transcript'])
        batch["transcript"] = re.sub('  ', ' ', batch["transcript"])
        return batch
    
    def convert_tones_to_onechar(batch):
        for x in range(len(tones)):
            batch["transcript"] = re.sub(
                tones[x], rep_tones[x], batch["transcript"])
        #Two lines below no longer necessary
        #for x in range(len(tone_chars)):
        #    batch["transcript"] = re.sub(tone_chars[x], "", batch["transcript"])
        return batch
    
    def convert_onechar_to_tones(batch):
        for x in range(len(tones)):
            batch["transcript"] = re.sub(
                rep_tones[x], tones[x], batch["transcript"])
        return batch
    
    def convert_dcrts_to_phones(batch):
        for x in range(len(trips)):
            batch["transcript"] = re.sub(
                trips[x], rep_trips[x], batch["transcript"])
        for x in range(len(doubs)):
            batch["transcript"] = re.sub(
                doubs[x], rep_doubs[x], batch["transcript"])
        return batch
    
    def convert_phones_to_dcrts(text):
        for x in range(len(trips)):
            text = re.sub(rep_trips[x], trips[x], text)
        for x in range(len(doubs)):
            text = re.sub(rep_doubs[x], doubs[x], text)
        return text
    
    logging.debug("preprocessing training data")
    np_train_full_ds = (np_train_full_ds
                         .cast_column("audio", Audio())
                         .map(remove_special_characters)
                         )
    
    logging.debug("preprocessing testing data")
    np_test_full_ds = (np_test_full_ds
                       .cast_column("audio", Audio())
                       .map(remove_special_characters)
                       )
    
    if remove_hyphens:
        logging.debug("removing hyphens")
        np_train_full_ds = np_train_full_ds.map(remove_hyphen_char)
        np_test_full_ds = np_test_full_ds.map(remove_hyphen_char)
    
    if remove_tones:
        logging.debug("removing tones")
        np_train_full_ds = np_train_full_ds.map(remove_tone_chars)
        np_test_full_ds = np_test_full_ds.map(remove_tone_chars)
        
    if remove_nontones:
        logging.debug("removing nontones")
        np_train_full_ds = np_train_full_ds.map(remove_nontone_chars)
        np_test_full_ds = np_test_full_ds.map(remove_nontone_chars)
    
    if combine_diac:
        logging.debug("convert diacritics to phones")
        np_train_full_ds = np_train_full_ds.map(convert_dcrts_to_phones)
        np_test_full_ds = np_test_full_ds.map(convert_dcrts_to_phones)
    
    if combine_tones:
        logging.debug("convert tones to one character")
        np_train_full_ds = np_train_full_ds.map(convert_tones_to_onechar)
        np_test_full_ds = np_test_full_ds.map(convert_tones_to_onechar)
                       
    logging.debug("Removing empty entries")
    np_train_full_ds = np_train_full_ds.filter(lambda example: len(example['transcript']) > 1)
    np_test_full_ds = np_test_full_ds.filter(lambda example: len(example['transcript']) > 1)
    
    
    
    if os.path.exists(data_out):
        logging.debug("Data output directory exists")
    else:
        os.mkdir(data_out)
    
    if os.path.exists(dtr_out):
        logging.debug("Training output directory exists")
    else:
        os.mkdir(dtr_out)
        
    if os.path.exists(dte_out):
        logging.debug("Testing output directory exists")
    else:
        os.mkdir(dte_out)
    
    np_train_full_ds.save_to_disk(output_dir+ "/data/training")
    np_test_full_ds.save_to_disk(output_dir + "/data/testing")

if __name__ == "__main__":
    process_data()
    