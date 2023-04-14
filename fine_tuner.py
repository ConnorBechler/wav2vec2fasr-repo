"""Fine Tuning Tool

Tool for fine tuning wav2-vec2 XLS-R model on data
Substantial borrowing from https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_Tune_XLSR_Wav2Vec2_on_Turkish_ASR_with_%F0%9F%A4%97_Transformers.ipynb
Requires Python version 3.9.13 or below

@Author: Connor Bechler
@Date: Fall 2022
"""

from datasets import load_from_disk, load_metric, Audio
import re
import numpy as np
import json


import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer


import os
project_dir = "npp_asr"
full_project = os.path.join(os.environ["HOME"], project_dir)
data_dir = os.path.join(full_project, "npp_data")
data_train = os.path.join(data_dir, "training")
data_test = os.path.join(data_dir, "testing")

#np_full_ds = load_from_disk(p)
np_train_full_ds = load_from_disk(p_train)
np_test_full_ds = load_from_disk(p_test)
np_train_full_ds = np_train_full_ds.select([i for i in range(len(np_train_full_ds['segment'])) if i != 243])

np_train_full_ds = np_train_full_ds.remove_columns(["from_file", "segment"])
np_test_full_ds = np_test_full_ds.remove_columns(["from_file", "segment"])

chars_to_ignore_regex = '[\,\?\.\!\;\:\"\“\%\‘\”\�\。\n\(\/\！\)\）\，]'
tone_regex = '[\¹\²\³\⁴\⁵\-]'
nontone_regex = '[^\¹\²\³\⁴\⁵ \-]'
diacritics = "ʲʷ ʰʷ ̥ ʰ ʲ ʰ ̃ ʷ".split(" ")
trips = ['sʰʷ', 'ʈʰʷ', 'ʂʰʷ', 'tʰʷ', 'qʰʷ', 'nʲʷ', 'kʰʷ', 'lʲʷ', 'ɕʰʷ', 'tʲʷ']
doubs = ['ɕʰ', 'n̥', 'qʷ', 'ɬʷ', 'qʰ', 'xʲ', 'xʷ', 'ɨ̃', 'ʈʷ', 'ʈʰ', 'ŋʷ', 'ʑʷ', 'mʲ', 'dʷ', 'ĩ', 'pʰ', 'ɕʷ', 
'tʷ', 'rʷ', 'lʲ', 'ɡʷ', 'bʲ', 'pʲ', 'tʲ', 'zʷ', 'ɬʲ', 'ʐʷ', 'dʲ', 'ɑ̃', 'lʷ', 'sʷ', 'ə̃', 'kʷ', 'æ̃', 'ɖʷ', 'm̥', 
'kʰ', 'ʂʷ', 'õ', 'ʂʰ', 'sʰ', 'r̥', 'nʲ', 'tʰ', 'jʷ', "õ", "ĩ"]
rep_trips = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
rep_doubs = "ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞⓟⓠⓡⓢⓣⓤⓥⓦⓧⓨⓩⒶⒷⒸⒹⒺⒻⒼⒽⒾⒿⓀⓁⓂⓃⓄⓅⓆⓇⓈⓉⓊⓋⓌⓍⓎⓏ"
tone_chars = "¹ ² ³ ⁵".split(" ")
tones = ["²²","³²", "³⁵", "⁵⁵", "⁵²","⁵¹"]
rep_tones = "1234567890"
#Question: currently stripping hyphen from nontone transcription, but treating it as syllable boundary
#(subbing with space) for tone transcription; is this the right move?
#Diacritics handled by combining them with consonants and vowels into abstract characters

def remove_special_characters(batch):
    batch["transcript"] = re.sub(chars_to_ignore_regex, '', batch["transcript"]).lower() + " "
    return batch

def remove_tone_chars(batch):
    batch["transcript"] = re.sub(tone_regex, '', batch["transcript"])
    return batch

def remove_nontone_chars(batch):
    batch["transcript"] = re.sub(nontone_regex, '', batch["transcript"])
    batch["transcript"] = re.sub('\-', ' ', batch["transcript"])
    batch["transcript"] = re.sub('[\¹\²\³\⁴\⁵][\¹\²\³\⁴\⁵]()[\¹\²\³\⁴\⁵][\¹\²\³\⁴\⁵]', ' ', batch['transcript'])
    batch["transcript"] = re.sub('  ', ' ', batch["transcript"])
    return batch

def convert_tones_to_onechar(batch):
    for x in range(len(tones)):
        batch["transcript"] = re.sub(tones[x], rep_tones[x], batch["transcript"])
    #There are single tone numbers in the transcripts (superscripts without pairs)
    ##I AM ASSUMING THESE ARE TRANSCRIPTIONS ERRORS AND AM REMOVING THEM FROM THE DATASET
    for x in range(len(tone_chars)):
        batch["transcript"] = re.sub(tone_chars[x], "", batch["transcript"])
    return batch
    
def convert_onechar_to_tones(batch):
    for x in range(len(tones)):
        batch["transcript"] = re.sub(rep_tones[x], tones[x], batch["transcript"])
    return batch

def convert_dcrts_to_phones(batch):    
    for x in range(len(trips)):
        batch["transcript"] = re.sub(trips[x], rep_trips[x], batch["transcript"])
    for x in range(len(doubs)):
        batch["transcript"] = re.sub(doubs[x], rep_doubs[x], batch["transcript"])
    return batch

def convert_phones_to_dcrts(text):
    for x in range(len(trips)):
        text = re.sub(rep_trips[x], trips[x], text)
    for x in range(len(doubs)):
        text = re.sub(rep_doubs[x], doubs[x], text)
    return text

#Make sure audio array is nd array instead of list
np_train_full_ds = np_train_full_ds.cast_column("audio", Audio())
np_test_full_ds = np_test_full_ds.cast_column("audio", Audio())
print("Audio columns cast to audio format")

#Remove special characters from data
np_train_full_ds = np_train_full_ds.map(remove_special_characters)
np_test_full_ds = np_test_full_ds.map(remove_special_characters)
print("Special characters removed")

#np_train_tones_ds = np_train_full_ds.map(remove_nontone_chars)
#np_test_tones_ds = np_test_full_ds.map(remove_nontone_chars)
#print(np_test_tones_ds['transcript'][:25])

#np_train_tones_ds2 = np_train_tones_ds.map(convert_tones_to_onechar)
#np_test_tones_ds2 = np_test_tones_ds.map(convert_tones_to_onechar)
#print("Tones converted to one char")

np_train_words_ds = np_train_full_ds.map(remove_tone_chars)
np_test_words_ds = np_test_full_ds.map(remove_tone_chars)
print("Tone characters removed")

np_train_words_ds = np_train_words_ds.map(convert_dcrts_to_phones)
np_test_words_ds = np_test_words_ds.map(convert_dcrts_to_phones)
print("Diacritics combined to phones")

np_train_ds = np_train_words_ds
np_test_ds = np_test_words_ds

def extract_all_chars(batch):
  all_text = " ".join(batch["transcript"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = np_train_ds.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=np_train_ds.column_names)
vocab_test = np_test_ds.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=np_test_ds.column_names)

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(vocab_list)}
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

print("Vocabulary set up")


tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")


feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)


processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def prepare_dataset(batch):
    audio = batch["audio"]

    #from transformers import Wav2Vec2CTCTokenizer

    #tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

    #from transformers import Wav2Vec2FeatureExtractor

    #feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

    #from transformers import Wav2Vec2Processor
    
    #processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcript"]).input_ids
    return batch

np_train_ds = np_train_ds.map(prepare_dataset, remove_columns=np_train_ds.column_names, num_proc=4)
np_test_ds = np_test_ds.map(prepare_dataset, remove_columns=np_test_ds.column_names, num_proc=4)

print("Dataset prepared")



@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

print("Data collator set up")

wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

print("Metric set up")

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53", 
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)

model.freeze_feature_extractor()

model.gradient_checkpointing_enable()

print("Model set up")


training_args = TrainingArguments(
  # output_dir="/content/gdrive/MyDrive/wav2vec2-large-xlsr-turkish-demo",
  output_dir=p+"/output",#"./wav2vec2-large-xlsr-turkish-demo",
  group_by_length=True,
  per_device_train_batch_size=8,
  gradient_accumulation_steps=2,
  evaluation_strategy="steps",
  num_train_epochs=30,
  fp16=True,
  save_steps=100,
  eval_steps=100,
  logging_steps=10,
  learning_rate=3e-4,
  warmup_steps=500,
  #save_total_limit=10,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=np_train_ds,
    eval_dataset=np_test_ds,
    tokenizer=processor.feature_extractor,
)
print("Running trainer...")

trainer.train()
trainer.save_model(p+"/output/model")

print("Trainer saved!")


