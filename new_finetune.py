from datasets import load_from_disk, load_metric, Audio
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)

import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer

import os


def main_program(project_dir = "npp_asr", 
        output_dir="", 
        data_dir=None, 
        vocab_dir=None,
        learn_rate=3e-4,
        batches=1,
        grdacc_steps=2,
        epochs=30,
        mixed_precision=True,
        atn_dout=0.1,
        hid_dout=0.1,
        ft_proj_dout=0.0,
        msk_tm_prob=0.05,
        ldrop=0.1,
        w2v2_model="facebook/wav2vec2-large-xlsr-53"):
            
    project_dir = project_dir#"npp_asr"
    full_project = os.path.join(os.environ["HOME"], project_dir)
    if output_dir == "": 
        output_dir = os.path.join(full_project, "output/" + output_dir)
    if data_dir == None: 
        data_dir = output_dir+"/data"
    data_dir = os.path.join(full_project, data_dir)
    data_train = os.path.join(data_dir, "training/")
    data_test = os.path.join(data_dir, "testing/")
    if vocab_dir == None: 
        vocab_dir = output_dir
    mod_dir = output_dir#os.path.join(output_dir, "model/")
    if os.path.exists(mod_dir):
        logging.debug(f"Output directory {mod_dir} exists")
    else:
        logging.debug(f"Creating output directory {mod_dir}")
        os.mkdir(mod_dir)


    logging.debug(f"Loading training data from {data_train}")
    np_train_ds = load_from_disk(data_train)

    logging.debug(f"Loading test data from {data_test}")
    np_test_ds = load_from_disk(data_test)


    logging.debug("tokenizer setup")
    tokenizer = Wav2Vec2CTCTokenizer(vocab_dir + "/vocab.json", 
                                    unk_token="[UNK]", 
                                    pad_token="[PAD]", 
                                    word_delimiter_token="|")

    logging.debug("extractor setup")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, 
                                                sampling_rate=16000, 
                                                padding_value=0.0, 
                                                do_normalize=True, 
                                                return_attention_mask=True)

    logging.debug("processor setup")
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, 
                                tokenizer=tokenizer)


    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(audio["array"], 
                                        sampling_rate=audio["sampling_rate"]).input_values[0]
        
        with processor.as_target_processor():
            batch["labels"] = processor(batch["transcript"]).input_ids
        return batch

    logging.debug("training prep")
    np_train_ds = np_train_ds.map(prepare_dataset, remove_columns=np_train_ds.column_names, num_proc=4)

    logging.debug("test prep")
    np_test_ds = np_test_ds.map(prepare_dataset, remove_columns=np_test_ds.column_names, num_proc=4)

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

    logging.debug("collator prep")
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    logging.debug("loading wer and cer")
    wer_metric = load_metric("wer")
    cer_metric = load_metric("cer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer, "wer": wer}

    logging.debug("Downloading model")
    model = Wav2Vec2ForCTC.from_pretrained(
        w2v2_model, 
        attention_dropout=atn_dout,#0.1,
        hidden_dropout=hid_dout,#0.1,
        feat_proj_dropout=ft_proj_dout,#0.0,
        mask_time_prob=msk_tm_prob,#0.05,
        layerdrop=ldrop,#0.1,
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )

    logging.debug("freezing extractor")
    model.freeze_feature_extractor()

    logging.debug("gradient checkpointing")
    model.gradient_checkpointing_enable()

    logging.debug("Setting up training args")
    
    training_args = TrainingArguments(
    # output_dir="/content/gdrive/MyDrive/wav2vec2-large-xlsr-turkish-demo",
    output_dir = output_dir,
    group_by_length=True,
    per_device_train_batch_size=batches,#1,
    gradient_accumulation_steps=grdacc_steps,#2,
    evaluation_strategy="steps",
    num_train_epochs=epochs,#30,
    fp16=mixed_precision,#True,
    save_steps=1000,
    eval_steps=100,
    logging_steps=10,
    learning_rate=learn_rate,#3e-4,
    warmup_steps=500,
    #save_total_limit=10,
    )

    logging.debug("setting up trainer")
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=np_train_ds,
        eval_dataset=np_test_ds,
        tokenizer=processor.feature_extractor,
    )
    # print("Running trainer...")

    logging.debug("training")
    trainer.train()
    logging.debug("saving model")
    trainer.save_model(mod_dir)
    #New test line 2.9.23
    processor.save_pretrained(output_dir)

    # print("Trainer saved!")


if __name__ == "__main__":
    #main_program()
    if torch.cuda.is_available():
        main_program()
    else:
        print("no cuda, attempting without")
        main_program(mixed_precision=False)
