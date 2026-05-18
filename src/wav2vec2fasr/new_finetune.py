from datasets import load_from_disk, load_metric, Audio
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)

import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import os
from pathlib import Path


def main_program( 
        data_dir,
        output_dir,
        vocab_dir=None,
        learn_rate=3e-4,
        batches=1,
        grdacc_steps=2,
        epochs=30,
        mixed_precision=True,
        use_cpu=False,
        no_cuda=False,
        atn_dout=0.1,
        hid_dout=0.1,
        ft_proj_dout=0.0,
        msk_tm_prob=0.05,
        ldrop=0.1,
        w2v2_model="facebook/wav2vec2-large-xlsr-53",
        save_steps = 1000,
        eval_steps = 100,
        logging_steps = 10,
        warmup_steps = 500,
        max_steps = 5000):

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    data_train = data_dir.joinpath("training/")
    data_test = data_dir.joinpath("testing/")
    if vocab_dir == None:
        vocab_dir = output_dir
    mod_dir = output_dir#os.path.join(output_dir, "model/")
    if os.path.exists(mod_dir):
        logging.debug(f"Output directory {mod_dir} exists")
    else:
        logging.debug(f"Creating output directory {mod_dir}")
        os.mkdir(mod_dir)
    if "facebook" in w2v2_model:
        model_type = "wav2vec2"
        from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
        if w2v2_model == "facebook/mms-1b-all":
            just_adapter = True
        else:
            just_adapter = False
    elif "openai" in w2v2_model:
        model_type = "whisper"
        from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
        just_adapter = False

    logging.debug(f"Loading training data from {data_train}")
    np_train_ds = load_from_disk(data_train)

    logging.debug(f"Loading test data from {data_test}")
    np_test_ds = load_from_disk(data_test)


    logging.debug("tokenizer setup")
    if model_type == "wav2vec2":
        tokenizer = Wav2Vec2CTCTokenizer(vocab_dir.joinpath("vocab.json"), 
                                        unk_token="[UNK]", 
                                        pad_token="[PAD]", 
                                        word_delimiter_token="|")
    elif model_type == "whisper":
        tokenizer = WhisperTokenizer.from_pretrained(w2v2_model,
                                                     language="German",
                                                     task="transcribe")
    logging.debug("extractor setup")
    if model_type == "wav2vec2":
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, 
                                                    sampling_rate=16000, 
                                                    padding_value=0.0, 
                                                    do_normalize=True, 
                                                    return_attention_mask=True)
    elif model_type == "whisper":
        feature_extractor = WhisperFeatureExtractor.from_pretrained(w2v2_model)
    logging.debug("processor setup")
    if model_type == "wav2vec2": 
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, 
                                    tokenizer=tokenizer)
    elif model_type == "whisper":
        processor = WhisperProcessor(feature_extractor=feature_extractor,
                                     tokenizer=tokenizer)
    def prepare_dataset(batch):
        audio = batch["audio"]
        if model_type == "wav2vec2":
            batch["input_values"] = processor(audio["array"], 
                                            sampling_rate=audio["sampling_rate"]).input_values[0]
            with processor.as_target_processor():
                batch["labels"] = processor(batch["transcript"]).input_ids
        elif model_type == "whisper":
            batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
            batch["labels"] = tokenizer(batch["transcript"]).input_ids
        return batch

    logging.debug("training prep")
    np_train_ds = np_train_ds.map(prepare_dataset, remove_columns=np_train_ds.column_names)

    logging.debug("test prep")
    np_test_ds = np_test_ds.map(prepare_dataset, remove_columns=np_test_ds.column_names)

    if model_type == "wav2vec2":
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
    elif model_type=="whisper":
        @dataclass
        class DataCollatorSpeechSeq2SeqWithPadding:
            processor: Any
            decoder_start_token_id: int

            def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
                # split inputs and labels since they have to be of different lengths and need different padding methods
                # first treat the audio inputs by simply returning torch tensors
                input_features = [{"input_features": feature["input_features"]} for feature in features]
                batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

                # get the tokenized label sequences
                label_features = [{"input_ids": feature["labels"]} for feature in features]
                # pad the labels to max length
                labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

                # replace padding with -100 to ignore loss correctly
                labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

                # if bos token is appended in previous tokenization step,
                # cut bos token here as it's append later anyways
                if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                    labels = labels[:, 1:]

                batch["labels"] = labels

                return batch

    if model_type == "wav2vec2":
        logging.debug("collator prep")
        data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    logging.debug("loading wer and cer")
    wer_metric = load_metric("wer", trust_remote_code=True)
    cer_metric = load_metric("cer", trust_remote_code=True)

    def compute_metrics(pred):
        if model_type == "wav2vec2":
            pred_logits = pred.predictions
            pred_ids = np.argmax(pred_logits, axis=-1)

            pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

            pred_str = processor.batch_decode(pred_ids)
            # we do not want to group tokens when computing the metrics
            label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        elif model_type == "whisper":
            pred_ids = pred.predictions
            label_ids = pred.label_ids

             # replace -100 with the pad_token_id
            label_ids[label_ids == -100] = tokenizer.pad_token_id

            # we do not want to group tokens when computing the metrics
            pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer, "wer": wer}

    #if mixed_precision: torch_dtype = torch.float16
    #else: torch_dtype = torch.float32

    logging.debug("Downloading model")
    if model_type == "wav2vec2":
        model = Wav2Vec2ForCTC.from_pretrained(
            w2v2_model, 
            # Experimental feature 6-21-24
            #torch_dtype=torch_dtype,
            attention_dropout=atn_dout,#0.1,
            hidden_dropout=hid_dout,#0.1,
            feat_proj_dropout=ft_proj_dout,#0.0,
            mask_time_prob=msk_tm_prob,#0.05,
            layerdrop=ldrop,#0.1,
            ctc_loss_reduction="mean", 
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer),
            ignore_mismatched_sizes=True
        )

        if just_adapter:
            logging.debug("initializing adapter")
            #Initialize adapter layers
            model.init_adapter_layers()
            logging.debug("freezing model besides adapter layers")
            #Freeze all model layers besides adapter layers
            model.freeze_base_model()
            adapter_weights = model._get_adapters()
            for param in adapter_weights.values():
                param.requires_grad = True
            save_steps = 200
            eval_steps = save_steps
            logging_steps = 10
            warmup_steps = 100
        else:
            logging.debug("freezing extractor")
            model.freeze_feature_extractor()
    elif model_type == "whisper":
        model = WhisperForConditionalGeneration.from_pretrained(w2v2_model)
        model.generation_config.language = "german"
        model.generation_config.task = "transcribe"
        #model.generation_config.forced_decoder_ids = None
        
        logging.debug("collator prep")
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor,
                                             decoder_start_token_id=model.config.decoder_start_token_id)

    logging.debug("gradient checkpointing")
    model.gradient_checkpointing_enable()

    logging.debug("Setting up training args")
    if model_type == "wav2vec2":
        training_args = TrainingArguments(
            output_dir = output_dir,
            group_by_length=True,
            per_device_train_batch_size=batches,#1,
            gradient_accumulation_steps=grdacc_steps,#2,
            eval_strategy="steps",
            logging_strategy="steps",
            no_cuda = no_cuda,
            use_cpu= use_cpu,
            fp16=mixed_precision,#True,
            save_steps=save_steps,
            eval_steps=eval_steps,
            logging_steps=logging_steps,
            learning_rate=learn_rate,#3e-4,
            warmup_steps=warmup_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_cer",
            greater_is_better=False,
            save_total_limit=2,
            )
        if max_steps != None: training_args.max_steps = max_steps#5000
        else: training_args.num_train_epochs = epochs#30
    elif model_type == "whisper":
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,  # change to a repo name of your choice
            per_device_train_batch_size=batches,#16
            gradient_accumulation_steps=grdacc_steps, #1 increase by 2x for every 2x decrease in batch size
            learning_rate=learn_rate,#1e-5,
            warmup_steps=warmup_steps,#500,
            #gradient_checkpointing=True,
            fp16=mixed_precision,#True,
            use_cpu= use_cpu,
            eval_strategy="steps",
            save_total_limit=2,
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=save_steps,#1000,
            eval_steps=eval_steps,#1000,
            logging_steps=logging_steps,#logging_steps,#25,
            #report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="eval_cer",
            greater_is_better=False,
            #push_to_hub=True,
        )
        if max_steps != None: training_args.max_steps = max_steps#5000
        else: training_args.num_train_epochs = epochs#30



    logging.debug("setting up trainer")
    if model_type == "wav2vec2":
        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=np_train_ds,
            eval_dataset=np_test_ds,
            tokenizer=processor.feature_extractor,
        )
    elif model_type == "whisper":
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=np_train_ds,
            eval_dataset=np_test_ds,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=processor.feature_extractor,
        ) 
    # print("Running trainer...")

    logging.debug("training")
    trainer.train()
    logging.debug("saving model")
    trainer.save_model(mod_dir)
    #New test line 2.9.23
    processor.save_pretrained(output_dir)

    if just_adapter:
        from safetensors.torch import save_file as safe_save_file
        from transformers.models.wav2vec2.modeling_wav2vec2 import WAV2VEC2_ADAPTER_SAFE_FILE
        import os

        adapter_file = WAV2VEC2_ADAPTER_SAFE_FILE.format("zzz")
        adapter_file = os.path.join(training_args.output_dir, adapter_file)

        safe_save_file(model._get_adapters(), adapter_file, metadata={"format": "pt"})


    # print("Trainer saved!")


if __name__ == "__main__":
    #main_program()
    if torch.cuda.is_available():
        main_program()
    else:
        print("no cuda, attempting without")
        main_program(mixed_precision=False)
