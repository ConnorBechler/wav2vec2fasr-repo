# wav2vec2fasr Package

A set of tools for fine-tuning and applying wav2vec 2.0 models for forced alignment and speech recognition, first developed specifically for assisting with the transcription of language documentation materials from Northern Prinmi.

## Requirements

wav2vec2fasr requires python versions between 3.8 and 3.11, although it may work for others if you want to wrestle with the dependencies.

It also requires about 1.5 GB of disk space, as transformers is a bulky package.

Finetuning your own models, while technically possible on a normal PC, is generally speaking not feasible without a relatively nice GPU. With the right GPU and CUDA installed it's possible on a higher end PC, but getting everything in place is non-trivial (a massive pain). I'd recommend trying to finetune models in either a google colab or on a high performance computing cluster, if you have access.

I've only tested these modules extensively on .wav files, although .mp3 will probably work.

## Installation

Clone the repo to get started. To actually make it usable, I recommend using [poetry](https://python-poetry.org/docs/#installation) with the command `poetry install`, as it will handle dependencies and the local environment for you. You can also use `pip install git+https://github.com/ConnorBechler/wav2vec2fasr-repo/`, but no guarantees it will perform optimally.

## Basic Use

### Finetuning

1. Separate your audio corpus/dataset into a training directory and a testing directory, with audio files and transcripts (eafs or TextGrids)
2. Use `audio_processor.py` to chunk each directory into training and testing huggingface datasets, respectively
3. Create a tokenization scheme .tsv file, following the instructions [here](https://github.com/ConnorBechler/wav2vec2fasr-repo/wiki/Tokenization-Schemes#creating-a-tokenization-scheme-file), and put its name or path in src/wav2vec2fasr/resources/config.json. If you just add its name, be sure the tsv is also in the resources folder.
4. Create an evaluation settings .json file, following the instructions [here](), and put its name or path in src/wav2vec2fasr/resources/config.json. If you just add its name, be sure the json is also in the resources folder.
5. Run `full_finetune.py` from the command line with the hyper-parameters of your choice.
6. There will probably be a bug somewhere, but if not, you will have models! The evaluation logs should give you some idea of how well they actually trained.

### Automatic Speech Recognition

1. Have the path to either your own finetuned or downloaded wav2vec2 model / the name of one on the huggingface hub, like `facebook/wav2vec2-base-960h` for English
2. Use one of the several functions provided in `forcedalignment.py` with your model, directing it at your audio.
    1. transcribe_audio for single files
    2. transcribe_audio_dir for directories
3. This will produce both phrase, word, and phone alignments in either .eaf or .TextGrid format

### Forced Alignment of Existing Transcripts

1. Have the path to either your own finetuned or downloaded wav2vec2 model / the name of one on the huggingface hub, like `facebook/wav2vec2-base-960h` for English
2. Use one of the two `forcedalignment.py` functions on a directory of audio files with .eaf or TextGrid transcripts
    1. align_transcriptions for single files
    2. align_transcription_dir for directories
3. Make sure you only have one tier per speaker! Otherwise, check the function options to select which tiers to align.
4. This will produce word and phone alignments in either .eaf or .TextGrid format, with the files by default named after the audio.

## Questions

Please feel free to open a thread on issues if you run into any trouble!

## Thanks

My great personal thanks to [Josef Fruehwald](https://github.com/JoFrhwld) for his extensive assistance with the underlying research project that resulted in this repo, as well as his direct contributions to conceptualizing and implementing specific elements.

