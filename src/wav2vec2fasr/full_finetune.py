import os
from pathlib import Path
import logging

from wav2vec2fasr.new_data_process import process_data
from wav2vec2fasr.new_vocab_setup import setup_vocab
from wav2vec2fasr.new_finetune import main_program
from wav2vec2fasr.evaluate import main_program as eval_program
from wav2vec2fasr.orthography import Tokenization_Scheme
#from wav2vec2fasr.orthography import load_tokenization, load_config

#Arguments stuff added with help from https://machinelearningmastery.com/command-line-arguments-for-your-python-script/
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
warnings.simplefilter("ignore")

import datetime
cur_time = str(datetime.datetime.now()).replace(" ", "_")

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-n", "--run_name", default="model_"+cur_time, help="Run name")
parser.add_argument("-d", "--data_dir", default="./data/", help="The path from your project to the folder with the huggingface datasets")
parser.add_argument("-o", "--output_dir", default="./output/", help="The path where this run's processed data, vocab and model will be created")
parser.add_argument("--cpu", action="store_true", help="Run without mixed precision")
parser.add_argument("--tokenization", default="None", help="Name of tokenization file to load tokenization from")
parser.add_argument("--eval_settings", default="None", help="Path to eval_settings .json file")
parser.add_argument("--comb_tones", action="store_true", help="Combine tone pairs")
parser.add_argument("--comb_diac", action="store_true", help="Combine diacritic character clusters")
parser.add_argument("--no_tones", action="store_true", help="Remove tones from transcripts")
parser.add_argument("--no_hyph", action="store_true", help="Remove hyphens from transcripts")
parser.add_argument("-r", "--learning_rate", default = 3e-4, type=float, help="Learning rate")
parser.add_argument("-b", "--batch_size", default = 1, type=int, help="Number of batches per device")
parser.add_argument("-g", "--grdacc_steps", default = 2, type=int, help="Number of gradient accumulation steps")
parser.add_argument("-e","--epochs", default=30, type=int, help="Number of training epochs")
parser.add_argument("--atn_dout", default=0.1, type=float, help="Attention dropout")
parser.add_argument("--hid_dout", default=0.1, type=float, help="Hidden dropout")
parser.add_argument("--ft_proj_dout", default=0.0, type=float,  help="Feature projection dropout")
parser.add_argument("--msk_tm_prob", default=0.05, type=float,  help="Mask time probability")
parser.add_argument("--ldrop", default=0.1, type=float, help="Layer drop")
parser.add_argument("--model", default="facebook/wav2vec2-large-xlsr-53", type=str, help="Wav2vec 2.0 model to use")

args = vars(parser.parse_args())
logging.debug("***Configuration: " + str(args)+"***")

run_name = args['run_name']#"test_nochanges_2-9-23"
logging.debug(f"Run name: {run_name}")

output_dir = Path(args['output_dir'])
data_dir = Path(args['data_dir'])


if os.path.exists(output_dir): logging.debug(f"Output directory {output_dir} exists")
else:
    logging.debug(f"Creating output directory {output_dir}")
    os.mkdir(output_dir)

run_dir = output_dir.joinpath(run_name)
logging.debug(f"Creating run directory {run_dir}")
os.mkdir(run_dir)

if args["tokenization"] != "None":
    logging.debug("***Setting Tokenization Scheme***")
    ort_tokenizer = Tokenization_Scheme(args["tokenization"])
else: ort_tokenizer = None

if args["eval_settings"] != "None":
    eval_set_path = Path(args["eval_settings"])
    if not(os.path.exists(eval_set_path)) or eval_set_path.suffix != ".json": 
        raise Exception("No eval_settings.json at eval_settings path")
else: eval_set_path = None
    

logging.debug("***Processing data***")
process_data(
    data_dir = data_dir, output_dir = run_dir, 
    tokenization_scheme=ort_tokenizer,
    remove_tones=args['no_tones'],
    combine_tones=args['comb_tones'], 
    combine_diac=args['comb_diac'],
    remove_hyphens=args['no_hyph'])
    
logging.debug("***Setting up vocab***")
setup_vocab(data_dir = run_dir, 
            output_dir = run_dir)

logging.debug("***Finetuning model***")
main_program(data_dir = data_dir, output_dir = run_dir,
    learn_rate=args['learning_rate'],
    batches=args['batch_size'],
    grdacc_steps=args['grdacc_steps'],
    epochs=args['epochs'],
    mixed_precision=not(args['cpu']),
    use_cpu = args['cpu'],
    no_cuda = args['cpu'],
    atn_dout=args['atn_dout'],
    hid_dout=args['hid_dout'],
    ft_proj_dout=args['ft_proj_dout'],
    msk_tm_prob=args['msk_tm_prob'],
    ldrop=args['ldrop'],
    w2v2_model=args['model'])
    
logging.debug("***Evaluating model***")
eval_program(eval_dir = run_dir, 
             cpu=args['cpu'], 
             ort_tokenizer=ort_tokenizer,
             eval_set_path=eval_set_path)
logging.debug("***Fine-tuning complete!***")