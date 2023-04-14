import os
import logging


from new_data_process import process_data
from new_vocab_setup import setup_vocab
from new_finetune import main_program
from evaluate import main_program as eval_program

run_name = "test_nochanges_2-9-23"
logging.debug(f"Run name: {run_name}")
original_data = "data"


project_dir = "npp_asr"
full_project = os.path.join(os.environ["HOME"], project_dir)
output_dir = os.path.join(full_project, "output/"+run_name)

if os.path.exists(output_dir):
    logging.debug(f"Output directory {output_dir} exists")
else:
    logging.debug(f"Creating output directory {output_dir}")
    os.mkdir(output_dir)

logging.debug("***Processing data***")
process_data(data_dir = original_data, output_dir = output_dir)
logging.debug("***Setting up vocab***")
setup_vocab(output_dir = output_dir)
logging.debug("***Finetuning model***")
main_program(output_dir = output_dir)
logging.debug("***Evaluating model***")
eval_program(eval_dir = output_dir)