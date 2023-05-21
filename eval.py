"""
Module for calling evaluate.py from commandline/terminal with arguments
"""
import logging

from evaluate import main_program as eval_program

#Arguments stuff added with help from https://machinelearningmastery.com/command-line-arguments-for-your-python-script/
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
warnings.simplefilter("ignore")

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("eval_dir", help="Directory of model to be evaluated")
parser.add_argument("-d", "--data_dir", default=None, help="Directory of data to evaluate model with")
parser.add_argument("-c", "--checkpoint", default=None, help="Checkpoint of model to evaluate")
parser.add_argument("--cpu", action="store_true", help="Run without mixed precision")
args = vars(parser.parse_args())

logging.debug("***Evaluating model***")
eval_program(eval_dir=args['eval_dir'], 
    data_dir=args['data_dir'], 
    checkpoint=args['checkpoint'], 
    cpu=args['cpu'])