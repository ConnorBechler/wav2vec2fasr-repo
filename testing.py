#https://stackoverflow.com/a/50194143
#Solution did not work because pip install -e "D:/Northern Prinmi Data/wav2vec2faasr" somehow installs a link
# to the github package rather than the local package
#Can't find an answer online, ASK JOE

#The below line therefore fails (if testing.py is in the tests directory) as the module is not recognized
from importlib.resources import path as import_path
import pathlib
from src import forcedalignment
from src import orthography as ort
import tests.test_files as testingfiles

with import_path(testingfiles, "td21-22_020.wav") as test_path:
    test_path = str(test_path)
test_dir = pathlib.Path(test_path).parent
test_rec = pathlib.Path(test_path)
cor_words_eaf = test_dir.joinpath("td21-22_020_preds_wt_cor_words.eaf")
cor_phrase_eaf = test_dir.joinpath("td21-22_020_preds_wt_cor_phrase.eaf")
#Loads model from folder above location of wav2vec2faasr, negative indexes are not allowed with path parents for some reason
# BE SURE TO HAVE A MODEL IN THIS DIRECTORY FOR THIS TEST TO WORK (TODO: Place model directory in package)
host_dir = test_dir.parents[(len(test_dir.parents)-2)]
model = "model_6-8-23_xlsr53_nt_nh"
model_dir = host_dir.joinpath("models/"+model)

print(f"Testing alignment of {str(test_rec)} using the {str(model_dir)} model")

ort.load_tokenization("pumi_nt.tsv")
forcedalignment.chunk_and_align(test_rec, model_dir)