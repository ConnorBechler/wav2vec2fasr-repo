#https://stackoverflow.com/a/50194143
#Solution did not work because pip install -e "D:/Northern Prinmi Data/wav2vec2faasr" somehow installs a link
# to the github package rather than the local package
#Can't find an answer online, ASK JOE

#The below line therefore fails (if testing.py is in the tests directory) as the module is not recognized
import inspect
import pathlib
from wav2vec2fasr import forcedalignment
from wav2vec2fasr import orthography as ort
import tests

test_path = pathlib.Path(inspect.getabsfile(inspect.currentframe()))
test_dir = test_path.parent.joinpath("test_files")
test_rec = test_dir.joinpath("td21-22_020.wav")
cor_words_eaf = test_dir.joinpath("td21-22_020_preds_wt_cor_words.eaf")
cor_phrase_eaf = test_dir.joinpath("td21-22_020_preds_wt_cor_phrase.eaf")
#Loads model from folder above location of wav2vec2faasr, negative indexes are not allowed with path parents for some reason
# BE SURE TO HAVE A MODEL IN THIS DIRECTORY FOR THIS TEST TO WORK (TODO: Place model directory in package)
host_dir = test_dir.parents[2]
model = "model_1-11-24_xls-r_ct_nh_1e-4_12e_fsd"
model_dir = host_dir.joinpath("models/"+model)

print(f"Testing alignment of {str(test_rec)} using the {str(model_dir)} model")

ort.load_tokenization("pumi_nt.tsv")
forcedalignment.chunk_and_align(test_rec, model_dir, output=".eaf")