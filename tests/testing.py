import inspect
import pathlib
from wav2vec2fasr import forcedalignment
from wav2vec2fasr import orthography as ort
from wav2vec2fasr import segment
from wav2vec2fasr import evaluate
from wav2vec2fasr import mfa_tools
from wav2vec2fasr.audio_processor import return_tiers
import tests
import pympi
import time

test_path = pathlib.Path(inspect.getabsfile(inspect.currentframe()))
test_dir = test_path.parent.joinpath("test_files")
test_rec = test_dir.joinpath("td21-22_020.wav")
orig_eaf = test_dir.joinpath("td21-22_020.eaf")
base_eaf = test_dir.joinpath("td21-22_020_preds.eaf")
cor_phrase_eaf = test_dir.joinpath("td21-22_020_preds_pm.eaf")
cor_phrase_ac_eaf = test_dir.joinpath("td21-22_020_preds_pm_ac.eaf")
cor_words_eaf = test_dir.joinpath("td21-22_020_preds_pm_ac_wm.eaf")
#Loads model from folder above location of wav2vec2faasr
# BE SURE TO HAVE A MODEL IN THIS DIRECTORY FOR THIS TEST TO WORK (TODO: Place model directory in package)
host_dir = test_dir.parents[2]
model = "model_4-4-24_xls-r_phons_nt_1e-4_12e_fsd"
model2 = "model_6-8-23_xlsr53_nt_nh"
model_dir = host_dir.joinpath("models/"+model)
raw_corpus_dir = host_dir.joinpath("wav-eaf-meta/")
resources_dir = test_dir.parents[1].joinpath("src/wav2vec2fasr/resources/")
mfa_dir = host_dir.joinpath("mfa/")
mfa_corpus_dir = mfa_dir.joinpath("mfa_corpus/")
lm_model = "c_npplm_nh_cb_5g.binary"
lm_dir = host_dir.joinpath("models/kenlm_models/"+lm_model)


#MFA Pipeline
# Create MFA ready text corpus
for path in raw_corpus_dir.iterdir():
    if path.suffix == ".eaf":
        if not(mfa_corpus_dir.joinpath(path.name).exists()):
            mfa_tools.preprocess_ts_for_mfa(raw_corpus_dir.joinpath(pathlib.Path(path.stem+".wav")), path, output_dir=mfa_corpus_dir)
# Generate vocab and pronunciation dictionary from corpus
vocab, chars = mfa_tools.load_vocab_from_ts_directory(mfa_corpus_dir)
pron_dict = mfa_tools.generate_pron_dict_w_phonemap(vocab, resources_dir.joinpath("phoneMapping2.txt"), mfa_dir)
#print(pron_dict)

stt = time.time()
for path in mfa_corpus_dir.iterdir():
    if path.suffix == ".TextGrid":
        print("Aligning ", str(path.name))
        forcedalignment.align_transcriptions(raw_corpus_dir.joinpath(pathlib.Path(path.stem+".wav")), path, model_dir)
print("alignment took ", time.time()-stt, " seconds")
#print(f"Testing alignment of {str(test_rec)} using the {str(model_dir)} model")

#methods = ["rvad_chunk", "rvad_chunk_faster", "pitch_chunk"]
#segment.create_chunked_annotation(test_rec, methods)

#Testing evaluation
#evaluate.main_program(home=str(host_dir.drive), project_dir=str(host_dir), eval_dir="models/"+model, 
#                      data_dir=str(host_dir)+"/hf_datasets/data_all_speakers/", cpu=True)

#ort.load_tokenization("pumi_phons.tsv")
#forcedalignment.generate_alignments_for_phrases(test_rec, orig_eaf, model_dir, src_tier="A_phrase-segnum-en")
#forcedalignment.chunk_and_align(test_rec, model_dir, output=".eaf", lm_dir=lm_dir)

#4/27/24 - Rewriting forced alignment function to automatically interpret textgrid or ELAN input
#forcedalignment.align_transcriptions(test_rec, orig_eaf, model_dir, tier_list=return_tiers(pympi.Eaf(orig_eaf), "phrase-segnum"))
#mfa_tools.strip_unnecessary_tiers(test_rec, orig_eaf)
#forcedalignment.generate_alignments_for_phrases(test_rec, orig_eaf, model_dir, src_tier="A_phrase-segnum-en")
#forcedalignment.chunk_and_align(test_rec, model_dir, output=".eaf", lm_dir=lm_dir)
#forcedalignment.chunk_and_align(test_rec, model_dir, output=".eaf")
#forcedalignment.correct_alignments(test_rec, base_eaf, cor_phrase_eaf, model_dir, cor_tier="prediction")
#forcedalignment.correct_alignments(test_rec, old_doc=cor_phrase_ac_eaf, corrected_doc=cor_words_eaf, model_dir=model_dir, cor_tier="words")