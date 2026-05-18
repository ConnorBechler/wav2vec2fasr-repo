import inspect
import pathlib
#from wav2vec2fasr import forcedalignment
from wav2vec2fasr import orthography as ort
#from wav2vec2fasr import segment
#from wav2vec2fasr import evaluate
#from wav2vec2fasr import mfa_tools
from wav2vec2fasr.audio_processor import return_tiers, create_dataset_from_dir
#import tests
#import pympi
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
mfa_corpus_dir = mfa_dir.joinpath("mfa_corpus/full")
lm_model = "c_npplm_nh_cb_5g.binary"
lm_dir = host_dir.joinpath("models/kenlm_models/"+lm_model)
mfa_alignments_dir = mfa_dir.joinpath("mfa_alignments", "7-15-24")
w2v2_alignments_dir = mfa_dir.joinpath("wv2vc2fasr_alignments", "7-6-24xls-r")

#MFA Pipeline
if False:
    # Create MFA ready text corpus
    for path in raw_corpus_dir.iterdir():
        if path.suffix == ".eaf":
            if not(mfa_corpus_dir.joinpath(path.name).exists()):
                mfa_tools.preprocess_ts_for_mfa(raw_corpus_dir.joinpath(pathlib.Path(path.stem+".wav")), path, output_dir=mfa_corpus_dir)
    # Generate vocab and pronunciation dictionary from corpus
    #vocab, chars = mfa_tools.load_vocab_from_ts_directory(mfa_corpus_dir)
    #pron_dict = mfa_tools.generate_pron_dict_w_phonemap(vocab, resources_dir.joinpath("phoneMapping3.txt"), mfa_dir)
    #print(pron_dict)

stt = time.time()
#create_dataset_from_dir(host_dir.joinpath("unaligned-wavs/toywavdata/raw/training"), "training", 
#                        host_dir.joinpath("unaligned-wavs/toywavdata/processed/"), tar_tier_type="sr_sentences")
#create_dataset_from_dir(host_dir.joinpath("unaligned-wavs/toywavdata/raw/testing"), "testing", 
#                        host_dir.joinpath("unaligned-wavs/toywavdata/processed/"), tar_tier_type="sr_sentences")
"""
texts = ort.load_directory(host_dir.joinpath("unaligned-wavs/toywavdata/raw_both"), ext=".TextGrid", tier_target="sr_sentences")
text = "\n".join([t[1] for t in texts])
scheme = ort.load_config()[0]
print(scheme)
text = ort.remove_special_chars(text)
text = scheme.revert(scheme.apply(text))
print(text)
"""
#evaluate.main_program(home=str(host_dir.drive), project_dir=str(host_dir), eval_dir="models/"+model, 
#                      data_dir=str(host_dir)+"/hf_datasets/data_all_speakers/", cpu=True)
#print(segment.chunk_audio(path=mfa_corpus_dir.joinpath("wq09_073.wav"),
#                          method="rvad_chunk_faster",))
                          #src_eaf=mfa_corpus_dir.joinpath("wq09_073.TextGrid")))
#mfa_tools.search_ts_corpus(mfa_corpus_dir, "ɬ")

#mfa_tools.describe_ts_corpus(mfa_corpus_dir, speech_tier_key="phrase-segnum", 
#                             collate_data=model_dir.joinpath("results.csv"),save_dir=mfa_corpus_dir)

#mfa_tools.compare_tss(mfa_alignments_dir.joinpath("wq10_011.TextGrid"), w2v2_alignments_dir.joinpath("wq10_011.TextGrid"))

#mfa_tools.compare_ts_dirs(mfa_alignments_dir, w2v2_alignments_dir, comp_tier_key=" - words")

#forcedalignment.align_transcription_dirs(mfa_corpus_dir, mfa_corpus_dir, model_dir)

#forcedalignment.align_transcriptions(test_rec, orig_eaf, model_dir, tier_list=["A_phrase-segnum-en"])
#mfa_tools.preprocess_ts_for_mfa(raw_corpus_dir.joinpath("wq09_001.wav"), raw_corpus_dir.joinpath("wq09_001.eaf"), output_dir=mfa_corpus_dir)
#forcedalignment.align_transcriptions(mfa_corpus_dir.joinpath("wq09_001.wav"),
#                                     mfa_corpus_dir.joinpath("wq09_001.TextGrid"),
#                                     model_dir)

#print(f"Testing alignment of {str(test_rec)} using the {str(model_dir)} model")

#methods = ["rvad_chunk", "rvad_chunk_faster", "pitch_chunk"]
#segment.create_chunked_annotation(test_rec, methods)

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

print("process took ", time.time()-stt, " seconds")
