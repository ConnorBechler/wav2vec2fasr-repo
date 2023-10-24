from transcribe import chunk_audio
import pympi

filepath = "../wav-eaf-meta/td21-22_020.wav"
methods = ["rvad_chunk", "pitch_chunk"]

def create_chunked_annotation(path, methods, format):
    eaf = pympi.Eaf(author="transcribe.py")
    eaf.add_linked_file(file_path=filepath, mimetype=filepath[-3])
    eaf.remove_tier('default')
    for method in methods:
        eaf.add_tier(method)
        chunks = chunk_audio(None, filepath, method=method)
        print(len(chunks))
        for chunk in chunks:
            eaf.add_annotation(method, chunk[0], chunk[1], 'CHUNK')
    if format=="TextGrid":
        tg = eaf.to_textgrid()
        tg.to_file(f"{filepath[filepath.rfind('/')+1:]}_chunked.TextGrid")
    else:
        eaf.to_file(f"{filepath[filepath.rfind('/')+1:]}_chunked.eaf")
#eaf.to_file(f"{method}_segments_of_{filepath[filepath.rfind('/'):]}.eaf")

create_chunked_annotation(filepath, methods, 'eaf')