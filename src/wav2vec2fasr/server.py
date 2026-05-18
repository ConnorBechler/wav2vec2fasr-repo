"""
Basic server for interfacing wav2vec2fasr with other apps
Draws on the following references:
https://medium.com/@faz.pak/run-flask-rest-api-app-with-file-upload-api-its-easier-than-you-think-8620679265c3
https://community.hetzner.com/tutorials/building-a-flask-api-to-transcribe-audio-files-using-whisper-ai
https://stackoverflow.com/questions/26980713/solve-cross-origin-resource-sharing-with-flask
https://copilot.microsoft.com/shares/RRUANj2vQveFhniSJ3m83 - used copilot to help debug NPM axios error on the MiDRF backend
TODO: Maybe add whisper finetuning support
"""


from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from flask_cors import CORS
from importlib import resources as il_resources
import os

from wav2vec2fasr.forcedalignment import transcribe_audio, load_model_and_processor
from wav2vec2fasr.transcribe import load_whisper_pipeline, whisper_transcribe, load_whisper_model_and_processor, whisper_transcribe_mp
import wav2vec2fasr
from torch.cuda import is_available
from pathlib import Path
with il_resources.path(wav2vec2fasr, "serverfiles") as path:
        file_dir = Path(path)
        upload_dir = file_dir.joinpath("uploads")
        transcript_dir = file_dir.joinpath("transcripts")
#os.makedirs(upload_dir, exist_ok=True)
#os.makedirs(transcript_dir, exist_ok=True)

def allowedFile(filename):
    return('.' in filename and filename.rsplit('.', 1)[1].lower() in ["mp3","wav"])

def reformat_output(ts_output):
    sentences = [
        {
            'startTime' : sent[0]/1000, 
            'endTime' : sent[1]/1000, 
            'content' : sent[2],
            'metadata' : {
               'confidence' : None,
               'wordCount' : len(sent[2].split(' ')),
               'speaker' : 1,
               'header2' : 'SID'
            }
        } 
            for sent in ts_output['utterances']]
    return(sentences)

if is_available(): device = "cuda"
else: device = "cpu"
asr_type = "whisper"

if asr_type == "wav2vec2":
    model, processor = load_model_and_processor("C:/Users/bechl/code/npp_asr/output/model_2025-07-21_3_fsd/", device=device)
elif asr_type == "whisper":
    model, processor = load_whisper_model_and_processor("openai/whisper-tiny", device=device)
    #pipeline = load_whisper_pipeline("openai/whisper-tiny", device=device)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 3000 * 1024 * 1024  # 3 GB
app.config['CORS_HEADER'] = 'application/json'
CORS(app)

@app.route('/upload', methods=['POST', 'GET'])
def fileUpload():
    if request.method == 'POST':
        print(request)
        file = request.files.getlist('file')
        filename = ""
        filepath = ""
        output = ""
        print(request.files, "....")
        for f in file:
            print(f.filename)
            filename = secure_filename(f.filename)
            #print(allowedFile(filename))
            if allowedFile(filename):
                filepath = file_dir.joinpath(filename)
                f.save(filepath)
                if asr_type=="wav2vec2":
                    output = transcribe_audio(filepath, model=model, processor=processor,output=None, device=device)
                elif asr_type=="whisper":
                    output = whisper_transcribe_mp(str(filepath), model=model, processor=processor, language="english", device=device)
                    #output = whisper_transcribe(filepath, pipeline=pipeline, language="english")
                output = reformat_output(output)
                os.remove(filepath)
            else:
                return jsonify({'message': 'File type not allowed'}), 400
        return jsonify({"name": filename, "status": "success", "results" : output})
    else:
        return jsonify({"status": "Upload API GET Request Running"})
    
if __name__ == "__main__":
     app.run(debug=True)
     """Testing Command
     curl -v -X POST http://127.0.0.1:5000/upload -F "file=@C:/Users/bechl/code/npp_asr/wav-eaf-meta/wq14_039.wav;filename=wq14_039.wav"
     """