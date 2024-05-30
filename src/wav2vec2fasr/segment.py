"""
Collection of functions for segmenting audio.
Links together a bunch of other packages' functions to produce one uniform output.
"""
#from transcribe import chunk_audio
import os
import pathlib
from pydub import AudioSegment, silence
import librosa
import pympi
import parselmouth
from rVADfast import rVADfast
import time
from math import ceil
import pathlib
try:
    from speechbrain.pretrained import VAD
    VAD = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty")
except Exception as e:
    print("Speechbrain VAD import failed: " + str(e) +"\nTry to enable admin privileges to activate. In the meantime, Speechbrain VAD chunking disabled")
    VAD = None
import soundfile
from wav2vec2fasr import rvad_faster



def stride_chunk(max_chunk, stride, length, start=0, end=None):
    """
    Simple chunking using aribtary divisions based on max chunk length and striding
    """
    if end == None: end = start + round(length*1000)
    num_chunks = ceil((length*1000)/max_chunk)
    nchunks = [[start, start+max_chunk+stride, (0, stride)]]
    nchunks += [[start+x*max_chunk-stride, start+(x+1)*max_chunk+stride, (stride, stride)] for x in range(1, num_chunks-1)]
    if (end - start+max_chunk*(num_chunks-1)-stride) > 100:
        nchunks += [[start+max_chunk*(num_chunks-1)-stride, end, (stride, 0)]]
    else:
        nchunks[-1][1], nchunks[-1][3] = end, (stride, 0)
    return(nchunks)

def silence_stride_chunk(fullpath, aud_ext, max_chunk, min_chunk, stride, min_sil):
    """
    Slightly more complex chunking using pydub's silence function once and then subdividing chunks that are 
    too long using arbitrary divisions and striding
    """
    aud = AudioSegment.from_file(fullpath, format=aud_ext[1:])
    chunks = silence.detect_nonsilent(aud, min_silence_len=min_sil, silence_thresh=-35)
    nchunks = []
    for chunk in chunks:
        start, stop = chunk[0], chunk[1]
        diff = stop-start
        if diff > max_chunk:
            num_chunks = ceil(diff/max_chunk)
            step = round(diff/num_chunks)
            nchunks.append([start, start+step+stride, (0, stride)])
            for x in range(1, num_chunks):
                nchunks.append([(start+x*step)-stride, (start+(x+1)*step)+stride, (stride, stride)])
            nchunks.append([(start+(num_chunks)*step)-stride, stop, (stride, 0)])
        elif diff > min_chunk:
            nchunks.append([start, stop, (0, 0)])
    return(nchunks)

def og_silence_chunk(fullpath, aud_ext, min_sil, min_chunk, max_chunk, stride):
    """
    Uses pydub detect non-silent to chunk audio by silences into speech segments, then iterates through resulting chunks
    using a mixture of strategies to try and reduce their length to below max_chunk. This version integrates stride chunking
    as a last resort.
    """
    aud = AudioSegment.from_file(fullpath, format=aud_ext[1:])
    chunks = silence.detect_nonsilent(aud, min_silence_len=min_sil, silence_thresh=-35)
    nchunks = []
    for chunk in chunks:
        start, stop = chunk[0], chunk[1]
        diff = stop-start
        if start > 100: start -= 100
        if diff >= min_chunk:
            if diff >= max_chunk:
                solved = False
                for x in range(2, 11):
                    tch = silence.detect_nonsilent(aud[chunk[0]:chunk[1]], min_silence_len=round(min_sil/x), silence_thresh=-35)
                    if len([y for y in tch if y[1]-y[0] > max_chunk]) == 0: 
                        print("solved at", round(min_sil/x))
                        nchunks += [[y[0]+start, y[1]+start, (0, 0)] for y in tch]
                        solved=True
                        break
                if not(solved): 
                    print("Couldn't solve using shorter minimum silence lengths, increasing silence threshold instead")
                    tch = silence.detect_nonsilent(aud[chunk[0]:chunk[1]], min_silence_len=round(min_sil/2), silence_thresh=-35)
                    for y in tch:
                        ydiff = y[1]-y[0]
                        if ydiff > max_chunk:
                            print(ydiff)
                            for x in range(1, 6):
                                ntch = silence.detect_nonsilent(aud[start+y[0]:start+y[1]], min_silence_len=round(min_sil/2), silence_thresh=-35+x)
                                if len([z for z in ntch if z[1]-z[0] > max_chunk]) == 0: 
                                    print("solved at silence thresh of", -35+x)
                                    nchunks += [[z[0]+y[0]+start, z[1]+y[0]+start, (0, 0)] for z in ntch]
                                    break
                                else:
                                    if x==5:
                                        print("Couldn't divide automatically, instead just chopping up into windows of arbitrary length")
                                        nchunks+= stride_chunk(max_chunk, stride, ydiff/1000, y[0]+start, y[1]+start)
                                    pass
                        else: 
                            nchunks += [[y[0]+start, y[1]+start, (0, 0)]]
            else: nchunks.append([start, stop, (0, 0)])
    chunks = [chunk for chunk in nchunks if chunk[1]-chunk[0] > min_chunk]
    return(chunks)

def vad_chunk(lib_aud, max_chunk, sr, stride):
    """Uses speechbrain VAD to chunk audio, with fall-back stride chunking for segments that are too long"""
    if VAD != None:
        tempf = "./.temp_audio.wav"
        soundfile.write(tempf, lib_aud, samplerate=sr)
        #boundaries = VAD.get_speech_segments(tempf)
        #print(boundaries)
        prob_chunks = VAD.get_speech_prob_file(tempf, overlap_small_chunk=True)
        prob_th = VAD.apply_threshold(prob_chunks).float()
        boundaries = VAD.get_boundaries(prob_th)
        boundaries = VAD.energy_VAD(tempf,boundaries, .1)
        boundaries = VAD.merge_close_segments(boundaries, close_th=0.5)
        boundaries = VAD.remove_short_segments(boundaries, len_th=0.1)
        #boundaries = VAD.double_check_speech_segments(boundaries, tempf,  speech_th=0.5)
        chunks = [[round(int(y*1000)) for y in x] for x in boundaries.numpy()]
        nchunks = []
        for x in range(len(chunks)):
            diff = chunks[x][1] - chunks[x][0]
            if diff > max_chunk: 
                nchunk = stride_chunk(max_chunk, stride, (chunks[x][1]-chunks[x][0])/1000, chunks[x][0], chunks[x][1])
                nchunks += nchunk
            else: 
                nchunks += [[chunks[x][0], chunks[x][1], (0, 0)]]
        os.remove(tempf)
        return(nchunks)
    else: print("Speechbrain VAD disabled")

def rvad_chunk_base(audio, sr=None, vad=rVADfast()):
    """Uses rVAD to segment audio in a single pass"""
    vad_labels, vad_timestamps = vad(audio, sr)
    segments = []
    start, end = None, None
    for i, speech in enumerate(vad_labels):
        if speech == 1:
            if start == None: start = i
        else:
            if end == None and start != None:
                end = i
                segments.append([int(vad_timestamps[start]*1000), int(vad_timestamps[end]*1000), (0, 0)])
                start, end = None, None
    return(segments)

def rvad_chunk(audio, min_chunk, max_chunk, sr, vad=rVADfast()):
    """Uses rvad to segment audio recursively"""
    run = True
    segs = rvad_chunk_base(audio, sr, vad)
    while run:
        problems = [i for i in range(len(segs)) if segs[i][1]-segs[i][0]>max_chunk or segs[i][1]-segs[i][0]<min_chunk]
        if problems != []:
            for p in problems:
                st_ind = librosa.time_to_samples(int(segs[p][0]/1000), sr=sr)
                end_ind = librosa.time_to_samples(int(segs[p][1]/1000), sr=sr)
                new_segs = rvad_chunk_base(audio[st_ind:end_ind], sr, vad)
                new_segs = [[seg[0]+segs[p][0], seg[1]+segs[p][0], (0, 0)] for seg in new_segs]
                segs = segs[:p] + new_segs + segs[p+1:]
        else:
            run = False
    return(segs)

def rvad_chunk_faster(lib_aud, min_chunk, max_chunk, sr, stride):
    """Uses rvad_faster to chunk audio, with fall-back stride chunking for segments that are too long"""
    aud_mono = librosa.to_mono(lib_aud)
    soundfile.write("rvad_working_mono.wav", aud_mono, sr)
    segs = rvad_faster.rVAD_fast("rvad_working_mono.wav", ftThres = 0.4)
    os.remove("rvad_working_mono.wav")
    win_st = None
    win_end = None
    nchunks = []
    for x in range(len(segs)):
        if segs[x] == 1:
            if win_st == None: win_st = x*10
            win_end = x*10
        elif segs[x] == 0:
            if win_end != None:
                diff = win_end - win_st
                if diff > max_chunk:
                    num_chunks = ceil(diff/max_chunk)
                    step = round(diff/num_chunks)
                    newest_chunks = [[win_st+(step*x)-stride, win_st+(step*(x+1))+stride, 
                                    (stride, stride)] for x in range(num_chunks)]
                    if stride > 0 :
                        newest_chunks[0] = [win_st, win_st+(step+stride), (0, stride)]
                        newest_chunks[-1] = [win_end-step-stride, win_end, (stride, 0)]
                    nchunks += newest_chunks
                elif diff > min_chunk:
                    nchunks.append([win_st, win_end, (0, 0)])
                win_st, win_end = None, None
    return(nchunks)

def pitch_chunk(fullpath, min_chunk, max_chunk, stride):
    """Chunking method that uses praat pitch contours to chunk audio, with fall-back stride chunking"""
    rec = parselmouth.Sound(fullpath)
    pitch = rec.to_pitch()
    pitch_values = pitch.selected_array['frequency']    
    segs = list(pitch_values)
    win_st = None
    win_end = None
    nchunks = []
    for x in range(len(segs)):
        if segs[x] >0:
            if win_st == None: win_st = x*10
            win_end = x*10
        else:
            if win_end != None:
                diff = win_end - win_st
                if diff > max_chunk:
                    num_chunks = ceil(diff/max_chunk)
                    step = round(diff/num_chunks)
                    nchunks.append([win_st, win_st+step+stride, (0, stride)])
                    for x in range(1, num_chunks):
                        nchunks.append([(win_st+x*step)-stride, (win_st+(x+1)*step)+stride, (stride, stride)])
                    nchunks.append([(win_st+(num_chunks)*step)-stride, win_end, (stride, 0)])
                elif diff > min_chunk:
                    nchunks.append([win_st, win_end, (0, 0)])
                win_st, win_end = None, None
    nnchunks = []
    comb_chunk = None
    for x in range(len(nchunks)-1):
        if comb_chunk == None:
            if nchunks[x+1][0] - nchunks[x][1] <= 100:
                comb_chunk = [nchunks[x][0], nchunks[x+1][1], (nchunks[x][2][0], nchunks[x+1][2][1])]
            else:
                nnchunks.append(nchunks[x])
        else:
            if nchunks[x+1][0] - comb_chunk[1] <= 100:
                comb_chunk = [comb_chunk[0], nchunks[x+1][1], (comb_chunk[2][0], nchunks[x+1][2][1])]
            else:
                nnchunks.append(comb_chunk)
                comb_chunk = None
    return(nnchunks)

def chunk_audio(lib_aud=None, path=None, aud_ext=None, min_sil=1000, min_chunk=100, 
                    max_chunk=10000, stride = 1000, method='stride_chunk', length = None, sr = 16000) -> list:
    """
    Function for chunking long audio into shorter chunks with a specified method
        Requires either audio array or path to audio file
    Args:
        lib_aud (ndarray) : audio array representing waveform, as loaded by librosa
        path (str | pathlib.Path) : path to an audio file
        aud_ext (str) : file extension of audio (wav or mp3)
        min_sil (int) : minimum silence duration in milliseconds
        min_chunk (int) : minimum chunk duration in milliseconds
        max_chunk (int) : maximum chunk duration in milliseconds
        stride (int) : length of stride for stride chunking and fallback chunking for other methods
        method (str) : method used for chunking audio; can be silence_chunk, og_chunk, vad_chunk, rvad_chunk, pitch_chunk, or stride_chunk
        length (int) : duration of audio file in seconds, calculated using librosa if not provided
        sr (int) : sampling rate, 16000 by default
    Returns:
        list : chunks paired with ndarrays of audio 
        [ [ (chunk1_start_ms, chunk1_end_ms, (front_stride_ms, back_stride_ms) ), chunk1_audio_ndarray], ... 
          [(chunkN_start_ms, chunkN_end_ms, (front_stride_ms, back_stride_ms)), chunkN_audio_ndarray]]
    """
    if type(lib_aud) == type(None) and path != None:
        if pathlib.Path(path).exists():
            aud_ext = pathlib.Path(path).suffix
            lib_aud, sr = librosa.load(path, sr=16000)
    if length == None: length = librosa.get_duration(y=lib_aud, sr=sr)
    if method == 'silence_chunk': nchunks = silence_stride_chunk(path, aud_ext, max_chunk, 
                                                                   min_chunk, stride, min_sil)
    elif method == 'og_chunk': nchunks = og_silence_chunk(path, aud_ext, min_sil, min_chunk, max_chunk, stride)
    elif method == 'vad_chunk': nchunks = vad_chunk(lib_aud, max_chunk, sr, stride)
    elif method == 'rvad_chunk_faster': nchunks = rvad_chunk_faster(lib_aud, min_chunk, max_chunk, sr, stride)
    elif method == 'rvad_chunk': nchunks = rvad_chunk(lib_aud, min_chunk, max_chunk, sr)
    elif method == 'pitch_chunk': nchunks = pitch_chunk(path, min_chunk, max_chunk, stride)
    else: 
        method = 'stride_chunk'
        nchunks = stride_chunk(max_chunk, stride=stride, length=length)
    print(f'Chunked using {method} method')
    chunks = [nchunk + [lib_aud[librosa.time_to_samples(nchunk[0]/1000, sr=sr):
                                librosa.time_to_samples(nchunk[1]/1000, sr=sr)]] for nchunk in nchunks]
    return chunks

def create_chunked_annotation(filepath : str, methods, format=".eaf"):
    """Function for chunking a given audio file (mp3 or wav) by the specified chunking methods
    Args:
        filepath (str) : a path to an mp3 or wav audio file
        methods (str | list) : a single method or list of methods using the method names
            described by the chunk_audio function
        format (str) : the format of the resulting transcription file (.eaf or .TextGrid)
    Output:
        An eaf or TextGrid file named f"{filepath.stem}_chunked{format}" with tiers
            for each chunking method
    """
    filepath = pathlib.Path(filepath)
    ts = pympi.Eaf()
    ts.add_linked_file(file_path=filepath, mimetype=filepath.suffix)
    ts.remove_tier('default')
    for method in methods:
        ts.add_tier(method)
        tastt = time.time()
        chunks = chunk_audio(None, str(filepath), method=method)
        print(f"{method} took {time.time()-tastt} to chunk")
        print(len(chunks))
        for chunk in chunks:
            ts.add_annotation(method, chunk[0]+chunk[2][0], chunk[1]-chunk[2][1], 'CHUNK')
    ts.to_file(f"{filepath.stem}_chunked{format}")


if __name__ == "__main__":
    
    #filepath = "../wav-eaf-meta/td21-22_020.wav"
    filepath = "../../tests/test_files/td21-22_020.wav"
    methods = ["rvad_chunk", "rvad_chunk_faster", "pitch_chunk"]
    create_chunked_annotation(filepath, methods)