"""
This is a wholesale copy of coldtaco's version fork of the original rVAD package
fork: https://github.com/coldtaco/rVAD-faster
original: https://github.com/zhenghuatan/rVAD

I'm sure there is a better way to integrate this, but quite frankly I don't know one; maybe a github link in poetry?
TODO: Figure out that better way
"""

from __future__ import division
import numpy as np
import math
import scipy.io.wavfile as wav
from copy import deepcopy

from scipy.signal import lfilter
import time
import sys

#References
# Z.-H. Tan and B. Lindberg, Low-complexity variable frame rate analysis for speech recognition and voice activity detection.
# IEEE Journal of Selected Topics in Signal Processing, vol. 4, no. 5, pp. 798-807, 2010.
# Achintya Kumar Sarkar and Zheng-Hua Tan 2017
# Version: 02 Dec 2017

def speech_wave(fileName_):
     
     (fs,sig) = wav.read(fileName_)
     if 'int' not in str(sig.dtype):
          return fs, sig.astype('float32')
     if sig.dtype ==  'int16':
          nb = 16 # -> 16-bit wav files
     elif sig.dtype ==  'int32':
          nb = 32 # -> 32-bit wav files
     max_nb = float(2 ** (nb - 1))
     sig = sig / (max_nb + 1.0)  
     return fs, sig
 
def enframe(speech, fs, winlen, ovrlen):
     
      N, flth, foVr = len(speech), int(np.fix(fs*winlen)),  int(np.fix(fs*ovrlen))
      
      if len(speech) < flth:
          print("speech file length shorter than window length")
          exit()
      

      frames = int(np.ceil( (N - flth + foVr)/foVr))
      slen = (frames-1)*foVr + flth


      if len(speech) < slen:
          signal = np.concatenate((speech, np.zeros((slen - N))))

      else:
          signal = deepcopy(speech)
  

      idx = np.tile(np.arange(0,flth),(frames,1)) + np.tile(np.arange(0,(frames)*foVr,foVr),(flth,1)).T
      idx = np.array(idx,dtype = np.int64)
     
 
      return signal[idx]


def sflux(data, fs, winlen, ovrlen, nftt):
     
     eps = np.finfo(float).eps

     xf = enframe(data, fs, winlen, ovrlen) #framing
     w = np.matrix(np.hamming(int(fs*winlen)) )
     w = np.tile(w,(np.size(xf, axis = 0), 1))

     xf = np.multiply (xf, w) #apply window
     #fft
     ak = np.abs(np.fft.fft(xf,nftt))
#      idx = range(0,int(nftt/2) +1)
     idx_len = int(nftt/2) +1
     ak = ak[:,0:int(nftt/2) +1]
     Num = np.exp( float(1/idx_len) * np.sum(np.log(ak+eps), axis = 1) ) 
     Den = float(1/idx_len) * np.sum(ak, axis = 1)
     
     ft = (Num+eps)/(Den+eps)


     flen, fsh10 = int(np.fix(fs*winlen)),  int(np.fix(fs*ovrlen))
     nfr10 = int(np.floor((len(data)-(flen-fsh10))/fsh10))

     #syn frames as per nfr10
     if nfr10 < len(ft):
        ft = ft[:nfr10]
     else:
        ft = np.concatenate((ft, np.repeat(ft[:1], nfr10 -len(ft), axis = 0) ))


     
     return ft, flen, fsh10, nfr10


def snre_highenergy(fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk):

     ## ---*******- important *******
     #here [0] index array element has  not used 

     Dexpl = 18
     Dexpr = 18
     segThres = 0.25

     # TODO inserting infinity not needed
     fdata_ = np.insert(fdata,0,'inf')
     pv01_ = np.insert(pv01,0,'inf')


     #energy estimation
     e = np.zeros(nfr10+1,  dtype = 'float64')
     e[0] = np.inf

     for i in range(1, nfr10+1):
          ind = (i-1)*fsh10
          ind_start = ind + 1
          ind_end = ind + flen + 1
          e[i] += np.sum(fdata_[ind_start:ind_end])
          
     e[e <= ENERGYFLOOR] = ENERGYFLOOR

     emin = np.ones(nfr10 + 1)
     emin[0] = np.inf

     NESEG = min(nfr10, 200)

     # TODO come back later, can probably be improved
     for i in range(1, int(np.floor(nfr10/NESEG))+1):
          eY = np.sort(e[(i-1)*NESEG+1: (i*NESEG)+1])

          ind = int(np.floor(NESEG*0.1)) - 1
          val = eY[ind] if ind != -1 else np.inf

          emin[(i-1)*NESEG+1: i*NESEG+1] = val
          if i !=  1:
               emin[(i-1)*NESEG+1:i*NESEG+1] = 0.9*emin[(i-1)*NESEG]+0.1*emin[(i-1)*NESEG+1]

     if i*NESEG !=  nfr10:
          eY = np.sort(e[(i-1)*NESEG+1: nfr10+1])
          eY = np.insert(eY,0,'inf')
          ind = int(np.floor((nfr10-(i-1)*NESEG)*0.1)) - 1
          val = eY[ind] if ind != -1 else np.inf

          emin[i*NESEG+1:nfr10+1] = 0.9*emin[i*NESEG]+0.1*emin[i*NESEG+1]


#      D = np.zeros(nfr10)
#      D = np.insert(D,0,'inf')
     D = np.zeros(nfr10 + 1, dtype = 'float64')
     D[0] = np.inf


     postsnr = np.zeros(nfr10 + 1)
     postsnr[0] = np.inf

     # Slicing before operation to ignore inf related errors
     postsnr[2:nfr10 + 1] = (np.log10(e[2:nfr10 + 1]) - np.log10(emin[2:nfr10 + 1]))
     postsnr[postsnr < 0] = 0
     
     previous_e = np.roll(e, 1)

     D[2:nfr10 + 1] = np.sqrt(np.abs(e[2:nfr10 + 1]-previous_e[2:nfr10 + 1])*postsnr[2:nfr10 + 1])
     D[1] = D[2]
     
     tm1 = np.hstack((np.ones(Dexpl)*D[1], D[1:len(D)]))
     Dexp = np.hstack((tm1, np.ones(Dexpr)*D[nfr10] ))
     Dexp = np.insert(Dexp,0,'inf')
  
#      Dsmth = np.zeros(nfr10, dtype = 'float64')
#      Dsmth = np.insert(Dsmth,0,'inf')
     Dsmth = np.zeros(nfr10 + 1, dtype = 'float64')
     Dsmth[0] = np.inf
  
#      Dsmth_max = deepcopy(Dsmth)
     Dsmth_max = np.zeros(nfr10 + 1, dtype = 'float64')
     Dsmth_max[0] = np.inf


     for i in range(1,nfr10+1):
          Dsmth[i] = sum(Dexp[i: i+Dexpl+Dexpr+1])

     for i in range(1, int(np.floor(nfr10/NESEG))+1):
          Dsmth_max[(i-1)*NESEG+1: i*NESEG+1] =  np.amax(e[(i-1)*NESEG+1: i*NESEG+1])


     if i*NESEG != nfr10:
          Dsmth_max[i*NESEG+1: nfr10+1] = np.amax(e[(i-1)*NESEG+1: nfr10+1])

     snre_vad = np.zeros(nfr10 + 1)
     snre_vad[0] = np.inf

#      for i in range(1, nfr10+1):
#           if np.greater(Dsmth[i], Dsmth_max[i]*segThres):
#                snre_vad[i] = 1
     snre_vad[1:nfr10 + 1][(Dsmth[1:nfr10 + 1] > (Dsmth_max[1:nfr10 + 1] * segThres))] = 1

     #block based processing to remove noise part by using snre_vad1.
     sign_vad = 0
     noise_seg = np.zeros(int(np.floor(nfr10/1.6)) + 1)
     noise_seg[0] = np.inf
 
     noise_samp = np.zeros((nfr10,2))
     n_noise_samp = -1

     for i in range(1, nfr10+1):
          if (snre_vad[i] ==  1) and (sign_vad ==  0): #% start of a segment
               sign_vad = 1
               nstart = i
          elif ((snre_vad[i] == 0) or (i == nfr10)) and (sign_vad ==  1): # % end of a segment
               sign_vad = 0
               nstop = i-1
               if sum(pv01_[nstart: nstop+1]) == 0:
                    noise_seg[int(np.round(nstart/1.6)): int(np.floor(nstop/1.6))+1] = 1
                    n_noise_samp = n_noise_samp+1
                    noise_samp[n_noise_samp,:] = np.array([(nstart-1)*fsh10+1, nstop*fsh10])

     noise_samp = noise_samp[:n_noise_samp+1,]

     #syn  from [0] index
     noise_samp = noise_samp-1
     noise_seg = noise_seg[1:len(noise_seg)]
 
     return noise_samp, noise_seg, len(noise_samp)   




def snre_vad(fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk, vadThres):

     ## ---*******- important *******
     #here [0] index array element has  not used 

     Dexpl, Dexpr = 18, 18
     Dsmth = np.zeros(nfr10 + 1, dtype = 'float64')
     Dsmth[0] = np.inf
   
     fdata_ = fdata
     pv01_ = pv01
     pvblk_ = pvblk
 
     fdata_ = np.insert(fdata_,0,'inf')
     pv01_ = np.insert(pv01_,0,'inf')
     pvblk_ = np.insert(pvblk_,0,'inf')


     #energy estimation
     e = np.zeros(nfr10 + 1,  dtype = 'float64')
     e[0] = np.inf

     for i in range(1, nfr10+1):
          ind = (i-1)*fsh10
          start_ind = ind + 1
          end_ind = ind + flen
          e[i] = np.sum(fdata_[start_ind: end_ind]**2)
     
     e[e <= ENERGYFLOOR] = ENERGYFLOOR

     segsnr = np.zeros(nfr10 + 1)
     segsnr[0] = np.inf
     segsnrsmth = 1
     sign_segsnr = 0
     D = np.zeros(nfr10 + 1)
     D[0] = np.inf
     postsnr = np.zeros(nfr10 + 1)
     postsnr[0] = np.inf
     snre_vad = np.zeros(nfr10 + 1)
     snre_vad[0] = np.inf
     sign_pv = 0

     for i in range(1, nfr10+1):
          
          if (pvblk_[i] == 1) and (sign_pv == 0):
               nstart = i
               sign_pv = 1

          elif ( (pvblk_[i] == 0) or (i == nfr10) ) and (sign_pv == 1): 

               nstop = i-1
               if i == nfr10:
                    nstop = i
               sign_pv = 0
               datai = fdata_[(nstart-1)*fsh10+1: (nstop-1)*fsh10+flen-fsh10+1]
               datai = np.insert(datai,0,'inf')

               for j in range(nstart, nstop):  #previously it was for j = nstart:nstop-1
                    ind = (j-nstart)*fsh10
                    ind_start = ind + 1
                    ind_end = ind + flen + 1
                    e[j] += np.sum(datai[ind_start: ind_end]**2)
               
               e[nstart: nstop][e[nstart: nstop] <= ENERGYFLOOR] = ENERGYFLOOR

               e[nstop] = e[nstop-1]

               eY = np.sort(e[nstart: nstop+1] )
               # eY = np.insert(eY,0,'inf') #as [0] is discarding
               ind = int(np.floor((nstop-nstart+1)*0.1)) - 1
               emin = eY[ind]
               if ind == -1:
                    emin = np.inf

               for j in range(nstart+1, nstop+1):
                    
                    postsnr[j]  = math.log10(e[j]) - math.log10(emin)
                    if postsnr[j] < 0:
                         postsnr[j] = 0
                    
                    D[j] = math.sqrt(np.abs(e[j]-e[j-1])*postsnr[j] )
               
               D[nstart] = D[nstart+1]


               tm1 = np.hstack((np.ones(Dexpl)*D[nstart], D[nstart: nstop+1]))
               Dexp = np.hstack((tm1, np.ones(Dexpr)*D[nstop] ))
               
               Dexp = np.insert(Dexp,0,'inf')

               for j in range(0, nstop-nstart+1):
                    Dsmth[nstart+j] = sum(Dexp[j+1: j+Dexpl+Dexpr+1])

               Dsmth_thres = sum(Dsmth[nstart: nstop+1]*pv01_[nstart: nstop+1])/sum(pv01_[nstart: nstop+1])

               Dsmth_selection = Dsmth[nstart: nstop+1] > Dsmth_thres*vadThres
               snre_vad[nstart: nstop+1][Dsmth_selection] = 1
                       
     #      
     pv_vad = np.array(snre_vad)
          

     nexpl_1 = 33
     nexpr_1 = 47 # % 29 and 39, estimated statistically, 95% ; 33, 47 %98 for voicebox pitch
     nexpl_2  = 5
     nexpr_2 = 12 #; % 9 and 13, estimated statistically 5%; 5, 12 %2 for voicebox pitch
     sign_vad = 0

     pv01_eq_1 = pv01_ == 1

     for i in range(1, nfr10+1):
          if (snre_vad[i] == 1) and (sign_vad == 0):
               nstart = i
               sign_vad = 1
          elif ((snre_vad[i] == 0) or (i == nfr10)) and (sign_vad == 1):
               # Setup for both for-loops
               nstop = i-1
               if i == nfr10:
                    nstop = i
               sign_vad = 0

               # First for loop
               first_one_start = np.where(pv01_eq_1[nstart: nstop + 1])[0]
               first_one_0 = np.where(pv01_eq_1[0: nstop-nstart+1])[0]

               # Two different j's from different stopping conditions
               j_start = nstop if len(first_one_start) == 0 else first_one_start[0] + nstart
               j_end = nstop - nstart if len(first_one_start) == 0 else nstop - first_one_start[-1] - nstart

               pv_vad[nstart: np.max([j_start-nexpl_1-1,1])+1] = 0
               
               pv_vad[nstop-j_end+1+nexpr_1:nstop+1] = 0

               # Second for loop
               if sum(pv01_[nstart: nstop+1]) > 4:                    
                    pv_vad[max(j_start-nexpl_2,1):j_start] = 1

                    pv_vad[nstop-j_end+1:min(nstop-j_end+nexpr_2,nfr10)+1] = 1
          
               
               esegment = sum(e[nstart: nstop+1])/(nstop-nstart+1)
               if esegment < 0.001:
                    pv_vad[nstart: nstop+1] = 0
          
               if sum(pv01_[nstart: nstop+1]) <= 2:
                    pv_vad[nstart: nstop+1] = 0
          
     #
     sign_vad = 0
     vad_seg = np.zeros((nfr10,2), dtype = "int64")
     n_vad_seg = -1 #for indexing array
     for i in range(1,nfr10+1):
          if (pv_vad[i] == 1) and (sign_vad == 0):
               nstart = i
               sign_vad = 1
          elif ((pv_vad[i] == 0) or (i == nfr10)) and (sign_vad == 1):
               nstop = i-1
               sign_vad = 0
               n_vad_seg = n_vad_seg+1
               #print i, n_vad_seg, nstart, nstop
               vad_seg[n_vad_seg,:] = np.array([nstart, nstop])
     

     vad_seg = vad_seg[:n_vad_seg+1,]


     #syn  from [0] index
     vad_seg = vad_seg - 1

     #print vad_seg

     # make one dimension array of (0/1) 
     xYY = np.zeros(nfr10, dtype = "int64")
     for i in range(len(vad_seg)):  
          xYY[vad_seg[i,0]: vad_seg[i,1]+1] = 1

     vad_seg = xYY


     return vad_seg



def pitchblockdetect(pv01, pitch, nfr10, opts):
   

     pv01_ = deepcopy(pv01)

     if nfr10 ==  len(pv01_)+1:
        np.append(pv01_, pv01_[nfr10-1])  
     if opts ==  0:
          sign_pv = 0
          for i in range(0, nfr10):

               if ( pv01_[i] == 1) and (sign_pv == 0):
 
                    nstart, sign_pv  = i, 1

               elif ( (pv01_[i] ==  0) or (i == nfr10-1) ) and (sign_pv == 1):

                    nstop = i
                    if i == nfr10-1:
                       nstop = i+1
                    sign_pv = 0
                    pitchseg = np.zeros(nstop-nstart)
                    #print len(pitchseg)
                    for j in range (nstart, nstop):
                       
                       pitchseg[j-nstart] = pitch[j]
          
                    if (sum(np.abs( np.round( pitchseg-np.average(pitchseg) ) )) == 0)  and (nstop-nstart+1 >= 10):
                       pv01_[nstart: nstop] = 0 
     #
     sign_pv = 0
     pvblk = deepcopy(pv01_)   

     #print i
     for i in range(0, nfr10):
          
          if (pv01_[i] == 1) and (sign_pv == 0):
               #print("i = %s " %(i))
               nstart, sign_pv = i, 1
               pvblk[max([nstart-60,0]): nstart+1] = 1
               #print("fm P2: i = %s %s % " %(i,max([nstart-60,0]), nstart+1))
               
          elif ( (pv01_[i] == 0) or (i == nfr10-1 )) and (sign_pv == 1):

               nstop, sign_pv =  i, 0

               pvblk[nstop: np.amin([nstop+60,nfr10-1])+1 ] = 1 
               #print("fm P2: i = %s %s %s " %(i,nstop, np.amin([nstop+60,nfr10-1])+1 ))
               
     return pvblk 

winlen, ovrlen, pre_coef, nfilter, nftt = 0.025, 0.01, 0.97, 20, 512
opts=1

def rVAD_fast(finwav, fvad=None, ftThres=0.5, vadThres=0.4):
    if isinstance(finwav, str):
        fs, data = speech_wave(finwav)
    ft, flen, fsh10, nfr10 = sflux(data, fs, winlen, ovrlen, nftt)

    # --spectral flatness --
    #total_time = time.perf_counter()
    pv01=np.zeros(nfr10)
    pv01[np.less_equal(ft, ftThres)]=1 
    pitch=deepcopy(ft)
    #start = time.perf_counter()
    pvblk= pitchblockdetect(pv01, pitch, nfr10, opts)
    #print(time.perf_counter() - start)

    # --filtering--
    ENERGYFLOOR = np.exp(-50)
    b=np.array([0.9770,   -0.9770])
    a=np.array([1.0000,   -0.9540])
    #start = time.perf_counter()
    fdata=lfilter(b, a, data, axis=0)
    #print(time.perf_counter() - start)


    #--pass 1--
    #start = time.perf_counter()
    noise_samp, noise_seg, n_noise_samp= snre_highenergy(fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk)
    #print(time.perf_counter() - start)

    #sets noisy segments to zero
    #start = time.perf_counter()
    for j in range(n_noise_samp):
        fdata[range(int(noise_samp[j,0]),  int(noise_samp[j,1]) +1)] = 0 
    #print(time.perf_counter() - start)

    #start = time.perf_counter()

    vad_seg= snre_vad(fdata,  nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk, vadThres)
    #print(time.perf_counter() - start)
    #print(time.perf_counter()-total_time)
    if fvad != None: np.savetxt(fvad, vad_seg.astype(int),  fmt='%i')
    return(vad_seg)

if __name__ == '__main__':
    finwav=str(sys.argv[1])
    fvad=str(sys.argv[2])
    rVAD_fast(finwav, fvad)