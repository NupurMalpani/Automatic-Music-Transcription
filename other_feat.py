import librosa
# import  audiolazy
# from audiolazy import lpc
import os
import torchaudio
import scipy.signal as signal
import math
import numpy as np
def spec_cent(y,sr):
    cent = librosa.feature.spectral_centroid(y,sr)
    return cent
def spec_roll(y,sr):
    roll = librosa.feature.spectral_rolloff(y,sr)
    return roll
def zcr(y):
    cr = librosa.feature.zero_crossing_rate(y)
    return cr
# def mylpc(y):
#     analysis_filt = lpc.kautocor(y,25)
#     residual = analysis_filt(y)
#     synth_filt  = 1 /analysis_filt
#     return synth_filt(residual)
def spec_spread(y,sr,stfts):
    cent = spec_cent(y,sr)
    frames = cent.shape[1]
    #print("frames by cent",frames)
    bins = stfts[0].shape[0]
    freqs = [(t + 1) * sr /bins for t in range(bins)]
    freqs = np.array(freqs)
    spec_sp = []
    for frame in range(frames):
        amp = get_amp(stfts[frame])
        s = sum([(freqs[t] - cent[0][frame])**2 * amp[t] for t in range(bins)])
        a = np.sum(amp)
        # print(s,a)
        if a != 0:
            spec_sp.append(s/a)
        else:
            spec_sp.append(a)
    spec_sp = np.array(spec_sp)
    return spec_sp,cent

def get_amp(stft):
    c_amp =  np.square(stft.imag)
    r_amp = np.square(stft.real)
    amp = np.sqrt(c_amp + r_amp)
    return amp

def spec_flux(stfts):
    flux = []
    first = np.sqrt(np.sum(np.square(get_amp(stfts[0]))))
    flux.append(first)
    for x in range(1,len(stfts)):
        stft_diff = stfts[x-1] - stfts[x]
        diff = math.sqrt(np.sum(np.square(get_amp(stft_diff))))
        flux.append(diff)
        # print(diff)
    return np.array(flux)

def all_feat(filename):
    y,sr = torchaudio.load(filename)
    if(y.shape[1] >  1 ):
        d = y.shape[1]
        y =y.numpy()
        y = np.sum(y,axis = 1 )/d
    else:
        y = y.view(-1,)
        y =y.numpy()
    # y =y.numpy()
    # print(y.shape,filename)
    # new_y = mylpc(y)
    #noise = y - new_y
    stfts = librosa.stft(y,window=signal.get_window('blackman',2048))
    stfts = np.transpose(stfts)
    #print("frames by others are",stfts.shape[0])
    spec_sp , cent= spec_spread(y,sr,stfts)
    roll = spec_roll(y,sr)
    flux = spec_flux(stfts)
    cr = zcr(y)
    #pitches = os.popen("aubiopitch -i filename -m fcomb -t 0.2")
    #pitches = pitches.read()
    return spec_sp,cent,roll,cr,flux
