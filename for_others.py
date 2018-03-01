from copy import deepcopy
from newmq import mysimpl
import torch
import torch.distributed as dist
import numpy as np
from torch.multiprocessing import Process
import librosa
def init_processes(port,addr,rank,size,backend='tcp'):
    init_addr = backend + '://' + addr +":" + str(port)
    dist.init_process_group(init_addr,rank=rank,world_size=size)
def run():
    src = dst = 0;
    mytensor = torch.zeros(1000)
    dist.scatter(mytensor,src=src)

    #processing
    features,num_frames,freqs = mysimpl(mytensor)
    frames_features = {}
    for frame in range(num_frames+1):
        frames_features[frame] = []
    for x in features:
        # print(x[0],x[1],x[2]) x[1] = framenumber x[0] amp x[2] freq
        frames_features[int(x[1])].append((x[0],x[2]))
    frame_freq_bins =[]
    for x in range(num_frames+1):
        freq_bins = np.zeros(2048)
        #dict with key as freqbin
        to_be_added ={}
        for y in frames_features[x]:
            index_i = np.abs(freqs-y[1]).argmin();
            if(y[1] < freqs[index_i]):
                index_i -=1;
            if index_i not in to_be_added.keys():
                to_be_added[index_i] = []
            to_be_added[index_i].append(y[0])
        all_non_zero_bins = to_be_added.keys()
        for x in all_non_zero_bins:
            amp_array =to_be_added[x]
            amp_array = np.array(amp_array)
            avg_amp = np.mean(amp_array)
            freq_bins[x] += avg_amp
            # freq_bins = torch.LongTensor(freq_bins)
        frame_freq_bins.append(freq_bins)
    frame_freq_bins = np.array(frame_freq_bins)
    frame_freq_bins = torch.from_numpy(frame_freq_bins)
    dist.gather(frame_freq_bins,dst=dst)
    return;

