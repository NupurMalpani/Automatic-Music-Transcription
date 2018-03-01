import peakutils
import librosa
import scipy.signal as signal
import numpy as np
class MQPeak:
    def __init__(self,amp=0.0,freq=0.0,phase=0.0,bin=0,next=None,prev=None):
        self.amp = amp
        self.freq = freq
        self.phase = phase
        self.bin = bin
        self.next = next
        self.prev = prev

# def removeDuplicates(arr):
#     if(len(arr) < 2):
#         return arr;
#     j = 0
#     new_arr = []
#     # Doing same as done
#     # in Method 1 Just
#     # maintaining another
#     # updated index i.e. j
#     for i in range(len(arr)):
#         if arr[i].freq != arr[i+1].freq:
#             new_arr.append(arr[i])
#     arr = new_arr
'''
class MQ_parameters:
    def __init__(self,fft_plan,frame_size=0,max_peaks=0,num_bins=0,peak_threshold=0.0,fundamental=0.0,matching_interval=0.0,window=None,fft_in=None,fft_out=None,MQPeakList=None):
        self.fft_plan= fft_plan
        self.frame_size = frame_size
        self.max_peaks =  max_peaks
        self.num_bins =  num_bins
        self.peak_threshold = peak_threshold
        self.fundamental = fundamental
        self.matching_interval =matching_interval
        self.window = window
        self.fft_in = fft_in
        self.fft_out = fft_out
        self.fft_plan = fft_plan
        self.MQPeakList = MQPeakList
    # def mq_add_peak(self,new_peak,peak_list):
    #     #min heap
    #     heapq_max.heappush_max(peak_list(new_peak.amp,new_peak))
'''
def mq_find_peaks(gotten_fft,win_size,samplingrate,max_peaks):
    amps = gotten_fft.real **2 + gotten_fft.imag **2
    indexes = peakutils.indexes(amps)
    phases = np.arctan(np.divide(gotten_fft.imag,gotten_fft.real))
    getpeak = lambda t:MQPeak(amps[t],(t+1) * samplingrate/(win_size/2 + 1),phases[t],t)
    get_peaks = np.vectorize(getpeak)
    peaks = []
    if indexes.size != 0:
        peaks = list(get_peaks(indexes))
        peaks.sort(key=lambda x:x.amp,reverse=True)
        peaks = peaks[:max_peaks]
    return peaks
def sort_peaks(peak_lists):
    peak_lists.sort(key=lambda x:x.freq)
    # peak_lists = removeDuplicates(peak_lists)
    return peak_lists
def mq_track_peaks(all_peaks,frame_number,prev_peak_list,curr_peak_list,matching_interval=0.0):
    all_freqs = list(all_peaks.keys())
    prev_peak_freqs = np.array([prev_peak.freq for prev_peak in prev_peak_list])
    # print(all_freqs,list(prev_peak_freqs))
    curr_peak_freqs = np.array([cur_peak.freq for cur_peak in curr_peak_list])
    matches = lambda t:np.argmin(np.fabs(curr_peak_freqs-t))
    get_matches = np.vectorize(matches)
    if(prev_peak_freqs.size!= 0 and curr_peak_freqs.size!=0):
        mymatchedpeaks  = list(np.take(curr_peak_list,get_matches(prev_peak_freqs)))
        mymatchedpeaks_freqs = [peak.freq for peak  in mymatchedpeaks]
        # print("my matched adn prev peaks",mymatchedpeaks_freqs,prev_peak_freqs)
        matchesnotFound = np.where((prev_peak_freqs != np.array(mymatchedpeaks_freqs)))
        matchesnotFound = matchesnotFound[0]
        all_indices = np.arange(0,prev_peak_freqs.size,1)
        matches_found = list(np.delete(all_indices,matchesnotFound))
        # print("found",matches_found,"not found",matchesnotFound)
        for i in matches_found:
            all_peaks[prev_peak_freqs[i]].append((mymatchedpeaks[i],frame_number))
        #lower freqs - curr_freq < matching_interval
        matched_again = []
        for i in range(matchesnotFound.size):
            prev_ind =int(matchesnotFound[i])
            cur_ind = int(get_matches(prev_peak_freqs)[i])
            # print(prev_ind,cur_ind)
            curr_interest = cur_ind if prev_peak_freqs[prev_ind] > curr_peak_freqs[cur_ind] else cur_ind -1
            curr_interest = int(curr_interest)
            if(curr_interest >= 0 and (prev_peak_freqs[prev_ind] - curr_peak_freqs[curr_interest] < matching_interval)):
                curr_peak_list[curr_interest].freq = prev_peak_freqs[prev_ind]
                # print(prev_ind,curr_interest)
                all_peaks[prev_peak_freqs[prev_ind]].append((curr_peak_list[curr_interest],frame_number))
                matched_again.append(i)
        # print("matched",matched_again)
        curr_matched = np.unique(np.array(matches_found+matched_again))
        # print("currmatched",curr_matched)
        all_curr_indices = np.arange(0,curr_peak_freqs.size,1)
        unmatched_curr_from_prev = np.delete(all_curr_indices,curr_matched)
        unmatched_curr_from_all_freq = np.where(np.isin(curr_peak_freqs,np.array(all_freqs))==False)[0]
        unmatched_curr = np.unique(np.concatenate([unmatched_curr_from_all_freq,unmatched_curr_from_prev]))
        for i in unmatched_curr:
            all_peaks[curr_peak_freqs[i]] = []
            all_peaks[curr_peak_freqs[i]].append((curr_peak_list[i],frame_number))
    elif(prev_peak_freqs.size == 0 and curr_peak_freqs.size != 0):
        curr_matched = np.where(np.isin(curr_peak_freqs,np.array(all_freqs)))[0]
        all_curr_indices = np.arange(0,curr_peak_freqs.size,1)
        unmatched_curr = np.delete(all_curr_indices,curr_matched)
        for i in unmatched_curr:
            all_peaks[curr_peak_freqs[i]] = []
            all_peaks[curr_peak_freqs[i]].append((curr_peak_list[i],frame_number))
    else:
        pass
    return all_peaks

def init_all_freqs(sorted_first_peaks):
    all_peaks = {}
    # first_peaks = sort(first_peaks)
    for peak in sorted_first_peaks:
        all_peaks[peak.freq]  = []
        all_peaks[peak.freq].append((peak,0))
    return  all_peaks

def mysimpl(ten):
    sr = ten[-1]
    y = ten[:-1]
     # = librosa.load(filenam
    y = y.to_numpy()
    stfts = librosa.stft(y,window=signal.get_window('blackman',2048))
    stfts = np.transpose(stfts)
    first_peaks = mq_find_peaks(stfts[0],2048,sr,30)
    first_peaks = sort_peaks(first_peaks)
    all_peaks = init_all_freqs(first_peaks)
    prev_peaks = first_peaks
    for i in range(1,stfts.shape[0]):
        curr_peaks =sort_peaks( mq_find_peaks(stfts[i],2048,sr,30))
        all_peaks = mq_track_peaks(all_peaks,i,prev_peaks,curr_peaks,100.0)
        prev_peaks = curr_peaks
    features =[]
    all_freqs = all_peaks.keys()
    max_amp  = 0
    for i in all_freqs:
        peaks_frames = all_peaks[i]
        # print(peaks_frames)
        amps = [abs(a[0].amp) for a in peaks_frames]
        curr_max_amp = max(amps)
        if(curr_max_amp > max_amp):
            max_amp = curr_max_amp
        for j in peaks_frames:
            features.append((j[0].amp,j[1],i))
    norm_features = [(j[0]/max_amp,j[1],j[2]) for j in features]
    freqs = np.array([i * sr/2048 for i in range(2049)])
    return norm_features,stfts.shape[0],freqs
