import numpy as np
from scipy import stats
from scipy import signal

# Taken from:
# https://github.com/dgaddy/silent_speech/blob/1c91d5cddd7a3f39414ed77d3a80189f4e515c4d/data_utils.py

def double_average(x):
    assert len(x.shape) == 1
    f = np.ones(9)/9.0
    v = np.convolve(x, f, mode='same')
    w = np.convolve(v, f, mode='same')
    return w

# Taken from:
# https://github.com/wjbladek/SilentSpeechClassifier/blob/master/features.py

class freq_dom_feat:

    def __init__(self, sequence):
        self.t = len(sequence)
        self.fft = abs(np.fft.fft(sequence*signal.windows.hamming(self.t, sym=False)))*2/self.t
        self.fft = self.fft[:self.t//2]
        self.freqs = np.fft.fftfreq(self.t)[:self.t//2]

    # mean freq, a centroid of the spectrum
    def mean_freq(self):
        sum = np.sum(self.fft)
        # sorted = np.sort(self.fft)
        ind = 1
        additive = 0
        while additive <= sum/2:
            additive+=self.fft[ind]
            ind+=1
        return self.freqs[ind-1]

    # mode freq
    def mode_freq(self):
        return self.freqs[np.argmax(self.fft)]

    # median freq
    def median_freq(self):
        index = np.argsort(self.fft)
        freq_left = self.freqs[index[self.t//4+1]]
        if self.t % 2 != 0:
            return freq_left
        else:
            freq_right = self.freqs[index[self.t//4]]
            return (freq_right+freq_left)/2

# Around 2x faster implementation of the function above. 
# Only unwrapped from all the function calls.
def fast_feat_array(sequence, channel):
    """Extract features for a given sequence and return an structured array.
    
    Notes
    -----
    This functions takes a sequence and, using a set of functions, extracts
    various characteristics of the sequence. It returns them in a structured
    ndarray with a given channel name, features's names and features's
    values. Optimised with speed in mind (reduction in function calls).
    Parameters
    ----------
    sequence : array_like
        Signal to extract features from. Should be an 1d array/list of
        ints, floats or doubles.
    channel : string (U4)
        ID of a source of a sequence, for instance a name of an EEG channel.
    Returns
    -------
    ndarray
        Structured numpy array (shape 24,) with 3 fields per feature:
        a channel name (4 unicode chars), a feature's name (4 unicode 
        chars) and a feature's value (float).
    """
    positive = [n for n in sequence if n >= 0]
    par = np.sum(positive)
    negative = [n for n in sequence if n <= 0]
    nar = np.sum(negative)
    tar = par + nar
    taar = par + np.abs(nar)

    mins = np.abs(np.min(sequence))
    maxs = np.max(sequence)
    if ( mins <= maxs ):
        amp = maxs
    elif ( maxs <= mins ):
        amp = np.min(sequence)

    latency = np.where(sequence == amp)[0]
    lar = latency[0]/amp
    aamp = np.abs(amp)
    alar = np.abs(lar)
    pp = np.max(sequence) - np.min(sequence)
    ppt = np.where(sequence == np.max(sequence))[0] - np.where(sequence == np.min(sequence))
    zc = np.count_nonzero(np.where(np.diff(np.sign(sequence)))[0])
    pps = pp/ppt
    zcd = zc/ppt
    std = np.std(sequence)
    variance = np.var(sequence)
    mean_value = np.mean(sequence)
    median_value = np.median(sequence)
    mode_value1 = stats.mode(np.round(sequence, decimals=1), axis=None)
    mode_value2 = stats.mode(np.round(sequence, decimals=2), axis=None)
    mode_value3 = stats.mode(np.round(sequence, decimals=3), axis=None)
    fdf = freq_dom_feat(sequence)

    feat_array = np.array([
        (channel, 'par', par),
        (channel, 'tar', tar),
        (channel, 'nar', nar),
        (channel, 'taar', taar),
        (channel, 'amp', amp),
        (channel, 'lat', latency[0]),
        (channel, 'lar', lar),
        (channel, 'aamp', aamp),
        (channel, 'alar', alar),
        (channel, 'pp', pp),
        (channel, 'ppt', ppt[0][0]),
        (channel, 'zc', zc),
        (channel, 'pps', pps[0][0]),
        (channel, 'zcd', zcd[0][0]),
        (channel, 'std', std),
        (channel, 'var', variance),
        (channel, 'mns', mean_value),
        (channel, 'mes', median_value),
        (channel, 'md1s', mode_value1[0][0]),
        (channel, 'md2s', mode_value2[0][0]),
        (channel, 'md3s', mode_value3[0][0]),
        (channel, 'mnf', fdf.mode_freq()),
        (channel, 'mef', fdf.median_freq()),
        (channel, 'mdf', fdf.mode_freq())],
        dtype = [('channel', 'U4'),('feature_name','U4'),('feature_value','f8')])

    return feat_array