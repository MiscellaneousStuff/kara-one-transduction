# MIT License
# 
# Copyright (c) 2022 MiscellaneousStuff
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Kara One helper library.

This library has been heavily adapated from:
https://github.com/wjbladek/SilentSpeechClassifier/blob/
b7f5ffd2a314ee14678a3e141d7addbd5320b5d0/SSC.py#L284"""

import os
import glob
from time import time

import torch
import mne
import librosa
import scipy

import soundfile as sf
import scipy.io as sio

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from .extra import *

PATH_INBETWEEN = "spoclab/users/szhao/EEG/data/"
DEFAULT_CHANNELS = [
    'FC6', 'FT8', 'C5', 'CP3', 'P3', 'T7', 'CP5', 'C3', 'CP1', 'C4']
DROP_CHANNELS = [
    'CB1', 'CB2', 'VEO', 'HEO', 'EKG', 'EMG', 'Trigger']

def remove_drift(signal, fs):
    b, a = scipy.signal.butter(3, 2, 'highpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)

def notch(signal, freq, sample_frequency):
    b, a = scipy.signal.iirnotch(freq, 30, sample_frequency)
    return scipy.signal.filtfilt(b, a, signal)

def notch_harmonics(signal, freq, sample_frequency):
    for harmonic in range(1,8):
        signal = notch(signal, freq*harmonic, sample_frequency)
    return signal

def subsample(signal, new_freq, old_freq):
    times = np.arange(len(signal))/old_freq
    sample_times = np.arange(0, times[-1], 1/new_freq)
    result = np.interp(sample_times, times, signal)
    return result

def load_audio(fname,
               n_mel_channels=128,
               filter_length=512,
               win_length=432,
               hop_length=160):
    audio, r = sf.read(fname)
    assert r == 16000

    audio_features = librosa.feature.melspectrogram(
        audio,
        sr=r,
        n_mels=n_mel_channels,
        center=False,
        n_fft=filter_length,
        win_length=win_length,
        hop_length=hop_length).T
    audio_features = np.log(audio_features + 1e-5)

    return audio, audio_features.astype(np.float32)

def read_eeg(eeg_path, channels_only=[], drop_channels=DROP_CHANNELS):
    print("PATH:", eeg_path)
    eeg_raw = mne.io.read_raw_cnt(eeg_path, preload=True) # .load_data()
    eeg_raw.drop_channels(drop_channels)
    if channels_only:
        eeg_raw.pick_channels(channels_only)
    print("EEG_RAW:", eeg_raw)
    return eeg_raw

def load_utterance(eeg_data, powerline_freq=60):
    eeg = []

    for eeg_channel in range(eeg_data.shape[0]):
        x = eeg_data[eeg_channel, :]
        x = notch_harmonics(x, powerline_freq, 1000)
        x = remove_drift(x, 1000)
        x = subsample(x, 600, 1000)
        eeg.append(x)

    eeg = np.stack(eeg, 1)
    return eeg

def get_semg_feats(eeg_data, stft=False, debug=False):
    xs = eeg_data - eeg_data.mean(axis=0, keepdims=True)
    frame_features = []
    for i in range(eeg_data.shape[1]):
        x = xs[:, i]
        w = double_average(x)
        p = x - w
        r = np.abs(p)

        w_h = librosa.util.frame(w, frame_length=16, hop_length=6).mean(axis=0)
        p_w = librosa.feature.rms(w, frame_length=16, hop_length=6, center=False)
        p_w = np.squeeze(p_w, 0)
        p_r = librosa.feature.rms(r, frame_length=16, hop_length=6, center=False)
        p_r = np.squeeze(p_r, 0)
        z_p = librosa.feature.zero_crossing_rate(p, frame_length=16, hop_length=6, center=False)
        z_p = np.squeeze(z_p, 0)
        r_h = librosa.util.frame(r, frame_length=16, hop_length=6).mean(axis=0)

        if stft:
            s = abs(librosa.stft(np.ascontiguousarray(x), n_fft=16, hop_length=6, center=False))

        if debug:
            plt.subplot(7,1,1)
            plt.plot(x)
            plt.subplot(7,1,2)
            plt.plot(w_h)
            plt.subplot(7,1,3)
            plt.plot(p_w)
            plt.subplot(7,1,4)
            plt.plot(p_r)
            plt.subplot(7,1,5)
            plt.plot(z_p)
            plt.subplot(7,1,6)
            plt.plot(r_h)

            plt.subplot(7,1,7)
            plt.imshow(s, origin='lower', aspect='auto', interpolation='nearest')

            plt.show()

        frame_features.append(np.stack([w_h, p_w, p_r, z_p, r_h], axis=1))
        if stft:
            frame_features.append(s.T)
    
    frame_features = np.concatenate(frame_features, axis=1)
    return frame_features.astype(np.float32)

def calculate_features(eeg_data, epoch_inds, prompts, condition_inds, prompts_list, end_idx=0, start_idx=0):
    offset  = int(eeg_data.info["sfreq"] / 2)
    X       = []
    X_raw   = []
    end_idx = end_idx if end_idx else len(prompts["prompts"][0])

    max_raw_size = 0
    for i, prompt in enumerate(prompts["prompts"][0][start_idx:end_idx]):
        t0 = time()
        if prompt[0] in prompts_list:
            start = epoch_inds[condition_inds][0][i][0][0] + offset
            end   = epoch_inds[condition_inds][0][i][0][1]
            channel_set = []
            channel_set_raw = []
            for idx, ch in enumerate(eeg_data.ch_names):
                epoch = eeg_data[idx][0][0][start:end]
                if epoch.shape[0] > max_raw_size:
                    max_raw_size = epoch.shape[0]
                #print(ch, epoch)
                #print(ch, "shape:", epoch.shape)
                channel_set.extend(fast_feat_array(epoch, ch))
                channel_set_raw.extend(epoch)
            # print("channel count:", len(eeg_data.ch_names), len(channel_set))
            X.append(channel_set)
            X_raw.append(channel_set_raw)
        print("Calc: %0.3fs" % (time() - t0), i, prompt)

    new_X_raw = []
    X_feats = []

    for cur in X_raw:
        seq_len = int(len(cur) / 62)

        # Cur raw, split into 62 channels
        cleaned_raw_eeg = np.reshape(cur, (62, seq_len))
        cleaned_raw_eeg = load_utterance(cleaned_raw_eeg)
        new_X_raw.append(cleaned_raw_eeg)

        # Using sEMG Silent Speech hand-crafted features
        X_feat = get_semg_feats(cleaned_raw_eeg)
        X_feats.append(X_feat)

    X_raw = new_X_raw

    return X, X_raw, X_feats


class KaraOneDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root_dir,
            pts=("MM05",),
            raw=True,
            start_idx=0,
            end_idx=0,
            scale_data=False):

        self.root_dir = root_dir
        self.pts = pts

        info_dir = \
            os.path.join(
                root_dir,
                PATH_INBETWEEN,
                pts[0],
                f"kinect_data/")

        with open(os.path.join(info_dir, f"{pts[0]}_p.txt")) as f:
            self.labels = f.read().split("\n")
            self.ids = range(len(self.labels))
            self.audios = [os.path.join(info_dir, f"{id_}.wav")
                            for id_ in self.ids]

        base_dir = os.path.join(
            self.root_dir,
            PATH_INBETWEEN,
            self.pts[0])

        for f in glob.glob(
            os.path.join(f"{base_dir}", "all_features_simple.mat")):
            prompts_to_extract = sio.loadmat(f)

        self.prompts = prompts = prompts_to_extract['all_features'][0,0]
        print("prompts:", len(self.prompts["prompts"][0]))

        for f in glob.glob(
            os.path.join(f"{base_dir}", "epoch_inds.mat")):
            self.epoch_inds = epoch_inds = \
                sio.loadmat(f, variable_names=
                    ('clearing_inds', 'thinking_inds', 'speaking_inds'))

        if raw:
            for f in glob.glob(
                os.path.join(f"{base_dir}", "*.cnt")):
                eeg_data = read_eeg(f)

        self.eeg_data = eeg_data

        t0 = time()

        end_idx = end_idx if end_idx else len(self.prompts["prompts"][0])
        self.Y_s = Y_s = self.prompts["prompts"][0][start_idx:end_idx]
        print("Actual Length:", len(self.prompts["prompts"][0]))

        self.X_s_rest, self.X_s_rest_raw, self.X_s_rest_feats = \
            calculate_features(
                eeg_data,
                epoch_inds,
                prompts,
                "clearing_inds",
                Y_s,
                end_idx,
                start_idx)

        self.X_s_active, self.X_s_active_raw, self.X_s_active_feats = \
            calculate_features(
                eeg_data,
                epoch_inds,
                prompts,
                "thinking_inds",
                Y_s,
                end_idx,
                start_idx)

        self.X_s_vocal, self.X_s_vocal_raw, self.X_s_vocal_feats = \
            calculate_features(
                eeg_data,
                epoch_inds,
                prompts,
                "speaking_inds",
                Y_s,
                end_idx,
                start_idx)

        self.X_s_rest   = np.asarray(self.X_s_rest)
        self.X_s_active = np.asarray(self.X_s_active)
        self.X_s_vocal  = np.asarray(self.X_s_vocal)

        if scale_data:
            self.X_s_rest["feature_value"] = \
                StandardScaler().fit_transform(
                    self.X_s_rest["feature_value"])
            self.X_s_active["feature_value"] = \
                StandardScaler().fit_transform(
                    self.X_s_active["feature_value"])
            self.X_s_vocal["feature_value"] = \
                StandardScaler().fit_transform(
                    self.X_s_vocal["feature_value"])

        self.Y_s = np.hstack(self.Y_s)

        print("Calc: %0.3fs" % (time() - t0))

    def __getitem__(self, i):
        audio_raw, audio_features = load_audio(self.audios[i])

        data = {
            "label":          self.Y_s[i],
            "audio_raw":      audio_raw,
            "audio_feats":    audio_features,
            
            "eeg_rest":       self.X_s_rest["feature_value"][i],
            "eeg_active":     self.X_s_active["feature_value"][i],
            "eeg_vocal":      self.X_s_vocal["feature_value"][i],

            "eeg_rest_raw":   self.X_s_rest_raw[i],
            "eeg_active_raw": self.X_s_active_raw[i],
            "eeg_vocal_raw":  self.X_s_vocal_raw[i],

            "eeg_rest_feats":   self.X_s_rest_feats[i],
            "eeg_active_feats": self.X_s_active_feats[i],
            "eeg_vocal_feats":  self.X_s_vocal_feats[i],
        }

        return data
    
    def __len__(self):
        return len(self.Y_s)