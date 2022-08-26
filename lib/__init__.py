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
"""Kara One helper library."""

import os
import glob
from time import time

import torch
import mne

import soundfile as sf
import scipy.io as sio

from .extra import *

PATH_INBETWEEN = "spoclab/users/szhao/EEG/data/"
DEFAULT_CHANNELS = [
    'FC6', 'FT8', 'C5', 'CP3', 'P3', 'T7', 'CP5', 'C3', 'CP1', 'C4']
DROP_CHANNELS = [
    'CB1', 'CB2', 'VEO', 'HEO', 'EKG', 'EMG', 'Trigger']

def load_audio(fname):
    audio, r = sf.read(fname)
    print(r)
    assert r == 16000
    return audio

def read_emg(emg_path, channels_only=[], drop_channels=DROP_CHANNELS):
    print("PATH:", emg_path)
    emg_raw = mne.io.read_raw_cnt(emg_path, preload=True) # .load_data()
    emg_raw.drop_channels(DROP_CHANNELS)
    if channels_only:
        emg_raw.pick_channels(channels_only)
    return emg_raw
    
def calculate_features(eeg_data, epoch_inds, prompts, condition_inds, prompts_list, max_prompts=0):
    offset = int(eeg_data.info["sfreq"] / 2)
    X = []
    max_prompts = max_prompts if max_prompts else len(prompts["prompts"][0])
    print("max_prompts:", max_prompts)
    for i, prompt in enumerate(prompts["prompts"][0][0:max_prompts]):
        t0 = time()
        if prompt[0] in prompts_list:
            start = epoch_inds[condition_inds][0][i][0][0] + offset
            end   = epoch_inds[condition_inds][0][i][0][1]
            channel_set = []
            for idx, ch in enumerate(eeg_data.ch_names):
                epoch = eeg_data[idx][0][0][start:end]
                channel_set.extend(fast_feat_array(epoch, ch))
            X.append(channel_set)
        print("Calc: %0.3fs" % (time() - t0), i, prompt)
    return X


class KaraOneDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root_dir,
            pts=("MM05",),
            raw=True,
            max_prompts=0):

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
                sio.loadmat(f, variable_names=('clearing_inds', 'thinking_inds'))

        if raw:
            for f in glob.glob(
                os.path.join(f"{base_dir}", "*.cnt")):
                emg_data = read_emg(f)

        self.emg_data = emg_data

        t0 = time()

        max_prompts = max_prompts if max_prompts else len(self.prompts["prompts"][0])
        self.Y_s = Y_s = self.prompts["prompts"][0][0:max_prompts]
        print("len(self.Y_s):", self.Y_s)
        self.X_s = calculate_features(
            emg_data,
            epoch_inds,
            prompts,
            "clearing_inds",
            Y_s,
            max_prompts)

        print("Calc: %0.3fs" % (time() - t0))

    def __getitem__(self, i):
        data = {
            "label": self.Y_s[i],
            "audio": load_audio(self.audios[i]),
            "emg":   self.X_s[i]
        }

        return data
    
    def __len__(self):
        return len(self.Y_s)