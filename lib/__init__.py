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

import torch
import os
import soundfile as sf
import mne

PATH_INBETWEEN = "spoclab/users/szhao/EEG/data/"
DEFAULT_CHANNELS = [
    'FC6', 'FT8', 'C5', 'CP3', 'P3', 'T7', 'CP5', 'C3', 'CP1', 'C4'
]

def load_audio(fname):
    audio, r = sf.read(fname)
    print(r)
    assert r == 16000
    return audio

def read_emg(emg_path, channels_only=[]):
    emg_raw = mne.io.read_raw_cnt(emg_path, preload=True).load_data()
    if channels_only:
        emg_raw.pick_channels(channels_only)
    return emg_raw.get_data()
    

class KaraOneDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root_dir,
            pts=("MM05",)):
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

    def __getitem__(self, i):
        base_dir = os.path.join(
            self.root_dir,
            PATH_INBETWEEN,
            self.pts[0])
        base_dir_files = os.listdir(base_dir)

        try:
            cnt_file = list(filter(lambda x: x.endswith(".cnt"), base_dir_files))[0]
        except Exception as e:
            print("Error loading .cnt file:", e)

        emg_path = os.path.join(
            self.root_dir,
            PATH_INBETWEEN,
            self.pts[0],
            cnt_file)

        data = {
            "label": self.labels[i],
            "audio": load_audio(self.audios[i]),
            "emg":   read_emg(emg_path)
        }
        return data