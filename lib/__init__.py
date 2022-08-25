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

PATH_INBETWEEN = "spoclab/users/szhao/EEG/data/"

def load_audio(fname):
    audio, r = sf.read(fname)
    print(r)
    assert r == 16000
    return audio

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
        data = {
            "label": self.labels[i],
            "audio": load_audio(self.audios[i])
        }
        return data