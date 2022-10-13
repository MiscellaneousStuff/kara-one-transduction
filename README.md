# EEG Imagined Speech - Transduction

## About

Assessing the feasibility of applying SOTA sEMG silent speech transduction methods to EEG imagined speech synthesis.

## FEIS Dataset
Dataset Emotiv EPOC+ 14-Channel Wireless EEG headset. Combines EEG and audio
during imagined and vocalised phonemes. Contains English and Chinese data.

Experimental runs:
[Public Neptune.ai](https://app.neptune.ai/miscellaneousstuff/imagined-speech-feis/experiments?split=tbl&dash=charts&viewId=9760c3bc-4c53-41e5-82c6-9764b1aa3d61)

### Progress

Below milestones are for participant 01.
- [x] Synthesize stimuli, vocal and imagined speech across multiple phonemes
   - 2 heads, 8 layers TransformerEncoder works very well
     (stimuli and vocal work extremely well and surprisingly, imagined speech
      synthesis shows promise. This might be working better on this dataset
      due to the emphasis on temporal alignment during the experimental
      condition).

## Kara One

Dataset combines 3 modalities (EEG, face tracking, audio) during imagined and
vocalised phonemic and single-word prompts.

[Paper](http://www.cs.toronto.edu/~complingweb/data/karaOne/ZhaoRudzicz15.pdf)

[Dataset](http://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html)

### Progress

Below milestones are for MM05:
- [x] Overfit on a single example (EEG imagined speech)
   - 1 layer, 128 dim Bi-LSTM network doesn't work well
     (most likely due to misalignment between imagined EEG signals and audio targets,
     this is a major issue for a transduction network)
- [x] Overfit on a single example (EEG vocalised speech)
   - 1 layer, 128 dim Bi-LSTM network works well
     (seems like the temporal alignment between the vocal EEG signals and the audio
     recordings make it easy to synthesize audio features from parallel vocal EEG signals)
- [x] Overfit on all /tiy/ examples (EEG vocalised speech)
    - 1 layer, 64 dim Bi-LSTM network works well
      (reducing the hidden dim compared to single samples prevents gradient explosion
       earlier in training. not sure why this happens when increasing task complexity...)
- [x] Overfit on all /m/ and /n/ examples (EEG vocalised speech)
    - 1 layer, 64 dim Bi-LSTM network works somewhat well
      (the temporal alignment of the signals is roughly correct however, the specific
       pitches of the utterances are much flatter than the original signal. This
       could be improved using better low level feature detectors, such as ResNet
       blocks as done in
       [An Improved Model for Voicing Slient Speech](https://arxiv.org/abs/2106.01933)).
- [x] Generalise on /n/ examples (EEG imagined speech)
   - 1 layer, 64-dim Bi-LSTM network needs improvement
     (the amplitude and mel spectrogram are predicted correctly, but the temporal
      alignment, duration and pitch waveform are incorrect)
- [x] Generalise on /m/ examples (EEG vocal speech)
   - 2 layer, 128 dim Bi-LSTM network works well
     (temporal alignment, duration and pitch waveform issues from before are resolved
      by using an LSTM network with a larger hidden dim. Using multiple LSTM layers
      may allow the network to process the EEG signals hierarchically. There is 
      evidence from the literature that EEG signals are arranged hierarchically,
      so there is rationale for using multiple LSTM layers for this purpose.
      ([paper 1](https://pubmed.ncbi.nlm.nih.gov/22361076/),
      [paper 2](https://www.sciencedirect.com/science/article/abs/pii/S1053811905025140)))
- [x] Transformers on /m/ examples (EEG vocal speech)
    - 2 heads, 8 layers TransformerEncoder works very well
      (maps vocal EEG to audio features very early during training and with very high
       temporal and spatial accuracy. Last thing after this would be using ResNet blocks
       to learn EEG features end-to-end )

### Dataset Details

#### Epochs

EEG epoching is a procedure where specific time-windows are extracted
from a continuous EEG signal. These time windows are called "epochs"
and are usually time-locked with respect to an event, e.g. a visual stimulus
or in the case of this dataset, imagined speech.

#### Citation (BibTeX)

```tex
@INPROCEEDINGS{7178118,
  author={Zhao, Shunan and Rudzicz, Frank},
  booktitle={2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Classifying phonological categories in imagined and articulated speech}, 
  year={2015},
  volume={},
  number={},
  pages={992-996},
  doi={10.1109/ICASSP.2015.7178118}}
```