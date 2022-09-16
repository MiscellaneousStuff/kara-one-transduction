# Kara One Dataset - Transduction

## About

Assessing the feasibility of applying SOTA sEMG silent speech transduction methods to EEG imagined speech synthesis.

## Kara One

Dataset combines 3 modalities (EEG, face tracking, audio) during imagined and
vocalised phonemic and single-word prompts.

[Paper](http://www.cs.toronto.edu/~complingweb/data/karaOne/ZhaoRudzicz15.pdf)

[Dataset](http://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html)

## Progress

Below milestones are for MM05:
- [x] Overfit on a single example (EEG imagined speech)
   - 1 layer, 128 dim Bi-LSTM network doesn't work well
     (most likely due to misalignment between imagined EEG signals and audio targets,
     this is a major issue for a transduction network)
- [x] Overfit on a single example (EEG vocalised speech)
   - 1 layer, 128 dim Bi-LSTM network works well
     (seems like the temporal alignment between the vocal EEG signals and the audio
     recordings make it easy to synthesize audio features from parallel vocal EEG signals)
- [x] Overfit on all /tiy/ examples
    - 1 layer, 64 dim Bi-LSTM network works well
      (reducing the hidden dim compared to single samples prevents gradient explosion
       earlier in training. not sure why this happens when increasing task complexity...)
- [x] Overfit on all /m/ and /n/ examples       
    - 1 layer, 64 dim Bi-LSTM network works somewhat well
      (the temporal alignment of the signals is roughly correct however, the specific
       pitches of the utterances are much flatter than the original signal. This
       could be improved using better low level feature detectors, such as ResNet
       blocks as done in
       [An Improved Model for Voicing Slient Speech](https://arxiv.org/abs/2106.01933)).

## Dataset Details

### Epochs

EEG epoching is a procedure where specific time-windows are extracted
from a continuous EEG signal. These time windows are called "epochs"
and are usually time-locked with respect to an event, e.g. a visual stimulus
or in the case of this dataset, imagined speech.

### Citation (BibTeX)

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