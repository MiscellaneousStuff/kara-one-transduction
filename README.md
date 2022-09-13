# Kara One Dataset - Transduction

## About

Assessing the feasibility of applying SOTA sEMG silent speech transduction methods to EEG imagined speech synthesis.

## Kara One

Dataset combines 3 modalities (EEG, face tracking, audio) during imagined and
vocalised phonemic and single-word prompts.

[Paper](http://www.cs.toronto.edu/~complingweb/data/karaOne/ZhaoRudzicz15.pdf)

[Dataset](http://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html)

## Progress

- [x] Overfit on a single example
   - 1 layer, 128 dim LSTM network works extremely well

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