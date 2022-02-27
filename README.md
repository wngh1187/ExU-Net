# ExU-Net

Pytorch code for following paper:

* **Title** : Extended U-Net for Speaker Verification in Noisy Environments (TBA) 
* **Autor** : Ju-ho Kim, Jungwoo Heo, Hye-jin Shim, and Ha-Jin Yu

# Abstract

Background noise is one of the well-known factors that deteriorates the accuracy and reliability of the speaker verification (SV) systems by blurring the intelligibility of speech. 
Various studies have used a separately pre-trained enhancement model as the front-end module of the SV system to compensate for noise utterance. 
However, independent enhancement approaches that are not customized for downstream task may cause the distortion of the speaker information included in utterances, which can adversely affect the performance. 
In order to alleviate this issue, we argue that the enhancement network and the speaker embedding extractor should be fully jointly trained. 
Therefore, this literature proposes the integrated framework that is simultaneously optimized speaker identification and feature enhancement losses, based on U-Net. 
Moreover, we analyzed the structural limitations of using U-Net directly for noise SV task and further proposed *Extended U-Net* to improve these drawbacks. 
We evaluated our models on the noise-synthesized VoxCeleb1 test set and the VOiCES development set recorded in various noisy environments. 
Experimental results demonstrate that U-Net-based joint training framework is effective compared to baseline for noise utterances, and the proposed Extended U-Net exhibited state-of-the-art performance versus the recently proposed compensation systems. 

# Prerequisites

## Environment Setting
* We used 'nvcr.io/nvidia/pytorch:21.04-py3' image of Nvidia GPU Cloud for conducting our experiments. 
* We used three Titan RTX GPUs for training. 
* Python 3.6.9
* Pytorch 1.8.1
* Torchaudio 0.8.1

## Datasets

We used VoxCeleb1 dataset for training and test. 
For noise synthesis, we used the MUSAN corpus.


# Training

```
Go into run directory
Activate the code you want in train.sh
./train.sh
```

# Test

```
Go into run directory
Activate the code you want in test.sh
./test.sh
```

# Citation
TBA

