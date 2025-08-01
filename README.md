# Learning to Forget: Information Divergence Reweighted Losses for Learning with Noisy Labels

This repository is the official **pytorch** implementation of ANTIDOTE. We built on the code repository
provided by (https://openreview.net/pdf?id=vjsd8Bcipv) [NeurIPS2024].

## How to use

**Benchmark Datasets:** The running file is `main_antidote.py`. 
* --dataset: cifar10 | cifar100, etc.
* --loss: antidote_kl_bisec, ECEandMAE, EFLandMAE, GCE, etc.
* --noise_type: symmetric | asymmetric

**Real-World Datasets:** The running file is `main_real_world_antidote.py`. 
* --dataset: webvision | clothing1m.
* --loss: ECEandMAE, EFLandMAE, CE, GCE, etc.

## Examples

For cifar10 or cifar100
```console
$ python3 main_antidote.py --dataset cifar10 --noise_type symmetric --noise_rate 0.4 --loss antidote_kl_bisec    
```

For webvision and ImageNet:
```console
$ python3 main_real_world_antidote.py --dataset webvision --loss antidote_kl_bisec
```

