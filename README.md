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

## Paprameter Settings

### Experimental Settings

We followed the experimental settings in \citep{wangepsilon_2024_neurips, zhou2021asymmetric_loss, ma2020normalized_normalized}. To ensure reproducibility, we integrated our implementation into the code provided by \cite{wangepsilon_2024_neurips, zhou2021asymmetric_loss}.  We compare the performance of \ant~ with multiple benchmark methods, including standard cross entropy loss (CE), FL \cite{ross2017focal_loss}, MAE loss \cite{ghosh2017robust_MAE_loss}, GCE \cite{zhang2018generalized_GCE_loss}, SCE \cite{wang2019symmetric_SCE_loss}, RCE \citep{wang2019symmetric_RCE_Loss}, APL \cite{ma2020normalized_APL_loss}, AUL \cite{zhou2021asymmetric_loss}, AEL \cite{zhou2021asymmetric_loss}, AGCE \cite{zhou2021asymmetric_loss}, Negative Learning for Noisy Labels (NLNL) \cite{kim2019nlnl_loss}, LDR-KL \cite{zhu2023label_LDR_KL_ICML}, 
LogitClip \cite{wei2023mitigating_LC_loss}, and $\epsilon$-softmax loss \cite{wangepsilon_2024_neurips}. For MAE, RCE, and asymmetric losses (AEL, AGCE, and AUL) we also considered their combination with normalized CE. Following \cite{wangepsilon_2024_neurips,zhou2021asymmetric_loss, ma2020normalized_normalized}, we used an eight-layer convolutional neural net (CNN) for CIFAR-10 and a ResNet-34 for CIFAR-100. The networks were trained for 120 and 200 epochs for CIFAR-10 and CIFAR-100, respectively. The batch size for both CIFAR-10 and CIFAR-100 was $128$. We used the SGD optimizer with momentum 0.9. The weight decay was set to $1e-4$ and $1e-5$ for CIFAR-10 and CIFAR-100, respectively. The initial learning rate was set to $0.05$ for CIFAR-10 and $0.1$ for CIFAR-100. We reduced the learning rate by a factor of 10 at epoch $100$ for both datasets. The gradient norms were clipped to $5.0$ and the minimum allowable value for log was set to $1e-8$. Standard data augmentations were applied on both datasets, including random shift and horizontal flip on CIFAR-10, and random shift, horizontal flip, and random rotation on CIFAR-100. 

For WebVision and ImageNet datasets, the tests follow \cite{wangepsilon_2024_neurips,zhou2021asymmetric_loss, ma2020normalized_normalized}. Consistent with these studies, we used the mini WebVision setting and trained a ResNet-50 using SGD for $250$ epochs with initial learning rate $0.4$, Nesterov
momentum $0.9$ and weight decay $3e-5$ with the batch size of 256. We reduced the learning rate by a factor of 10 at epoch $125$. All images were resized to $224 \times 224$. Typical data augmentations including random width/height shift, color jittering, and horizontal flip were applied. Following \cite{wangepsilon_2024_neurips,zhou2021asymmetric_loss, ma2020normalized_normalized}, we trained the
model on WebVision and evaluated it on the same 50 concepts on the corresponding WebVision and ILSVRC12 validation sets.


To foster reproducibility we provide a detailed list of the (hyper)parameter settings in our experiments below. 

### Symmetric and Asymmetric Noise on CIFAR-10 and CIFAR-100
For \ant~ on CIFAR-10, we set $\kappa=0.05$ for all levels of asymmetric noise. We set $\delta=0.1$ for 10\% asymmetric noise, $\delta=0.2$ for 20\% asymmetric noise, $\delta=0.3$ for 30\% asymmetric noise, and $\delta = 0.35$ for 40\% asymmetric noise. We set $\{\kappa=0.07,\delta = 0.02\}$ for clean data, $\{\kappa=0.05,\delta=0.27\}$ for 20\% symmetric noise, $\{\kappa=0.05,\delta=0.57\}$ for 40\% symmetric noise, $\{\kappa=0.05,\delta = 1\}$ for 60\% symmetric noise, $\{\kappa=0.07, \delta = 1.62\}$ for 80\% symmetric noise.

For \ant~ on CIFAR-100, we set $\kappa=0$ for both symmetric an asymmetric noises. We set $\delta=0.2$ for 10\% asymmetric noise, $\delta=0.4$ for 20\% asymmetric noise, $\delta=0.6$ for 30\% asymmetric noise, and $\delta=1$ for 40\% asymmetric noise. We set $\delta=0.04$ for clean data, $\delta=0.27$ for 20\% symmetric noise, $\delta=0.7$ for 40\% symmetric noise, $\delta=1.2$ for 60\% noise, and $\delta=1.7$ for 80\% asymmetric noise.


For benchmark methods, we used the numbers reported in \cite{wangepsilon_2024_neurips} and the same parameter settings, which follows the parameter settings in \citep{ma2020normalized_normalized,zhou2021asymmetric_loss} matching the settings of the original papers for all benchmark methods. For $CE{}_\epsilon+MAE$, we set $\beta = 5$, $m=1e5$, $\alpha = 0.01$ for CIFAR-10 symmetric noise, and $m=1e3$, $\alpha = 0.02$ for asymmetric noise. For CIFAR-100, we set $\beta = 1$, $m=1e4$ and $\alpha \in \{0.1, 0.05, 0.03, 0.0125, 0.0075\}$ for clean, 20\%, 40\%, 60\%, and 80\% symmetric noise, respectively. For asymmetric noise, we used $m = 1e2$, $\alpha \in \{0.015, 0.007, 0.005, 0.004\}$ for 10\%, 20\%, 30\%, and 40\% asymmetric noise, respectively. For $FL{}_\epsilon+MAE$, we set $\gamma=0.1$ and other parameters were set to be the same as those in $CE{}_\epsilon+MAE$. For FL, $\gamma = 0.5$ was used. For GCE, we set $q = 0.7$ for CIFAR-10, and $q = 0.5$ for clean and 20\% symmetric noise ratio, $q=0.7$ for 40\% and 60\% noise ratio, and $q=0.9$ for 80\% asymmetric noise on CIFAR-100. We set $q = 0.7$ for asymmetric noise on CIFAR-100. For SCE, we set $A=-4$, $\alpha = 0.1$, $\beta = 1$ for CIFAR-10, and $\alpha = 6$, $\beta = 0.1$ for CIFAR-100. For APL (MAE, RCE and RCE), we set $\alpha = 1$, $\beta = 1$ for CIFAR-10, and $\alpha = 10$, $\beta = 0.1$ for CIFAR-100. For AUL, we set $a = 6.3$, $q = 1.5$, $\alpha = 1$, $\beta = 4$ for CIFAR-10, and $a = 6$, $q = 3$, $\alpha = 10$, $\beta = 0.015$ for CIFAR-100. For AGCE, we set $a = 6$, $q = 1.5$, $\alpha = 1$, $\beta = 4$ for CIFAR-10, and $a = 1.8$, $q = 3$, $\alpha = 10$, $\beta = 0.1$ for CIFAR-100.
For AEL, we set $a = 5$, $\alpha = 1$, $\beta = 4$ for CIFAR-10, and $a = 1.5$, $\alpha = 10$, $\beta = 0.1$ for CIFAR-100. For LC, we set $\delta = 1$ for clean, 20\%, and 40\%, symmetric noise, $\delta=1.5$ for 60\% and 80\%symmetric noise on CIFAR-10. $\delta = 2.5$ was used for CIFAR-10 asymmetric noise. We set $\delta = 2.5$ for CIFAR-100 asymmetric noise and $\delta = 0.5$ for symmetric noises. For NLNL, the reported results in their original paper was used. For LDR-KL, we set $\lambda = 10$ for CIFAR-10 and 1 for CIFAR-100. 

### Real-World Noise on WebVision and ImageNet
For \ant, we set $\kappa=0.05$ and $\delta=0.27$. For benchmark methods, we used the numbers reported in \cite{wangepsilon_2024_neurips} and the same parameter settings, which uses the best settings from the original papers for all benchmark methods. For GCE, we set $q = 0.7$. For SCE, we set $A = -4$, $\alpha = 10$, and $\beta = 1$. For RCE, we set $\alpha = 50$, $\beta = 0.1$. For AGCE, we set $a = 1e-5$, $q = 0.5$. For $NCE+AGCE$, we set $a = 2.5$, $q = 3$, $\alpha = 50$, $\beta = 0.1$. For LDR-KL, we set $\lambda = 1$. For $CE{}_\epsilon+MAE$, we set $m = 1e3$, $\alpha = 0.015$, $\beta = 0.3$.

### Human Annotation-Induced Noise on CIFAR-100N
We followed the same settings and results reported in \cite{wangepsilon_2024_neurips}. For \ant~, we used the penalized-optimization variant \eqref{eq:reformulation_opt_delta_KL} with $\kappa=3.0$ and $C=1.2$.
