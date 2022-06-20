# Autoencoder-Assist-An-Efficient-Profiling-Attack-on-High-dimensional-Datasets

This respritory shows the scripts and dataset used in article:

"Autoencoder Assist: An Efficient Profiling Attack on High-dimensional Datasets" [eprint](https://eprint.iacr.org/2021/1418) 

The feature of the article is to use undercomplete auto-encoder (UAE) to extract key leakage information.

## Datasets

The trace sets used in this article are generated from ASCAD:[https://github.com/ANSSI-FR/ASCAD](https://github.com/ANSSI-FR/ASCAD), which include synchronized datasets and desynchronized datasets. Each dataset composed of L=3000 samples.

* Synchronized dataset: 

$N_D = 0$, interval $(44400,47400)$.

* Desynchronized dataset:

(1) $N_D = 50$ with a random delay of 50 samples.

(2) $N_D = 100$ with a random delay of 100 samples.

## Scripts

We use scripts and models listed as follows for each dataset.

* UAE.py: provides the UAE model for feature extraction.
* MLP.py: describes the MLP model architecture for key exploitation.
* trained_model: contains the best UAE and MLP models in the article.
* fig: contains key rank and loss evolution of the best models.
