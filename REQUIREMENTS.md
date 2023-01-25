# REQUIREMENTS

We expect that everyone with reasonable programming experience can open and run the artifacts with standard environments.
Part of the artifacts have some package requirements, but there is no deviation from standard environments for data science.

The artifacts are tested with MacBook Pro (14-inch, 2021, Apple M1 Max, 32 GB Memory).

## Google Chrome plugin used for the study: [`/SOREL/Plugin`](https://github.com/namdy0429/SOREL/tree/main/Plugin)

There is no requirement to run this plugin other than having a [Chrome browser](https://www.google.com/chrome/) installed. 
It was test with Chrome Version 109.0.5414.87 (Official Build) (arm64).

## User study analysis replication: [`/SOREL/Study`](https://github.com/namdy0429/SOREL/tree/main/Study)

R Notebook contains the user study result data and scripts to replicate the statistical analysis.
To run this file, R-Notebook should be installed on your machine.
Or, you can run it with a ready-to-run docker image [`jupyter/r-notebook`](https://hub.docker.com/r/jupyter/r-notebook) (see [`INSTALL.md`](https://github.com/namdy0429/SOREL/blob/main/INSTALL.md) for details).

#### R-notebook

- R version 3.6.3 (2020-02-29)
- Platform: aarch64-conda-linux-gnu (64-bit)
- Running under: Ubuntu 22.04.1 LTS

#### R Packages

- MuMIn_1.46.0
- lme4_1.1-31
- lmerTest_3.1-3
- Matrix_1.3-3

A script to install the packages is included in the R-notebook.

## SOREL implementation with Stack Overflow data labeled with comparable API methods: [`/SOREL/SOREL`](https://github.com/namdy0429/SOREL/tree/main/SOREL)

#### Hardware

In our paper, all SOREL training and experiments were done on the same machine, with:

- Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz
- Nvidia RTX 8000 GPU

You will be able to run SOREL with other similar hardware settings for standard machine learning.
You might be able to train and test without a GPU, but we highly recommend to use one.

#### Packages
- Python==3.8.10
- torch==1.12.0+cu113
- torchvision==0.13.0+cu113
- torchaudio==0.12.0+cu113
- transformers==4.21.0.dev0
- itsdangerous<2.1.0
- numpy==1.22.4
- sacred==0.8.2

To run SOREL, PyTorch and other pakcages should be installed on your machine.
Or, you can run it with a docker image [`huggingface/transformers-pytorch-gpu`](https://hub.docker.com/r/huggingface/transformers-pytorch-gpu) (see [`INSTALL.md`](https://github.com/namdy0429/SOREL/blob/main/INSTALL.md) for details).
We highly recommend you to consider using docker image to avoid unexpected issues and dependency clustering.
