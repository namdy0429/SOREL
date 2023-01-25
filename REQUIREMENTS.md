# REQUIREMENTS

We expect that everyone with reasonable programming experience can open and run the artifacts with standard environments.
Part of the artifacts have some package requirements, but there is no deviation from standard environments for data science.
We provide instructions on how to use docker for users' convenience.

The artifacts are tested with MacBook Pro (14-inch, 2021, Apple M1 Max, 32 GB Memory).

## Google Chrome plugin used for the study

There is no requirement to run this plugin other than a Chrome browser.

## User study analysis replication

R Notebook contains the user study result data and scripts to replicate the statistical analysis.
To run this file, R-Notebook should be installed on your machine.
Or, you can run it with a ready-to-run docker image `jupyter/r-notebook` (see `INSTALL.md` for details).

#### R-notebook

- R version 3.6.3 (2020-02-29)
- Platform: aarch64-conda-linux-gnu (64-bit)
- Running under: Ubuntu 22.04.1 LTS

#### R Packages

- MuMIn_1.46.0
- lme4_1.1-31
- lmerTest_3.1-3
- Matrix_1.3-3

A script to install the packages is included.

## SOREL implementation with Stack Overflow data labeled with comparable API methods

#### Packages

- torch==1.12.0+cu113
- torchvision==0.13.0+cu113
- torchaudio==0.12.0+cu113
- transformers==4.21.0.dev0
- itsdangerous<2.1.0
- numpy==1.22.4
- sacred==0.8.2

To run SOREL, PyTorch and other pakcages should be installed on your machine.
Or, you can run it with a ready-to-run docker image `huggingface/transformers-pytorch-gpu` (see `INSTALL.md` for details).
We highly recommend you to consider using docker image to avoid unexpected issues and dependency clustering.
