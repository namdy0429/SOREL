# SOREL

## Paper

Daye Nam, Brad Myers, Bogdan Vasilescu, and Vincent Hellendoorn, "Improving API Knowledge Discovery with ML: A Case Study of Comparable API Methods," ICSE 2023.

## Artifacts

Archived version of the artifacts and the documentation can be found here: [https://doi.org/10.5281/zenodo.7570586](https://doi.org/10.5281/zenodo.7570586)

There are three main components in this repository.

#### Google Chrome plugin used for the study: [`/SOREL/Plugin`](https://github.com/namdy0429/SOREL/tree/main/Plugin)

How to install this plugin is included in [`/SOREL/Install.md`](https://github.com/namdy0429/SOREL/blob/main/INSTALL.md), and [`/SOREL/Plugin/howto.png`](https://github.com/namdy0429/SOREL/blob/main/Plugin/howto.png) shows how to use the Plugin.

#### Data and R script for the user study analysis replication: [`/SOREL/Study`](https://github.com/namdy0429/SOREL/tree/main/Study)

You can replicate the statistical analysis included in the paper by running [`/SOREL/Study/Regression.ipynb`](https://github.com/namdy0429/SOREL/blob/main/Study/Regression.ipynb).
User study result data is included in the Notebook as well. [`/SOREL/Install.md`](https://github.com/namdy0429/SOREL/blob/main/INSTALL.md) describes how to set up R-Notebook environment to run this Notebook.

#### SOREL implementation with Stack Overflow data labeled with comparable API methods: [`/SOREL/SOREL`](https://github.com/namdy0429/SOREL/tree/main/SOREL)

You can train and test SOREL, and replicate the experiment results.
[`/SOREL/Install.md`](https://github.com/namdy0429/SOREL/blob/main/INSTALL.md) describes how to set up PyTorch environment using Docker image, and how to train and test SOREL.
