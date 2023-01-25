# INSTALL

## Chrome plugin used for the user study
To use the plugin, you have to set up Chrome as a developer mode.

> Click the Chrome menu icon and select Extensions from the Tools menu. Ensure that the "Developer mode" checkbox in the top right-hand corner is checked.

You can find more detailed instructions from [FAQ of Google Chrome Developers Documentation](https://developer.chrome.com/docs/extensions/mv3/faq/).

Now, you can load the plugin by clicking "Load unpacked" buttong in the top left-hand corner, and load the plugin (`Plugin/`). Then, you can enable the plugin by clicking the toggle button in the bottom right-hand corner of the plugin box.


## User study analysis replication

Our statistical analysis on the user study results were written in R, using Jupyter Notebook.
To run the study analysis script, you need to be able to run R-notebook.
If you have Jupyter Notebook already with R language and r-essentials already, simply load `Study/Regression.ipynb` from your notebook interface. The data is already included in the code.

If not, you can use a ready-to-run Docker image to launch Jupyter Notebook environment (no need to download Jupyter Notebook).
Downloading the jupyter/R-notebook docker image will require ~1G of memory space.

```
$ docker run -p 8888:8888 --name notebook -v /home/dayenam/Documents/Research/SO/ICSE/Artifacts:/home/dayenam/Artifacts jupyter/r-notebook
```

## SOREL implementation with Stack Overflow data labeled with comparable API methods

To train and test SOREL implementation, you need to be able to run PyTorch code with a GPU.
If you have PyTorch set up, you should be able to run the code after installing necessary Python packages.

If not, you can use a docker image to set up the environment. Downloading the transformer docker image will require ~15G of space.

```
$ docker pull huggingface/transformers-pytorch-gpu
```

Once you have the docker image, you can run the container with the following command:


```
$ docker run --rm -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -v [path-to-Artifacts]/SOREL/:/SOREL huggingface/transformers-pytorch-gpu
```

- `--rm`: Automatically remove the container when it exits.
- `--it`: Keep STDIN open even if not attached.
- `--runtime=nvidia`: use NVIDIA Container Runtime, a GPU aware container runtime.
- `-e NVIDIA_VISIBLE_DEVICES=0`: Assign GPU-0 to the container. You can check available GPUs with `nvidia-smi` and choose one GPU to assign.
- `-v [path-to-Artifacts]/SOREL/:/SOREL`: Bind mount a volume - tells the daemon where to execute the commands within the container.
- `huggingface/transformers-pytorch-gpu`: Docker image to use.

After you launch the docker container, move to `/SOREL/code` and run the following commands to install python package requirements.

```
$ pip3 install -U pip
$ pip3 install -r requirements.txt
```

Noww, you are ready to train SOREL.  Run the following command to train SOREL.

```
$ python3 train_sorel.py
```

The default number of epoch is set to be 1, for you to test whether you have the right environment to tarin and test SOREL.
If you can see training logs similar to the following, you have the right environment to train SOREL.

```
-----------------------------------------------------------------------------------------
Training Epoch 0
Read 411 documents, 141 with relation
3336 sentences, 75967 tokens
Total relation: 398, total no-relation: 1668
-----------------------------------------------------------------------------------------
Training RE Loss: 0.4632
Training SEP Loss: 0.6529

[RE]
Positive examples: Accuracy: 0.7987 | Precision: 0.2333 | Recall: 0.0186 | F1: 0.0706
True Positive: 7, False Positive: 23, True Negative: 1548, False Negative: 369

[SEP]
Accuracy: 0.6631 | Precision: 0.5000 | Recall: 0.0214 | F1: 0.0913
True Positive: 23, False Positive: 23, True Negative: 2095, False Negative: 1053

Finish training
Best epoch = 0.000000 | Best RE F1 = 0.0000 | Best SEP F1 = 0.0000
INFO - train_sorel - Completed after 0:00:49
```

The trained model, results, and logs will be stored in `/SOREL/code/logs/1`.

Now, to test the test code, run the following command.

```
$ python3 test_sorel.py
```

You should see test output like the following.

```
Test Epoch 0
Test RE Loss: 0.4646
Test SEP Loss: 0.6445

[RE]
Positive examples: Accuracy: 0.7825 | Precision: 0.6818 | Recall: 0.1071 | F1: 0.3289
True Positive: 15, False Positive: 7, True Negative: 460, False Negative: 125

[SEP]
Accuracy: 0.6751 | Precision: 0.6000 | Recall: 0.0517 | F1: 0.1923
True Positive: 21, False Positive: 14, True Negative: 808, False Negative: 385
```

You can find the test results and the logs under `/SOREL/code/logs/2`.

Finally, to run the trained model with the larger dataset without the manual labels, run the following command.

```
$ python3 large_test_sorel.py
```

You should see the output like the following.

```
/SOREL/code/logs/1/BERT_BiLSTM_pos_final_epoch
Read 604 documents, 0 with relation
4841 sentences, 114876 tokens
```

You can find the results under `/SOREL/code/logs/3`.


You have tried the whole pipeline of SOREL, and you can update the config file (`/SOREL/code/config/sorel_config.yaml`) to further explore it.
Detailed explanations for configuration variables are includedi n the config file.
All of your training and testing results of SOREL will be stored unter `/SOREL/code/logs/`, so please delete the ones that are not necessary to prevent them to take all of your spaces.






