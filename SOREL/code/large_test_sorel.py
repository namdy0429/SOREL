# coding: utf-8
import torch
import numpy as np
import random

import models
from BERT_SOREL_TEST import BERT_SOREL_TEST
from sacred import Experiment
from sacred.observers import FileStorageObserver


torch.manual_seed(33)
np.random.seed(33)
random.seed(33)

ex = Experiment()
ex.add_config('config/sorel_config.yaml')

ex.observers.append(FileStorageObserver("./logs"))
Tester = ex.capture(BERT_SOREL_TEST, prefix='bert')

@ex.automain
def my_main(_config, sorel_config):
    model = {
        'BERT_BiLSTM_pos': models.BERT_BiLSTM_pos,
    }
    
    tester = Tester(sorel_config)
    tester.load_test_data()

    cur_model = model[sorel_config['model']](config = sorel_config)
    if sorel_config['test_model'] != None:
        print(sorel_config['test_model'])
        cur_model.load_state_dict(torch.load(sorel_config['test_model']), strict=False)
    cur_model.cuda()

    tester.test(cur_model, sorel_config['model'], ex)