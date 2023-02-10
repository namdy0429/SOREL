# coding: utf-8
import torch
import numpy as np
import random

import models
from BERT_SOREL import BERT_SOREL
from sacred import Experiment
from sacred.observers import FileStorageObserver


torch.manual_seed(33)
np.random.seed(33)
random.seed(33)

ex = Experiment(save_git_info=False)
ex.add_config('config/sorel_config.yaml')

ex.observers.append(FileStorageObserver("./logs"))
Trainer = ex.capture(BERT_SOREL, prefix='bert')

@ex.automain
def sorel(_run, _config, sorel_config):
    model = {
        'BERT_BiLSTM_pos': models.BERT_BiLSTM_pos,
    }

    trainer = Trainer(sorel_config)
    trainer.load_test_data()

    cur_model = model[sorel_config['model']](config = sorel_config)
    if sorel_config['test_model'] != None:
        cur_model.load_state_dict(torch.load(sorel_config['test_model']), strict=False)
    cur_model.cuda()

    trainer.test(cur_model, _run._id)
