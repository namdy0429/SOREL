# coding: utf-8
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
from collections import Counter

class Relation_Score(object):
    def __init__(self):
        self._reset()

    def _reset(self):
        # the raw values we care about is how many: total pairs, real relations (as in, not no-relation), predicted as real, 
        # correctly predicted as real, predicted as not real, correctly predicted as not real
        self.origins = []
        self.founds = []
        self.rights = []

    def _compute(self, tp, tn, fp, fn):
        accuracy = 0 if (tp+tn+fp+fn) == 0 else ((tp + tn) / (tp+tn+fp+fn))
        recall = 0 if (tp + fn) == 0 else (tp / (tp + fn))
        precision  = 0 if (tp + fp) == 0 else (tp / (tp + fp))
        f1 = 0. if recall + precision == 0 else (1.25 * precision * recall) / (0.25 * precision + recall) 
        return accuracy, recall, precision, f1
    
    def result(self):
        result_string = "[RE]\n"
        origin_counter = Counter([x for x in self.origins])
        found_counter = Counter([x for x in self.founds])
        right_counter = Counter([x for x in self.rights])

        tp = right_counter.get(1, 0) 
        fp = found_counter.get(1, 0) - right_counter.get(1, 0)
        tn = right_counter.get(0, 0)
        fn = found_counter.get(0, 0) - right_counter.get(0, 0)
        accuracy, recall, precision, f1 = self._compute(tp, tn, fp, fn)
        result_string += "Positive examples: Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f\n"%(accuracy, precision, recall, f1)
        result_string += "True Positive: {}, False Positive: {}, True Negative: {}, False Negative: {}\n".format(tp, fp, tn, fn)
        
        return result_string, accuracy, f1, precision, recall

    def update(self, predict_re, label, val_theta):
        max_probs, pred_label = torch.max(predict_re,dim=-1)
        max_probs = max_probs.data.cpu().numpy()
        pred_label = pred_label.data.cpu().numpy()
        max_probs = max_probs.reshape(-1, 1).squeeze()
        pred_label = pred_label.reshape(-1, 1).squeeze()
        over_threshold = max_probs >= val_theta
        over_threshold = over_threshold.reshape(-1, 1).squeeze()

        y_pred = np.full(label.shape[0]*label.shape[1], -1, dtype=int)
        y_pred[over_threshold] = pred_label[over_threshold]
        y_true = label.contiguous().view(1, -1).squeeze().cpu().numpy()
        y_pred_probs = predict_re.contiguous().view(-1, 2).squeeze().cpu().detach().numpy()
        predicted_true_probs0 = []
        predicted_true_probs1 = []
        for i in range(y_true.shape[0]):
            l = y_true[i]
            if l == 0:
                predicted_true_probs0.append(np.around(y_pred_probs[i][l], decimals=2))
            elif l == 1:
                predicted_true_probs1.append(np.around(y_pred_probs[i][l], decimals=2))
        label_exist = y_true != -100
        preds = y_pred[label_exist]
        labels = y_true[label_exist]  
        self.origins.extend(labels)
        self.founds.extend(preds)
        self.rights.extend([pred for pred,label in zip(preds,labels) if pred == label ])

class Evidence_Score(object):
    def __init__(self):
        self._reset()

    def _reset(self):
        self.origins = []
        self.founds = []
        self.rights = []
        self.preds = []
        self.num_sents = []

    def _compute(self, tp, tn, fp, fn):
        accuracy = 0 if (tp+tn+fp+fn) == 0 else ((tp + tn) / (tp+tn+fp+fn))
        recall = 0 if (tp + fn) == 0 else (tp / (tp + fn))
        precision  = 0 if (tp + fp) == 0 else (tp / (tp + fp))
        f1 = 0. if recall + precision == 0 else (1.25 * precision * recall) / (0.25 * precision + recall) 
        return accuracy, recall, precision, f1

    def result(self, input_theta):
        result_string = "[SEP]\n"

        tp = 0
        fp = 0
        tn = 0
        fn = 0
        
        assert(len(self.origins) == len(self.preds))
        for d in range(len(self.origins)):
            origin = self.origins[d]
            prob = self.preds[d]
            num_sent = self.num_sents[d]
            origin_neg = num_sent - origin

            origin = [i for i,p in enumerate(origin) if p == 1] 
            origin_neg = [i for i,p in enumerate(origin_neg) if p == 1] 
            found = [i for i,p in enumerate(prob) if p >= input_theta]

            tp += len(list(set(found) & set(origin)))
            fp += len(list(set(found) - set(origin)))
            tn += len(list(set(origin_neg) - set(found)))
            fn += len(list(set(origin) - set(found)))

        total_accuracy, total_recall, total_precision, total_f1 = self._compute(tp, tn, fp, fn)
        result_string += 'Accuracy: {:3.4f} | Precision: {:3.4f} | Recall: {:3.4f} | F1: {:3.4f}\n'.format(total_accuracy, total_precision, total_recall, total_f1)
        result_string += "True Positive: {}, False Positive: {}, True Negative: {}, False Negative: {}\n".format(tp, fp, tn, fn)
        return result_string, total_recall, total_f1, total_precision, total_recall

    def update(self, pred_probs, label_evidences, num_sent):
        for i in range(pred_probs.shape[0]):
            for j in range(pred_probs.shape[1]):
                self.origins.append(label_evidences[i, j, :].data.cpu().numpy())
                self.preds.append(pred_probs[i, j, :].data.cpu().numpy())
                self.num_sents.append(num_sent[i, j, :].data.cpu().numpy())

class AverageMeter(object):
    '''
    computes and stores the average and current value
    '''
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self,val,n = 1):
        self.val  = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
