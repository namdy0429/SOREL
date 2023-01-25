import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, time, json, random, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from config.utils import *

class BERT_SOREL_TEST(object):
    def __init__(self, config):
        self.config = config

    def load_test_data(self):
        print("Reading test data...")
        
        self.test_file = json.load(open(os.path.join(self.config['data_path'], self.config['large_test_data'])))
        print("Finish reading")
        self.data_train_word = np.zeros((len(self.test_file), self.config['max_length']), dtype = np.int64)
        self.data_train_pos = np.zeros((len(self.test_file), self.config['max_length']), dtype = np.int64)
        for i, ins in enumerate(self.test_file):
            self.data_train_word[i][0:len(ins["sent_idxs"])] = np.array(ins["sent_idxs"])
            self.data_train_pos[i][0:len(ins["sent_pos"])] = np.array(ins["sent_pos"])
        
        self.val_len = math.floor(len(self.test_file) * self.config['val_ratio'])
        self.train_len = len(self.test_file) - self.val_len

        shuffled_docs = list(range(len(self.test_file)))
        random.seed(self.config['seed'])
        random.shuffle(shuffled_docs)

        self.train_order = shuffled_docs[:-self.val_len]
        self.val_order = shuffled_docs[len(shuffled_docs)-self.val_len:]   
        
        self.train_batches = self.train_len // self.config['batch_size']
        if self.train_len % self.config['batch_size'] != 0:
            self.train_batches += 1

        self.val_batches = self.val_len // self.config['batch_size']
        if self.val_len % self.config['batch_size'] != 0:
            self.val_batches += 1

        for b in range(self.train_batches):
            start_id = b * self.config['batch_size']
            cur_bsz = min(self.config['batch_size'], self.train_len - start_id)
            cur_batch = list(self.train_order[start_id: start_id + cur_bsz])


    def get_test_batch(self):
        context_idxs = torch.LongTensor(self.config['batch_size'], self.config['max_length']).cuda()
        context_pos = torch.LongTensor(self.config['batch_size'], self.config['max_length']).cuda()
        h_mapping = torch.Tensor(self.config['batch_size'], self.config['h_t_limit'], self.config['max_length']).cuda()
        t_mapping = torch.Tensor(self.config['batch_size'], self.config['h_t_limit'], self.config['max_length']).cuda()
        relation_mask = torch.Tensor(self.config['batch_size'], self.config['h_t_limit']).cuda()
        sent_mask = torch.Tensor(self.config['batch_size'], self.config['h_t_limit'], self.config['sent_limit']).cuda()

        sent_h_mapping = torch.Tensor(self.config['batch_size'], self.config['sent_limit'], self.config['max_length']).cuda()

        indexes = [0]*self.config['batch_size']
        raw_sentences = [[]]*self.config['batch_size']
        evidences = [''] * self.config['batch_size']
        num_vertices = [0] * self.config['batch_size']
        titles = [''] * self.config['batch_size']

        idx2term = {}
        index_with_rels = []
        num_sentences = 0
        num_tokens = 0


        for b in range(self.val_batches):
            start_id = b * self.config['batch_size']
            cur_bsz = min(self.config['batch_size'], self.val_len - start_id)
            cur_batch = list(self.val_order[start_id : start_id + cur_bsz])
            cur_batch.sort(key=lambda x: np.sum(self.data_train_word[x]>0) , reverse = True)

            for mapping in [h_mapping, t_mapping, relation_mask, sent_mask]:
                mapping.zero_()

            max_sents = 0
            max_h_t_cnt = 1

            for i, index in enumerate(cur_batch):
                ins = self.test_file[index]
                if len(ins['sent_ends']) > self.config['sent_limit']:
                    cur_bsz -= 1
                    continue

                cur_evidences = []
                context_idxs[i].copy_(torch.from_numpy(self.data_train_word[index, :self.config['max_length']]))
                context_pos[i].copy_(torch.from_numpy(self.data_train_pos[index, :self.config['max_length']]))

                raw_sentences[i] = ins["sents"]
                num_sentences += len(ins['sent_ends'])-1
                max_sents = max(max_sents, len(ins['sent_ends'])-1)

                num_vertex = len(ins['vertexSet'])
                titles[i] = ins['title']


                for e in range(len(ins['sent_ends']) - 1):
                    sent_h_mapping[i, e, ins['sent_ends'][e]] = 1


                j = 0
                for h_idx in range(num_vertex):
                    idx2term[(ins['title'], h_idx)] = ins['vertexSet'][h_idx][0]["name"].lower()
                    for t_idx in range(num_vertex):
                        if h_idx != t_idx:
                            hlist = ins['vertexSet'][h_idx]
                            tlist = ins['vertexSet'][t_idx]

                            for h in hlist:
                                h_mapping[i, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(hlist) / (h['pos'][1] - h['pos'][0])
                            for t in tlist:
                                t_mapping[i, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(tlist) / (t['pos'][1] - t['pos'][0])

                            relation_mask[i, j] = 1
                        
                            for k in range(len(ins['sent_ends'])-1):
                                sent_mask[i, j, k] = 1


                            j += 1

                max_h_t_cnt = max(max_h_t_cnt, j)

                num_vertices[i] = num_vertex
                indexes[i] = index
                evidences[i] = cur_evidences

            input_lengths = (context_idxs[:cur_bsz] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max())
            num_tokens += sum(input_lengths)

            yield {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'context_pos': context_pos[:cur_bsz, :max_c_len].contiguous(),
                   'h_mapping': h_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   't_mapping': t_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   'num_vertices': num_vertices,
                   'relation_mask': relation_mask[:cur_bsz, :max_h_t_cnt],
                   'input_lengths': input_lengths,
                   'titles': titles[:cur_bsz],
                   'indexes': indexes[:cur_bsz],
                   'idx2term': idx2term,
                   'sent_mask': sent_mask[:cur_bsz, :max_h_t_cnt, :max_sents],
                   'raw_sentences': raw_sentences[:cur_bsz],
                   'sent_h_mapping': sent_h_mapping[:cur_bsz, :max_sents, :max_c_len],
                   }
        print("Read {} documents, {} with relation".format(self.val_len, len(index_with_rels)))
        print("{} sentences, {} tokens".format(num_sentences, num_tokens))


    def test(self, model, model_name, sacred_ex, output=False, input_theta=-1, is_val_test=False):
        model.cuda()
        model.eval()

        test_result = []

        def logging(s, print_=True, log_=False):
            if print_:
                print(s)
            if log_:
                with open(os.path.join(os.path.join("log", "_validation")), 'a+') as f_log:
                        f_log.write(s + '\n')


        for batch_id, data in enumerate(self.get_test_batch()):
            with torch.no_grad():
                context_idxs = data['context_idxs']
                context_pos = data['context_pos']
                h_mapping = data['h_mapping']
                t_mapping = data['t_mapping']
                num_vertices = data['num_vertices']
                idx2term = data['idx2term']
                batch_size = context_idxs.shape[0]
                titles = data['titles']
                indexes = data['indexes']
                sent_mask = data['sent_mask']
                raw_sentences = data['raw_sentences']
                input_lengths = data['input_lengths']

                sent_h_mapping = data['sent_h_mapping']

                predict_re, predict_sent = model(context_idxs, h_mapping, t_mapping, sent_h_mapping ,None, context_pos, input_lengths)

                predict_re = torch.exp(predict_re)
                predict_sent = torch.sigmoid(predict_sent) * sent_mask
                predict_sent = predict_sent.data.cpu().numpy()
                
                context_idxs = context_idxs.data.cpu()

            predict_re = predict_re.data.cpu().numpy()

            for i in range(batch_size):
                index = indexes[i]

                l = num_vertices[i]
                j = 0

                for h_idx in range(l):
                    for t_idx in range(l):
                        if h_idx != t_idx:
                            r = int(np.argmax(predict_re[i, j]))
                            sent_pred = [float(s) for s in predict_sent[i, j][:int(torch.sum(sent_mask[i, j]).item())]]
                            if self.config['val_theta'] >= 0.5:
                                if r == 0:
                                    continue
                    
                            test_result.append(( \
                                titles[i], \
                                "S1", \
                                index, \
                                h_idx, \
                                t_idx, \
                                r, \
                                [float(predict_re[i,j,k]) for k in range(self.config['so_relation_num'])], \
                                [raw_sentences[i][s].encode('utf-8') for s, prob in enumerate(sent_pred) if prob > self.config['val_sentence_theta']], \
                                sent_pred \
                                ) )
                            j += 1

        test_result.sort(key = lambda x: x[6][x[5]]*max(x[8]) if len(x[8])!=0 else x[6][x[5]], reverse=True)
        logging('-' * 89)

        if len(test_result) < 2:
            return [0], 0, [0], [0], 0, 0, 0

        vw = 0

        for i, item in enumerate(test_result):
            if item[6][item[5]] > self.config['val_theta']:
                vw = i

        output = []

        if self.config['test_no_dup']:
            reported_pairs = []
            with open(self.config['test_result_filename'], "w") as f:
                for x in test_result[:vw+1]:
                    if (x[0], x[4], x[3]) not in reported_pairs:
                        f.write("{}\n{}\t{}\n{}\t{}\n".format("https://stackoverflow.com/questions/"+x[0], idx2term[(x[0],x[3])], idx2term[(x[0],x[4])], x[6][1], max(x[8]) if len(x[8])!=0 else 0))
                        f.write("{}\n{}\n{}\n\n".format(x[6], x[7], x[8]))
                        output.append({'title': x[0], 'h': idx2term[(x[0],x[3])], 't': idx2term[(x[0],x[4])], "prob": x[6][1], "sentences": x[6]})
                        reported_pairs.append((x[0], x[3], x[4]))
                f.write("="*80)

                for x in test_result[vw+1:]:
                    f.write("{}\n{}\t{}\n{}\t{}\n".format("https://stackoverflow.com/questions/"+x[0], idx2term[(x[0],x[3])], idx2term[(x[0],x[4])], x[6][1], max(x[8]) if len(x[8])!=0 else 0))
                    f.write("{}\n{}\n{}\n\n".format(x[6], [s.encode('utf-8') for s in x[7]], x[8]))

        else:
            with open(self.config['test_result_filename'], "w") as f:
                for x in test_result[:vw+1]:
                    f.write("{}\n{}\t{}\n{}\t{}\n".format("https://stackoverflow.com/questions/"+x[0], idx2term[(x[0],x[3])], idx2term[(x[0],x[4])], x[6][1], max(x[8]) if len(x[8])!=0 else 0))
                    f.write("{}\n{}\n{}\n\n".format(x[6], x[7], x[8]))
                    output.append({'title': x[0], 'h': idx2term[(x[0],x[3])], 't': idx2term[(x[0],x[4])], "prob": x[6][1], "sentences": x[6]})
                f.write("="*80)

                for x in test_result[vw+1:]:
                    f.write("{}\n{}\t{}\n{}\t{}\n".format("https://stackoverflow.com/questions/"+x[0], idx2term[(x[0],x[3])], idx2term[(x[0],x[4])], x[6][1], max(x[8]) if len(x[8])!=0 else 0))
                    f.write("{}\n{}\n{}\n\n".format(x[6], [s.encode('utf-8') for s in x[7]], x[8]))                

        json.dump(output, open(self.config['test_result_json'], "w"))
        sacred_ex.add_artifact(self.config['test_result_filename'])
        sacred_ex.add_artifact(self.config['test_result_json'])

