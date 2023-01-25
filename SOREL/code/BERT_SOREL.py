from statistics import mean
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, time, json, random, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers.optimization import get_scheduler, get_linear_schedule_with_warmup

from config.utils import *

import shutil

class BERT_SOREL(object):
    def __init__(self, config):
        self.config = config

        self.epoch = 0
        self.decreasing_val_metrics = 0
        self.best_val_metric = 0
        self.best_val_loss = 0
        self.facts = []
        
        self.train_relation_score = Relation_Score()
        self.val_relation_score = Relation_Score()
        self.train_evidence_score = Evidence_Score()
        self.val_evidence_score = Evidence_Score()

    def _reset(self):
        self.train_relation_score = Relation_Score()
        self.train_evidence_score = Evidence_Score()
        self.val_relation_score = Relation_Score()
        self.val_evidence_score = Evidence_Score()

    def load_train_data(self):
        print("Reading training data...")
        
        self.train_file = json.load(open(os.path.join(self.config['data_path'], self.config['train_data'])))
        print("Finish reading")
        self.data_train_word = np.zeros((len(self.train_file), self.config['max_length']), dtype = np.int64)
        self.data_train_pos = np.zeros((len(self.train_file), self.config['max_length']), dtype = np.int64)
        for i, ins in enumerate(self.train_file):
            self.data_train_word[i][0:len(ins["sent_idxs"])] = np.array(ins["sent_idxs"])
            self.data_train_pos[i][0:len(ins["sent_pos"])] = np.array(ins["sent_pos"])
        
        self.val_len = math.floor(len(self.train_file) * self.config['val_ratio'])
        self.train_len = len(self.train_file) - self.val_len

        shuffled_docs = list(range(len(self.train_file)))
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

            cur_batch_facts = []
            for i, index in enumerate(cur_batch):
                ins = self.train_file[index]
                labels = ins['labels']

                for label in labels:
                    cur_batch_facts.append((ins['vertexSet'][label['h']][0]["name"].lower(), ins['vertexSet'][label['t']][0]["name"].lower(), label['r']))
                    self.facts.append((ins['vertexSet'][label['h']][0]["name"].lower(), ins['vertexSet'][label['t']][0]["name"].lower(), label['r']))

    def load_fold_train_data(self):
        print("Reading training data...")

        if self.config['size_ablation'] == 'full':
        # Full
            train_split = [[0, 9, 15, 26, 28, 30, 32, 43, 50, 52, 57, 58, 94, 96, 98, 105, 109, 110, 152, 155, 157, 159, 168, 169, 173, 184, 209, 210, 213, 214, 222, 225, 228, 250, 263, 269, 277, 279, 280, 285, 286, 295, 299, 300, 319, 332, 333, 342, 359, 372, 374, 388, 393, 397, 400, 401, 403, 412, 413, 422, 426, 432, 436, 440, 441, 446, 448, 466, 468, 476, 485, 489, 496, 500, 516, 526, 527, 532, 534, 538, 542, 551, 555, 561, 567, 569, 571, 572, 575, 576, 577, 580, 582, 583], [18, 23, 25, 29, 38, 42, 46, 48, 53, 66, 72, 74, 76, 77, 80, 97, 106, 114, 119, 122, 127, 133, 138, 144, 148, 167, 177, 180, 191, 200, 206, 211, 217, 226, 232, 236, 244, 246, 249, 251, 254, 262, 265, 278, 281, 282, 287, 288, 297, 302, 303, 304, 306, 311, 318, 324, 335, 338, 341, 352, 356, 364, 365, 366, 377, 380, 394, 404, 408, 416, 417, 419, 433, 442, 444, 452, 453, 459, 463, 475, 482, 487, 488, 506, 515, 518, 519, 522, 543, 547, 550, 552, 553, 584], [2, 3, 4, 17, 45, 59, 60, 78, 79, 82, 85, 86, 87, 102, 103, 107, 111, 118, 123, 131, 134, 135, 142, 145, 146, 154, 162, 163, 176, 187, 190, 207, 220, 224, 227, 237, 259, 261, 266, 270, 274, 283, 290, 291, 292, 294, 305, 307, 313, 339, 347, 348, 350, 358, 362, 368, 369, 371, 383, 385, 395, 406, 414, 420, 421, 423, 430, 435, 449, 450, 458, 464, 467, 471, 474, 477, 480, 484, 491, 495, 502, 507, 508, 509, 513, 520, 521, 525, 540, 556, 557, 559, 570, 585], [6, 11, 14, 21, 24, 31, 36, 49, 55, 62, 67, 68, 73, 89, 91, 100, 115, 121, 143, 147, 149, 150, 158, 171, 174, 179, 182, 185, 189, 192, 194, 196, 197, 199, 201, 203, 205, 212, 216, 221, 223, 230, 231, 243, 245, 255, 272, 273, 289, 296, 312, 315, 321, 322, 323, 325, 327, 328, 337, 340, 344, 353, 386, 390, 396, 399, 402, 405, 428, 429, 438, 445, 451, 465, 469, 472, 473, 478, 490, 501, 504, 511, 512, 517, 529, 531, 535, 536, 541, 546, 564, 573, 574, 578], [1, 5, 8, 10, 12, 33, 34, 35, 37, 39, 40, 44, 54, 61, 64, 65, 70, 75, 81, 93, 95, 101, 112, 113, 120, 129, 130, 136, 137, 156, 161, 170, 172, 181, 183, 186, 195, 202, 208, 218, 219, 229, 240, 241, 248, 256, 257, 264, 268, 271, 276, 301, 308, 314, 326, 331, 334, 343, 345, 346, 373, 376, 378, 379, 381, 382, 384, 387, 392, 398, 407, 425, 431, 443, 460, 470, 486, 492, 494, 498, 503, 505, 510, 514, 528, 530, 548, 549, 554, 558, 566, 568, 579, 586]]
        elif self.config['size_ablation'] == '2/3':
        # 2/3
            train_split = [[26, 57, 58, 66, 81, 112, 115, 118, 121, 123, 131, 135, 138, 142, 152, 163, 176, 177, 184, 185, 187, 189, 194, 205, 208, 218, 221, 224, 240, 248, 251, 272, 273, 282, 285, 303, 324, 339, 341, 342, 353, 379, 380, 402, 416, 430, 432, 441, 444, 446, 448, 471, 477, 487, 488, 489, 494, 495, 525, 532, 550, 584, 585], [0, 2, 11, 14, 17, 25, 42, 44, 54, 59, 64, 78, 80, 103, 120, 136, 157, 180, 182, 199, 230, 236, 270, 271, 274, 279, 301, 305, 314, 322, 323, 356, 358, 383, 385, 386, 387, 392, 401, 406, 414, 422, 435, 438, 452, 453, 460, 463, 464, 465, 476, 478, 486, 512, 518, 531, 542, 553, 557, 561, 570, 579, 586], [3, 24, 29, 32, 36, 39, 45, 65, 67, 70, 79, 100, 107, 147, 161, 169, 179, 195, 202, 212, 213, 214, 217, 220, 226, 231, 232, 243, 255, 256, 276, 302, 304, 307, 319, 325, 333, 334, 346, 348, 373, 377, 413, 436, 445, 449, 459, 467, 468, 470, 482, 502, 505, 513, 514, 529, 541, 549, 559, 566, 575, 577, 583], [1, 9, 10, 23, 50, 52, 72, 75, 76, 77, 97, 98, 105, 114, 119, 122, 129, 145, 149, 158, 159, 170, 171, 173, 174, 181, 183, 197, 210, 228, 241, 245, 280, 281, 292, 295, 297, 313, 315, 345, 362, 368, 371, 376, 378, 382, 393, 396, 400, 419, 431, 473, 474, 490, 500, 509, 511, 527, 538, 552, 558, 569, 574], [15, 35, 40, 48, 55, 61, 86, 87, 91, 93, 101, 102, 111, 113, 127, 130, 143, 167, 172, 192, 196, 200, 209, 216, 219, 222, 223, 225, 229, 237, 244, 265, 266, 268, 269, 283, 294, 308, 335, 337, 347, 352, 365, 366, 381, 390, 397, 405, 407, 408, 429, 475, 485, 498, 508, 526, 534, 535, 540, 547, 554, 568, 578]]
        elif self.config['size_ablation'] == '1/3':
        # 1/3
            train_split = [[45, 52, 55, 66, 73, 94, 112, 150, 152, 189, 244, 272, 279, 282, 288, 306, 327, 341, 362, 364, 388, 406, 443, 451, 463, 474, 475, 526, 553, 555, 561], [2, 3, 34, 49, 64, 65, 82, 107, 157, 158, 181, 197, 241, 271, 283, 290, 299, 314, 319, 339, 408, 441, 460, 482, 492, 508, 516, 554, 557, 566, 575], [5, 28, 30, 32, 42, 75, 78, 89, 155, 173, 187, 228, 229, 248, 251, 303, 307, 308, 394, 480, 488, 494, 506, 509, 522, 528, 548, 559, 567, 576, 585], [33, 44, 50, 58, 70, 95, 114, 134, 162, 190, 192, 194, 196, 221, 225, 264, 295, 311, 334, 423, 438, 467, 490, 504, 518, 521, 525, 538, 547, 556, 568], [14, 59, 74, 80, 136, 149, 159, 177, 184, 205, 206, 207, 277, 315, 322, 332, 350, 352, 359, 373, 401, 414, 422, 432, 435, 442, 458, 501, 531, 580, 583]]
        elif self.config['size_ablation'] == '1/10':
        # 1/10
            train_split = [[62, 94, 144, 296, 315, 393, 428, 435, 506], [40, 98, 212, 283, 364, 442, 450, 473, 586], [6, 110, 230, 279, 394, 491, 540, 564, 579], [64, 138, 180, 197, 265, 331, 341, 356, 383], [30, 130, 221, 243, 408, 413, 488, 496, 510]]

        self.train_file = json.load(open(os.path.join(self.config['data_path'], self.config['train_data'])))
        print("Finish reading")
        self.data_train_word = np.zeros((len(self.train_file), self.config['max_length']), dtype = np.int64)
        self.data_train_pos = np.zeros((len(self.train_file), self.config['max_length']), dtype = np.int64)
        for i, ins in enumerate(self.train_file):
            self.data_train_word[i][0:len(ins["sent_idxs"])] = np.array(ins["sent_idxs"])
            self.data_train_pos[i][0:len(ins["sent_pos"])] = np.array(ins["sent_pos"])
        
        if self.config['fold_val_idx'] == 5:
            self.val_len = 0
            self.train_order = [item for split in train_split for item in split]
            self.train_len = len(self.train_order)
            self.val_order = []
        else:
            self.val_len = len(train_split[self.config['fold_val_idx']])

            random.seed(self.config['seed'])

            self.train_order = [item for i in range(5) for item in train_split[i] if i != self.config['fold_val_idx']]
            self.train_len = len(self.train_order)
            self.val_order = train_split[self.config['fold_val_idx']]

        random.shuffle(self.train_order)
        random.shuffle(self.val_order)
        
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

            # keep the facts (two entities and the relations) that are in the train set
            # it will be used to exclude entity pairs in test set later
            cur_batch_facts = []
            for i, index in enumerate(cur_batch):
                ins = self.train_file[index]
                labels = ins['labels']

                for label in labels:
                    cur_batch_facts.append((ins['vertexSet'][label['h']][0]["name"].lower(), ins['vertexSet'][label['t']][0]["name"].lower(), label['r']))
                    self.facts.append((ins['vertexSet'][label['h']][0]["name"].lower(), ins['vertexSet'][label['t']][0]["name"].lower(), label['r']))

    def load_validate_data(self):
        print("Reading training data...")
        with_label_docs = ['55221698', '42539696', '53902375', '44809583', '45186929', '53197786', '34596212', '50068955', '48347687', '41483033', '48006315', '44654244', '52055189', '37603765', '50049491', '58574655', '55796211', '37624060', '62771771', '55318851', '35530400', '42965020', '63669319', '47143624', '59953445', '42774074', '50080463', '34243720', '46525453', '60409372', '50259637', '50259737', '48340802', '47984274', '51089448', '50048405', '45960834', '41628828', '53791390', '62979584', '48472420', '53633664', '64080452', '47249767', '45459557', '46040638', '40981831', '37377303', '42932979', '51898852', '59454480', '59018552', '47777046', '56562119', '47021208', '49225225', '48085077', '54384163', '48791133', '53534547', '44289344', '40115530', '41471616', '49716693', '36267491', '48223162', '48910328', '36744555', '39981684', '47987212', '42144422', '38101834', '50090357', '48063821', '57427659', '43304425', '55601831', '34696282', '38820162', '44522505', '55431562', '62889525', '54223597', '53894805', '61724959', '43104527', '61071698', '50592353', '49837908', '43722074', '50229276', '43095070', '44486260', '57732427', '44551409', '60088810', '48589833', '53761947', '50307810', '58264744', '54718798', '60316058', '43587561', '34345827', '63435824', '38753139', '47288830', '59579870', '53855704', '51306350', '41352964', '36853403', '50359422', '37380546', '53749704', '61823969', '43536220', '36088396', '46686881', '49813762', '50161581', '38726143', '58538279', '56994817', '59390988', '41489484', '48537309', '57227547', '58428014', '55199043', '46219309', '57441807', '39133562', '35727708', '54372304', '35855219', '42562650', '54742449', '34232533', '48815730', '47583685', '47235448', '39646025', '42811536', '40032112', '36412802', '57560621', '48549654', '47321605', '59571286', '60777752', '48347663', '43576298', '56040520', '41600007', '51356461', '57043917', '47761718', '39783314', '43135048', '55413120', '58195903', '51928900', '40649185', '47889528', '55518654', '42708799', '41351418', '59239728', '57959532', '51732374', '51329871', '47809165', '56823828', '58533519', '57776691', '51586823', '47723110', '45308609', '61896836', '50337385', '47916567', '35110972', '45267542', '41943422', '48667570', '44479970', '43603466', '34272341', '36088778', '48932957', '51089993', '49896013', '45183635', '45347628', '44983945', '47526013', '59531820']        
        self.train_file = json.load(open(os.path.join(self.config['data_path'], self.config['train_data'])))
        print("Finish reading")
        self.data_train_word = np.zeros((len(self.train_file), self.config['max_length']), dtype = np.int64)
        self.data_train_pos = np.zeros((len(self.train_file), self.config['max_length']), dtype = np.int64)
        for i, ins in enumerate(self.train_file):
            self.data_train_word[i][0:len(ins["sent_idxs"])] = np.array(ins["sent_idxs"])
            self.data_train_pos[i][0:len(ins["sent_pos"])] = np.array(ins["sent_pos"])

        
        self.val_len = math.floor(len(self.train_file) * self.config['val_ratio'])

        shuffled_docs = list(range(len(self.train_file)))
        random.seed(self.config['seed'])
        random.shuffle(shuffled_docs)

        self.val_order = shuffled_docs[len(shuffled_docs)-self.val_len:]   
        
        num_docs_with_label = 0
        new_val_order = []
        
        
        for i in self.val_order:
            if self.train_file[i]['title'] in with_label_docs:
                num_docs_with_label += 1
                new_val_order.append(i)

        for i in self.val_order:
            if len(new_val_order) < num_docs_with_label*2 and self.train_file[i]['title'] not in with_label_docs:
                new_val_order.append(i)
        random.shuffle(new_val_order)
        self.val_order = new_val_order
        self.val_len = len(new_val_order)

        self.val_batches = self.val_len // self.config['batch_size']
        if self.val_len % self.config['batch_size'] != 0:
            self.val_batches += 1


    def load_test_data(self):
        print("Reading training data...")
        with_label_docs = ['55221698', '42539696', '53902375', '44809583', '45186929', '53197786', '34596212', '50068955', '48347687', '41483033', '48006315', '44654244', '52055189', '37603765', '50049491', '58574655', '55796211', '37624060', '62771771', '55318851', '35530400', '42965020', '63669319', '47143624', '59953445', '42774074', '50080463', '34243720', '46525453', '60409372', '50259637', '50259737', '48340802', '47984274', '51089448', '50048405', '45960834', '41628828', '53791390', '62979584', '48472420', '53633664', '64080452', '47249767', '45459557', '46040638', '40981831', '37377303', '42932979', '51898852', '59454480', '59018552', '47777046', '56562119', '47021208', '49225225', '48085077', '54384163', '48791133', '53534547', '44289344', '40115530', '41471616', '49716693', '36267491', '48223162', '48910328', '36744555', '39981684', '47987212', '42144422', '38101834', '50090357', '48063821', '57427659', '43304425', '55601831', '34696282', '38820162', '44522505', '55431562', '62889525', '54223597', '53894805', '61724959', '43104527', '61071698', '50592353', '49837908', '43722074', '50229276', '43095070', '44486260', '57732427', '44551409', '60088810', '48589833', '53761947', '50307810', '58264744', '54718798', '60316058', '43587561', '34345827', '63435824', '38753139', '47288830', '59579870', '53855704', '51306350', '41352964', '36853403', '50359422', '37380546', '53749704', '61823969', '43536220', '36088396', '46686881', '49813762', '50161581', '38726143', '58538279', '56994817', '59390988', '41489484', '48537309', '57227547', '58428014', '55199043', '46219309', '57441807', '39133562', '35727708', '54372304', '35855219', '42562650', '54742449', '34232533', '48815730', '47583685', '47235448', '39646025', '42811536', '40032112', '36412802', '57560621', '48549654', '47321605', '59571286', '60777752', '48347663', '43576298', '56040520', '41600007', '51356461', '57043917', '47761718', '39783314', '43135048', '55413120', '58195903', '51928900', '40649185', '47889528', '55518654', '42708799', '41351418', '59239728', '57959532', '51732374', '51329871', '47809165', '56823828', '58533519', '57776691', '51586823', '47723110', '45308609', '61896836', '50337385', '47916567', '35110972', '45267542', '41943422', '48667570', '44479970', '43603466', '34272341', '36088778', '48932957', '51089993', '49896013', '45183635', '45347628', '44983945', '47526013', '59531820']        
        self.train_file = json.load(open(os.path.join(self.config['data_path'], self.config['train_data'])))
        print("Finish reading")
        self.data_train_word = np.zeros((len(self.train_file), self.config['max_length']), dtype = np.int64)
        self.data_train_pos = np.zeros((len(self.train_file), self.config['max_length']), dtype = np.int64)
        for i, ins in enumerate(self.train_file):
            self.data_train_word[i][0:len(ins["sent_idxs"])] = np.array(ins["sent_idxs"])
            self.data_train_pos[i][0:len(ins["sent_pos"])] = np.array(ins["sent_pos"])

        # teacher forching
        if self.config['teacher_forcing']:
            self.val_order = [13, 16, 19, 22, 27, 47, 51, 56, 63, 69, 71, 83, 84, 88, 90, 92, 99, 104, 108, 116, 117, 124, 125, 126, 128, 132, 139, 141, 151, 153, 178, 188, 215, 260, 267, 349, 418, 481, 544, 581]
        else:
            self.val_order = [7, 13, 16, 19, 20, 22, 27, 41, 47, 51, 56, 63, 69, 71, 83, 84, 88, 90, 92, 99, 104, 108, 116, 117, 124, 125, 126, 128, 132, 139, 140, 141, 151, 153, 160, 164, 165, 166, 175, 178, 188, 193, 198, 204, 215, 233, 234, 235, 238, 239, 242, 247, 252, 253, 258, 260, 267, 275, 284, 293, 298, 309, 310, 316, 317, 320, 329, 330, 336, 349, 351, 354, 355, 357, 360, 361, 363, 367, 370, 375, 389, 391, 409, 410, 411, 415, 418, 424, 427, 434, 437, 439, 447, 454, 455, 456, 457, 461, 462, 479, 481, 483, 493, 497, 499, 523, 524, 533, 537, 539, 544, 545, 560, 562, 563, 565, 581]
        self.val_len = len(self.val_order)
        
        self.val_batches = self.val_len // self.config['batch_size']
        if self.val_len % self.config['batch_size'] != 0:
            self.val_batches += 1

    def get_train_batch(self):
        context_idxs = torch.LongTensor(self.config['batch_size'], self.config['max_length']).cuda()
        context_pos = torch.LongTensor(self.config['batch_size'], self.config['max_length']).cuda()
        h_mapping = torch.Tensor(self.config['batch_size'], self.config['h_t_limit'], self.config['max_length']).cuda()
        t_mapping = torch.Tensor(self.config['batch_size'], self.config['h_t_limit'], self.config['max_length']).cuda()
        relation_mask = torch.Tensor(self.config['batch_size'], self.config['h_t_limit']).cuda()
        relation_label = torch.LongTensor(self.config['batch_size'], self.config['h_t_limit']).cuda()
        evidence_label = torch.Tensor(self.config['batch_size'], self.config['h_t_limit'], self.config['sent_limit']).cuda()
        sent_mask = torch.Tensor(self.config['batch_size'], self.config['h_t_limit'], self.config['sent_limit']).cuda()
        sent_lengths = torch.LongTensor(self.config['batch_size'], self.config['h_t_limit']).cuda()
        sent_h_mapping = torch.Tensor(self.config['batch_size'], self.config['sent_limit'], self.config['max_length']).cuda()
        sent_t_mapping = torch.Tensor(self.config['batch_size'], self.config['sent_limit'], self.config['max_length']).cuda()

        total_relation = 0
        total_norelation = 0
        index_with_rels = []
        num_sentences = 0
        num_tokens = 0

        random.shuffle(self.train_order)

        for b in range(self.train_batches):
            start_id = b * self.config['batch_size']
            cur_bsz = min(self.config['batch_size'], self.train_len - start_id)
            cur_batch = list(self.train_order[start_id: start_id + cur_bsz])
            cur_batch.sort(key=lambda x: np.sum(self.data_train_word[x]>0) , reverse = True)

            for mapping in [h_mapping, t_mapping, relation_mask, sent_mask, evidence_label, sent_h_mapping, sent_t_mapping]:
                mapping.zero_()

            relation_label.fill_(self.config['IGNORE_INDEX'])

            max_h_t_cnt = 1

            raw_sentences = []
            evidences = []
            max_sents = 0

            for i, index in enumerate(cur_batch):
                cur_evidences = []
                context_idxs[i].copy_(torch.from_numpy(self.data_train_word[index, :]))
                context_pos[i].copy_(torch.from_numpy(self.data_train_pos[index, :]))

                ins = self.train_file[index]
                raw_sentences.append(ins["sents"])
                max_sents = max(max_sents, len(ins['sent_ends'])-1)

                labels = ins['labels']
                idx2label = defaultdict(list)
                for label in labels:
                    idx2label[(label['h'], label['t'])] = label['r']

                for e in range(len(ins['sent_ends']) - 1):
                    sent_h_mapping[i, e, ins['sent_ends'][e]] = 1
                    sent_t_mapping[i, e, ins['sent_ends'][e+1]-1] = 1

                j = 0
                for label in labels:
                    h_idx = label["h"]
                    t_idx = label["t"]
                    hlist = ins['vertexSet'][h_idx]
                    tlist = ins['vertexSet'][t_idx]

                    sent_lengths[i, j] = len(ins['sent_ends']) - 1

                    for h in hlist:
                        h_mapping[i, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(hlist) / (h['pos'][1] - h['pos'][0])
                    for t in tlist:
                        t_mapping[i, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(tlist) / (t['pos'][1] - t['pos'][0])
                    for e in label['evidence']:
                        evidence_label[i, j, int(e)] = 1
                    

                    evidence = [int(e) for e in label['evidence']]
                    cur_evidences.append(evidence)

                    num_sentences += len(ins['sent_ends'])-1
                    for k in range(len(ins['sent_ends'])-1):
                        sent_mask[i, j, k] = 1

                    relation_label[i, j] = 1
                    relation_mask[i, j] = 1
                    j += 1

                    total_relation += 1
                    if index not in index_with_rels:
                        index_with_rels.append(index)

                if j > 0:
                    lower_bound = j*self.config['reg_matching_ratio']
                else:
                    lower_bound = self.config["no_rel_ratio"]
                random.shuffle(ins['na_triple'])

                for j, (h_idx, t_idx) in enumerate(ins['na_triple'][:lower_bound], j):
                    hlist = ins['vertexSet'][h_idx]
                    tlist = ins['vertexSet'][t_idx]

                    for h in hlist:
                        h_mapping[i, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(hlist) / (h['pos'][1] - h['pos'][0])

                    for t in tlist:
                        t_mapping[i, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(tlist) / (t['pos'][1] - t['pos'][0])

                    cur_evidences.append([])

                    for k in range(len(ins['sent_ends'])-1):
                        sent_mask[i, j, k] = 0

                    relation_mask[i, j] = 1
                    relation_label[i, j] = 0
                    total_norelation += 1
                max_h_t_cnt = max(max_h_t_cnt, len(list(idx2label.keys())) + lower_bound)
                evidences.append(cur_evidences)
            
            input_lengths = (context_idxs[:cur_bsz] > 0).long().sum(dim=1)
            num_tokens += sum(input_lengths)

            max_c_len = int(input_lengths.max())
            yield {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'context_pos': context_pos[:cur_bsz, :max_c_len].contiguous(),
                   'h_mapping': h_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   't_mapping': t_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   'relation_label': relation_label[:cur_bsz, :max_h_t_cnt].contiguous(),
                   'input_lengths': input_lengths,
                   'relation_mask': relation_mask[:cur_bsz, :max_h_t_cnt],
                   'sent_lengths': sent_lengths[:cur_bsz, :max_h_t_cnt],
                   'sent_mask': sent_mask[:cur_bsz, :max_h_t_cnt, :max_sents],
                   'evidence_label': evidence_label[:cur_bsz, :max_h_t_cnt, :max_sents],
                   'evidences': evidences[:cur_bsz],
                   'raw_sentences': raw_sentences,
                   'sent_h_mapping': sent_h_mapping[:cur_bsz, :max_sents, :max_c_len],
                   'sent_t_mapping': sent_t_mapping[:cur_bsz, :max_sents, :max_c_len],
                   }
        print("Read {} documents, {} with relation".format(self.train_len, len(index_with_rels)))
        print("{} sentences, {} tokens".format(num_sentences, num_tokens))
        print("Total relation: {}, total no-relation: {}".format(total_relation, total_norelation))


    def get_validation_batch(self):
        context_idxs = torch.LongTensor(self.config['batch_size'], self.config['max_length']).cuda()
        context_pos = torch.LongTensor(self.config['batch_size'], self.config['max_length']).cuda()
        h_mapping = torch.Tensor(self.config['batch_size'], self.config['h_t_limit'], self.config['max_length']).cuda()
        t_mapping = torch.Tensor(self.config['batch_size'], self.config['h_t_limit'], self.config['max_length']).cuda()
        relation_mask = torch.Tensor(self.config['batch_size'], self.config['h_t_limit']).cuda()
        relation_label = torch.LongTensor(self.config['batch_size'], self.config['h_t_limit']).cuda()
        evidence_label = torch.Tensor(self.config['batch_size'], self.config['h_t_limit'], self.config['sent_limit']).cuda()
        sent_mask = torch.Tensor(self.config['batch_size'], self.config['h_t_limit'], self.config['sent_limit']).cuda()
        sent_lengths = torch.LongTensor(self.config['batch_size'], self.config['h_t_limit']).cuda()

        sent_h_mapping = torch.Tensor(self.config['batch_size'], self.config['sent_limit'], self.config['max_length']).cuda()
        sent_t_mapping = torch.Tensor(self.config['batch_size'], self.config['sent_limit'], self.config['max_length']).cuda()

        total_relation = 0
        total_norelation = 0
        index_with_rels = []
        num_sentences = 0
        num_tokens = 0
        idx2term = {}


        for b in range(self.val_batches):
            start_id = b * self.config['batch_size']
            cur_bsz = min(self.config['batch_size'], self.val_len - start_id)
            cur_batch = list(self.val_order[start_id: start_id + cur_bsz])
            cur_batch.sort(key=lambda x: np.sum(self.data_train_word[x]>0) , reverse = True)

            for mapping in [h_mapping, t_mapping, relation_mask, sent_mask, evidence_label, sent_h_mapping, sent_t_mapping]:
                mapping.zero_()

            relation_label.fill_(self.config['IGNORE_INDEX'])

            labels = []
            num_vertices = []
            titles = []
            indexes = []
            raw_sentences = []
            evidences = []
            max_sents = 0
            max_h_t_cnt = 1
            idx2labels = []


            for i, index in enumerate(cur_batch):
                cur_evidences = []
                context_idxs[i].copy_(torch.from_numpy(self.data_train_word[index, :]))
                context_pos[i].copy_(torch.from_numpy(self.data_train_pos[index, :]))

                idx2label = defaultdict(list)
                idx2evi = defaultdict(list)
                ins = self.train_file[index]
                raw_sentences.append(ins["sents"])
                max_sents = max(max_sents, len(ins['sent_ends'])-1)

                cur_labels = ins['labels']
                
                for label in cur_labels:
                    idx2label[(label['h'], label['t'])] = label['r']
                    idx2evi[(label['h'], label['t'])] = label['evidence']

                num_vertex = len(ins['vertexSet'])
                titles.append(ins['title'])
                
                for e in range(len(ins['sent_ends']) - 1):
                    sent_h_mapping[i, e, ins['sent_ends'][e]] = 1
                    sent_t_mapping[i, e, ins['sent_ends'][e+1]-1] = 1

                j = 0
                for label in cur_labels:
                    h_idx = label["h"]
                    t_idx = label["t"]
                    hlist = ins['vertexSet'][h_idx]
                    tlist = ins['vertexSet'][t_idx]
                    idx2term[(ins['title'], h_idx)] = ins['vertexSet'][h_idx][0]["name"].lower()
                    idx2term[(ins['title'], t_idx)] = ins['vertexSet'][t_idx][0]["name"].lower()
                    sent_lengths[i, j] = len(ins['sent_ends']) - 1

                    # i: doc index, j: h-t relation index
                    for h in hlist:
                        h_mapping[i, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(hlist) / (h['pos'][1] - h['pos'][0])
                    for t in tlist:
                        t_mapping[i, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(tlist) / (t['pos'][1] - t['pos'][0])
                    for e in label['evidence']:
                        evidence_label[i, j, int(e)] = 1

                    evidence = [int(e) for e in label['evidence']]
                    cur_evidences.append(evidence)

                    num_sentences += len(ins['sent_ends'])-1
                    for k in range(len(ins['sent_ends'])-1):
                        sent_mask[i, j, k] = 1

                    relation_label[i, j] = 1
                    relation_mask[i, j] = 1
                    j += 1

                    total_relation += 1
                    if index not in index_with_rels:
                        index_with_rels.append(index)

                if j > 0:
                    lower_bound = max(self.config["no_rel_ratio"], j)
                else:
                    lower_bound = self.config["no_rel_ratio"]
                random.shuffle(ins['na_triple'])

                for j, (h_idx, t_idx) in enumerate(ins['na_triple'][:lower_bound], j):
                    hlist = ins['vertexSet'][h_idx]
                    tlist = ins['vertexSet'][t_idx]
                    idx2term[(ins['title'], h_idx)] = ins['vertexSet'][h_idx][0]["name"].lower()
                    idx2term[(ins['title'], t_idx)] = ins['vertexSet'][t_idx][0]["name"].lower()

                    for h in hlist:
                        h_mapping[i, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(hlist) / (h['pos'][1] - h['pos'][0])

                    for t in tlist:
                        t_mapping[i, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(tlist) / (t['pos'][1] - t['pos'][0])

                    cur_evidences.append([])

                    for k in range(len(ins['sent_ends'])-1):
                        sent_mask[i, j, k] = 0

                    idx2label[(h_idx, t_idx)] = 0

                    relation_label[i, j] = 0
                    relation_mask[i, j] = 1

                    total_norelation += 1

                max_h_t_cnt = max(max_h_t_cnt, len(list(idx2label.keys())) + lower_bound)
                labels.append(cur_labels)
                num_vertices.append(num_vertex)
                indexes.append(index)
                evidences.append(cur_evidences)
                idx2labels.append(idx2label)
            
            input_lengths = (context_idxs[:cur_bsz] > 0).long().sum(dim=1)
            num_tokens += sum(input_lengths)

            max_c_len = int(input_lengths.max())
            yield {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'context_pos': context_pos[:cur_bsz, :max_c_len].contiguous(),
                   'h_mapping': h_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   't_mapping': t_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   'relation_label': relation_label[:cur_bsz, :max_h_t_cnt].contiguous(),
                   'labels': labels,
                   'input_lengths': input_lengths,
                   'num_vertices': num_vertices,
                   'relation_mask': relation_mask[:cur_bsz, :max_h_t_cnt],
                   'sent_lengths': sent_lengths[:cur_bsz, :max_h_t_cnt],
                   'titles': titles,
                   'indexes': indexes,
                   'idx2term': idx2term,
                   'sent_mask': sent_mask[:cur_bsz, :max_h_t_cnt, :max_sents],
                   'evidence_label': evidence_label[:cur_bsz, :max_h_t_cnt, :max_sents],
                   'evidences': evidences[:cur_bsz],
                   'raw_sentences': raw_sentences,
                   'idx2labels': idx2labels,
                   'sent_h_mapping': sent_h_mapping[:cur_bsz, :max_sents, :max_c_len],
                   'sent_t_mapping': sent_t_mapping[:cur_bsz, :max_sents, :max_c_len],
                   }
                   
        print("Read {} documents, {} with relation".format(self.val_len, len(index_with_rels)))
        print("{} sentences, {} tokens".format(num_sentences, num_tokens))
        print("Total relation: {}, total no-relation: {}".format(total_relation, total_norelation))


    def train(self, model, model_name, sacred_ex, experiment_id):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.config['lr'])
        if self.config['lr_warm_up']:
            scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=self.config['num_warmup']*(self.train_batches-1), num_training_steps=self.config['max_epoch'] * (self.train_batches-1))
        
        NLL = nn.NLLLoss(ignore_index=self.config['IGNORE_INDEX'], reduction='none')
        BCE = nn.BCEWithLogitsLoss(reduction='none')

        best_loss = 1000
        best_re_f1 = 0.0
        best_sep_f1 = 0.0
        best_epoch = 0
        
        model.train()

        def logging(s, print_=True, log_=True):
            if print_:
                print(s)
            if log_:
                path = './logs/{}/{}'.format(experiment_id, 'logs.txt')
                with open(path, 'a+') as f_log:
                    f_log.write(s + '\n')

        losses = AverageMeter()
        re_losses = AverageMeter()
        sep_losses = AverageMeter()

        for epoch in range(self.config['max_epoch']):
            self.epoch = epoch
            print('-' * 89)
            print("Training Epoch {}".format(epoch))
            if not self.config['nonstop_training']:
                if self.decreasing_val_metrics > self.config['decreasing_limit'] / self.config['test_epoch']:
                    break

            self.train_relation_score._reset()
            self.train_evidence_score._reset()

            for batch_id, data in enumerate(self.get_train_batch()):
                context_idxs = data['context_idxs']
                if context_idxs.shape[0] < self.config['batch_size']:
                    continue
                context_idxs = torch.where(torch.logical_or(context_idxs == 0, torch.rand(context_idxs.shape).cuda() > self.config['input_dropout']), context_idxs, self.config['mask_id'])
                context_pos = data['context_pos']
                h_mapping = data['h_mapping']
                t_mapping = data['t_mapping']
                relation_label = data['relation_label']
                relation_mask = data['relation_mask']
                evidences = data['evidences']
                evidence_label = data['evidence_label']
                sent_mask = data['sent_mask']
                sent_h_mapping = data['sent_h_mapping']
                sent_t_mapping = data['sent_t_mapping']
                input_lengths = data['input_lengths']
                sent_lengths = data['sent_lengths']

                predict_re, predict_sent  = model(context_idxs, h_mapping, t_mapping, sent_h_mapping, sent_t_mapping, context_pos, input_lengths)
                loss_re = torch.sum(NLL(predict_re.view(-1,1,predict_re.shape[2]).squeeze(), relation_label.view(-1,1).squeeze())*relation_mask.reshape(-1,1).squeeze()) / torch.sum(relation_mask)                
                
                loss_sent = torch.sum(BCE(predict_sent, evidence_label) * sent_mask) / torch.sum(sent_mask)
                loss = self.config['loss_re_coeff'] * loss_re + loss_sent


                losses.update(loss.item(), torch.sum(relation_mask))
                re_losses.update(loss_re.item(), torch.sum(relation_mask))
                sep_losses.update(loss_sent.item(), torch.sum(sent_mask))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if self.config['lr_warm_up']:
                    scheduler.step()

                predict_re = torch.exp(predict_re)
                predict_sent = torch.sigmoid(predict_sent) * sent_mask

                self.train_relation_score.update(predict_re, relation_label, 0)
                self.train_evidence_score.update(predict_sent, evidence_label, sent_mask)
                predict_re = predict_re.data.cpu().numpy()
                relation_label = relation_label.data.cpu().numpy()

            
            logging('-' * 89)
            logging('Training RE Loss: {:3.4f}'.format(re_losses.avg))
            logging('Training SEP Loss: {:3.4f}'.format(sep_losses.avg))
            logging("")

            re_result_string, re_accuracy, re_f1, re_precision, re_recall = self.train_relation_score.result()
            sep_result_string, sep_accuracy, sep_f1, sep_precision, sep_recall = self.train_evidence_score.result(self.config['val_sentence_theta'])
            logging(re_result_string)
            logging(sep_result_string)

            if self.config['fold_val_idx'] == 5:
                continue

            if (epoch+1) % self.config['validation_period'] == 0:
                logging('-' * 89)
                eval_start_time = time.time()
                model.eval()
                val_re_f1, val_sep_f1, val_loss =  self.test(model, experiment_id)
                model.train()
                logging('| epoch {:3d} | time: {:5.2f}s'.format(epoch, time.time() - eval_start_time))
                logging('-' * 89)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_re_f1 = val_re_f1
                    best_sep_f1 = val_sep_f1
                    best_epoch = epoch
                    path = './logs/{}/{}'.format(experiment_id, model_name)
                    torch.save(model.state_dict(), path)
                    result_path = './logs/{}/{}'.format(experiment_id, self.config['result_filename'])
                    best_result_path = './logs/{}/best_result.txt'.format(experiment_id)
                    shutil.copy2(result_path, best_result_path)
                    # sacred_ex.add_artifact(best_result_path)
                    logging('Best Epoch: {0}, Total Loss: {1:3.4f}'.format(best_epoch, best_loss))

        path = './logs/{}/{}_final_epoch'.format(experiment_id, model_name)
        torch.save(model.state_dict(), path)
        
        logging("Finish training")
        logging("Best validation epoch = {:3f} | Best validation RE F1 = {:3.4f} | Best validation SEP F1 = {:3.4f}".format(best_epoch, best_re_f1, best_sep_f1))


    def test(self, model, experiment_id):
        test_result = []

        losses = AverageMeter()
        re_losses = AverageMeter()
        sep_losses = AverageMeter()

        NLL = nn.NLLLoss(ignore_index=self.config['IGNORE_INDEX'], reduction='none')
        BCE = nn.BCEWithLogitsLoss(reduction='none')

        def logging(s, print_=True, log_=True):
            if print_:
                print(s)
            if log_:
                path = './logs/{}/{}'.format(experiment_id, 'logs.txt')
                with open(path, 'a+') as f_log:
                    f_log.write(s + '\n')

        self.val_relation_score._reset()
        self.val_evidence_score._reset()

        for batch_id, data in enumerate(self.get_validation_batch()):        
            with torch.no_grad():
                context_idxs = data['context_idxs']
                context_pos = data['context_pos']
                h_mapping = data['h_mapping']
                t_mapping = data['t_mapping']
                labels = data['labels']
                num_vertices = data['num_vertices']
                relation_mask = data['relation_mask']
                idx2term = data['idx2term']
                relation_label = data['relation_label']
                batch_size = context_idxs.shape[0]
                titles = data['titles']
                indexes = data['indexes']
                evidence_label = data['evidence_label']
                sent_mask = data['sent_mask']
                evidences = data['evidences']
                raw_sentences = data['raw_sentences']
                idx2labels = data['idx2labels']
                sent_h_mapping = data['sent_h_mapping']
                sent_t_mapping = data['sent_t_mapping']
                input_lengths = data['input_lengths']
                sent_lengths = data['sent_lengths']

                predict_re, predict_sent = model(context_idxs, h_mapping, t_mapping, sent_h_mapping, sent_t_mapping, context_pos, input_lengths)
                loss_re = torch.sum(NLL(predict_re.view(-1,1,predict_re.shape[2]).squeeze(), relation_label.view(-1,1).squeeze())*relation_mask.reshape(-1,1).squeeze()) / torch.sum(relation_mask)
                evi_re_label = relation_label.unsqueeze(-1).expand(-1, -1, evidence_label.size(2))
                loss_sent = torch.sum(BCE(predict_sent, evidence_label) * sent_mask) / torch.sum(sent_mask)
                loss = self.config['loss_re_coeff'] * loss_re + loss_sent


                predict_re = torch.exp(predict_re)
                predict_sent = torch.sigmoid(predict_sent) * sent_mask

                if self.config['log_val_example']:
                    logging('-' * 89)
                    logging(titles[0])
                    logging(raw_sentences[0])
                    logging(relation_label[0])
                    logging(evidences[0])
                    logging(predict_re[0][0])
                    logging(predict_sent[0][0])

                self.val_relation_score.update(predict_re, relation_label, self.config['val_theta'])
                self.val_evidence_score.update(predict_sent, evidence_label, sent_mask)
                
                losses.update(loss.item(), torch.sum(relation_mask))
                re_losses.update(loss_re.item(), torch.sum(relation_mask))
                sep_losses.update(loss_sent.item(), torch.sum(sent_mask))

                context_idxs = context_idxs.data.cpu()

            predict_re = predict_re.data.cpu().numpy()
            relation_label = relation_label.data.cpu().numpy()
            
            for i in range(len(labels)):
                cur_labels = labels[i]
                index = indexes[i]
                evi = evidences[i]

                l = num_vertices[i]
                j = 0


                if not self.config['use_extensive_combination_for_val']:
                    # for j, label in enumerate(cur_labels):
                    for j, label in enumerate(list(idx2labels[i].keys())):
                        h_idx = label[0]
                        t_idx = label[1]
                        r = int(np.argmax(predict_re[i, j]))
                        if r != 0:
                            test_result.append( \
                                # ((h_idx, t_idx, 'S1') in cur_labels, \
                                (idx2labels[i][(h_idx, t_idx)] == 'S1', \
                                titles[i], \
                                "S1", \
                                index, \
                                h_idx, \
                                t_idx, \
                                r, \
                                [float(predict_re[i,j,k]) for k in range(self.config['so_relation_num'])], \
                                [raw_sentences[i][s].encode('utf-8') for s in evi[j]], \
                                evi[j], \
                                [float(s) for s in predict_sent[i, j][:int(torch.sum(sent_mask[i, j]).item())]] \
                                ) )
                        else:
                            test_result.append( \
                                # (h_idx, t_idx, 'S1') not in cur_labels, \
                                (idx2labels[i][(h_idx, t_idx)] == 0, \
                                titles[i], \
                                "None", \
                                index, \
                                h_idx, \
                                t_idx, \
                                r, \
                                [float(predict_re[i,j,k]) for k in range(self.config['so_relation_num'])], \
                                [raw_sentences[i][s].encode('utf-8') for s in evi[j]], \
                                evi[j], \
                                [float(s) for s in predict_sent[i, j][:int(torch.sum(sent_mask[i, j]).item())]] \
                                ))
                else:
                    for h_idx in range(l):
                        for t_idx in range(l):
                            if h_idx != t_idx:
                                r = int(np.argmax(predict_re[i, j]))
                                if r != 0:
                                    test_result.append( \
                                        (idx2labels[i][(h_idx, t_idx)] == 'S1', \
                                        titles[i], \
                                        "S1", \
                                        index, \
                                        h_idx, \
                                        t_idx, \
                                        r, \
                                        [float(predict_re[i,j,k]) for k in range(self.config['so_relation_num'])], \
                                        [raw_sentences[i][s].encode('utf-8') for s in evi[j]], \
                                        evi[j], \
                                        [float(s) for s in predict_sent[i, j][:int(torch.sum(sent_mask[i, j]).item())]] \
                                        ) )
                                else:
                                    test_result.append( \
                                        (idx2labels[i][(h_idx, t_idx)] == 0, \
                                        titles[i], \
                                        "None", \
                                        index, \
                                        h_idx, \
                                        t_idx, \
                                        r, \
                                        [float(predict_re[i,j,k]) for k in range(self.config['so_relation_num'])], \
                                        [raw_sentences[i][s].encode('utf-8') for s in evi[j]], \
                                        evi[j], \
                                        [float(s) for s in predict_sent[i, j][:int(torch.sum(sent_mask[i, j]).item())]] \
                                        ))
                                j += 1

        test_result.sort(key = lambda x: x[7][x[6]]* max(x[9]) if len(x[9]) > 0 else x[7][x[6]], reverse=True)
        logging('-' * 89)

        if len(test_result) < 2:
            return [0], 0, [0], [0], 0, 0, 0

        vw = 0

        for i, item in enumerate(test_result):
            if item[7][item[6]] > self.config['val_theta']:
                vw = i

        logging('Test Epoch {}'.format(self.epoch))
        logging('Test RE Loss: {:3.4f}'.format(re_losses.avg))
        logging('Test SEP Loss: {:3.4f}'.format(sep_losses.avg))
        logging("")
        

        re_result_string, re_accuracy, re_f1, re_precision, re_recall = self.val_relation_score.result()
        sep_result_string, sep_accuracy, sep_f1, sep_precision, sep_recall = self.val_evidence_score.result(self.config['val_sentence_theta'])

        logging(re_result_string)
        logging(sep_result_string)

        if not self.config['nonstop_training']:
            cur_val_criteria = losses.avg
            if self.best_val_loss < cur_val_criteria:
                self.best_val_loss = cur_val_criteria
            elif self.best_val_loss >= cur_val_criteria:
                self.decreasing_val_metrics += 1
                print("best: {0:.3f}, cur: {0:.3f}".format(self.best_val_loss, cur_val_criteria))

        path = './logs/{}/{}'.format(experiment_id, self.config['result_filename'])
        with open(path, 'w') as f:
            for x in test_result[:vw+1]:
                f.write("{}\n{}\t{}\n{}\t{}\n{}\t{}\n".format("https://stackoverflow.com/questions/"+x[1], idx2term[(x[1],x[4])], idx2term[(x[1],x[5])], x[6], x[0], x[7][0], x[7][1]))
                f.write("{}\n{}\n{}\n\n".format(x[8], x[9], x[10]))

            f.write("="*80)

            for x in test_result[vw+1:]:
                if x[6] == 0 and x[1] == False:
                    f.write("{}\n{}\t{}\n{}\t{}\n{}\t{}\n".format("https://stackoverflow.com/questions/"+x[1], idx2term[(x[1],x[4])], idx2term[(x[1],x[5])], x[6], x[0], x[7][0], x[7][1]))
                    f.write("{}\n{}\n{}\n\n".format(x[8], x[9], x[10]))

        return re_f1, sep_f1, losses.avg


