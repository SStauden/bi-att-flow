import json
import os
import random
import itertools
import math
from collections import defaultdict

import numpy as np

from my.tensorflow import grouper
from my.utils import index

import sys
from pathlib import Path
import tensorflow as tf


class Data(object):
    def get_size(self):
        raise NotImplementedError()

    def get_by_idxs(self, idxs):
        """
        Efficient way to obtain a batch of items from filesystem
        :param idxs:
        :return dict: {'X': [,], 'Y', }
        """
        data = defaultdict(list)
        for idx in idxs:
            each_data = self.get_one(idx)
            for key, val in each_data.items():
                data[key].append(val)
        return data

    def get_one(self, idx):
        raise NotImplementedError()

    def get_empty(self):
        raise NotImplementedError()

    def __add__(self, other):
        raise NotImplementedError()


class DataSet(object):
    def __init__(self, data, data_type, shared=None, valid_idxs=None):
        self.data = data  # e.g. {'X': [0, 1, 2], 'Y': [2, 3, 4]}
        self.data_type = data_type
        self.shared = shared
        total_num_examples = self.get_data_size()
        self.valid_idxs = range(total_num_examples) if valid_idxs is None else valid_idxs
        self.num_examples = len(self.valid_idxs)

    def _sort_key(self, idx):
        rx = self.data['*x'][idx]
        x = self.shared['x'][rx[0]][rx[1]]
        return max(map(len, x))

    def get_data_size(self):
        if isinstance(self.data, dict):
            return len(next(iter(self.data.values())))
        elif isinstance(self.data, Data):
            return self.data.get_size()
        raise Exception()

    def get_by_idxs(self, idxs):
        if isinstance(self.data, dict):
            out = defaultdict(list)
            for key, val in self.data.items():
                out[key].extend(val[idx] for idx in idxs)
            return out
        elif isinstance(self.data, Data):
            return self.data.get_by_idxs(idxs)
        raise Exception()

    def get_batches(self, batch_size, num_batches=None, shuffle=False, cluster=False):
        """

        :param batch_size:
        :param num_batches:
        :param shuffle:
        :param cluster: cluster examples by their lengths; this might give performance boost (i.e. faster training).
        :return:
        """
        num_batches_per_epoch = int(math.ceil(self.num_examples / batch_size))
        if num_batches is None:
            num_batches = num_batches_per_epoch
        num_epochs = int(math.ceil(num_batches / num_batches_per_epoch))

        if shuffle:
            random_idxs = random.sample(self.valid_idxs, len(self.valid_idxs))
            if cluster:
                sorted_idxs = sorted(random_idxs, key=self._sort_key)
                sorted_grouped = lambda: list(grouper(sorted_idxs, batch_size))
                grouped = lambda: random.sample(sorted_grouped(), num_batches_per_epoch)
            else:
                random_grouped = lambda: list(grouper(random_idxs, batch_size))
                grouped = random_grouped
        else:
            raw_grouped = lambda: list(grouper(self.valid_idxs, batch_size))
            grouped = raw_grouped

        batch_idx_tuples = itertools.chain.from_iterable(grouped() for _ in range(num_epochs))
        for _ in range(num_batches):
            batch_idxs = tuple(i for i in next(batch_idx_tuples) if i is not None)
            batch_data = self.get_by_idxs(batch_idxs)
            shared_batch_data = {}
            for key, val in batch_data.items():
                if key.startswith('*'):
                    assert self.shared is not None
                    shared_key = key[1:]
                    shared_batch_data[shared_key] = [index(self.shared[shared_key], each) for each in val]
            batch_data.update(shared_batch_data)

            batch_ds = DataSet(batch_data, self.data_type, shared=self.shared)
            yield batch_idxs, batch_ds

    def get_multi_batches(self, batch_size, num_batches_per_step, num_steps=None, shuffle=False, cluster=False):
        batch_size_per_step = batch_size * num_batches_per_step
        batches = self.get_batches(batch_size_per_step, num_batches=num_steps, shuffle=shuffle, cluster=cluster)
        multi_batches = (tuple(zip(grouper(idxs, batch_size, shorten=True, num_groups=num_batches_per_step),
                         data_set.divide(num_batches_per_step))) for idxs, data_set in batches)
        return multi_batches

    def get_empty(self):
        if isinstance(self.data, dict):
            data = {key: [] for key in self.data}
        elif isinstance(self.data, Data):
            data = self.data.get_empty()
        else:
            raise Exception()
        return DataSet(data, self.data_type, shared=self.shared)

    def __add__(self, other):
        if isinstance(self.data, dict):
            data = {key: val + other.data[key] for key, val in self.data.items()}
        elif isinstance(self.data, Data):
            data = self.data + other.data
        else:
            raise Exception()

        valid_idxs = list(self.valid_idxs) + [valid_idx + self.num_examples for valid_idx in other.valid_idxs]
        return DataSet(data, self.data_type, shared=self.shared, valid_idxs=valid_idxs)

    def divide(self, integer):
        batch_size = int(math.ceil(self.num_examples / integer))
        idxs_gen = grouper(self.valid_idxs, batch_size, shorten=True, num_groups=integer)
        data_gen = (self.get_by_idxs(idxs) for idxs in idxs_gen)
        ds_tuple = tuple(DataSet(data, self.data_type, shared=self.shared) for data in data_gen)
        return ds_tuple


def load_metadata(config, data_type):
    metadata_path = os.path.join(config.data_dir, "metadata_{}.json".format(data_type))
    with open(metadata_path, 'r') as fh:
        metadata = json.load(fh)
        for key, val in metadata.items():
            config.__setattr__(key, val)
        return metadata

def read_data_extended(config, data_type, ref, data_filter=None, data_path="", shared_path=""):
    data_file_path = os.path.join(data_path, "data_{}.json".format(data_type))
    shared_file_path = os.path.join(shared_path, "shared_{}.json".format(data_type))
    with open(data_file_path, 'r') as fh:
        data = json.load(fh)
    with open(shared_file_path, 'r') as fh:
        shared = json.load(fh)

    num_examples = len(next(iter(data.values())))
    if data_filter is None:
        valid_idxs = range(num_examples)
    else:
        mask = []
        keys = data.keys()
        values = data.values()
        for vals in zip(*values):
            each = {key: val for key, val in zip(keys, vals)}
            mask.append(data_filter(each, shared))
        valid_idxs = [idx for idx in range(len(mask)) if mask[idx]]

    print("Loaded {}/{} examples from {}".format(len(valid_idxs), num_examples, data_type))
    
    shared_path = config.shared_path or os.path.join(config.out_dir, "shared.json")
    print("Shared Path is {}".format(shared_path))
    # shared_path = shared_file_path
    if not ref:
        word2vec_dict = shared['lower_word2vec'] if config.lower_word else shared['word2vec']
        word_counter = shared['lower_word_counter'] if config.lower_word else shared['word_counter']
        char_counter = shared['char_counter']
        if config.finetune:
            shared['word2idx'] = {word: idx + 2 for idx, word in
                                  enumerate(word for word, count in word_counter.items()
                                            if count > config.word_count_th or (config.known_if_glove and word in word2vec_dict))}
        else:
            assert config.known_if_glove
            assert config.use_glove_for_unk
            shared['word2idx'] = {word: idx + 2 for idx, word in
                                  enumerate(word for word, count in word_counter.items()
                                            if count > config.word_count_th and word not in word2vec_dict)}
        shared['char2idx'] = {char: idx + 2 for idx, char in
                              enumerate(char for char, count in char_counter.items()
                                        if count > config.char_count_th)}
        NULL = "-NULL-"
        UNK = "-UNK-"
        shared['word2idx'][NULL] = 0
        shared['word2idx'][UNK] = 1
        shared['char2idx'][NULL] = 0
        shared['char2idx'][UNK] = 1
        json.dump({'word2idx': shared['word2idx'], 'char2idx': shared['char2idx']}, open(shared_path, 'w'))
    else:
        new_shared = json.load(open(shared_path, 'r'))
        for key, val in new_shared.items():
            shared[key] = val

    if config.use_glove_for_unk:
        # create new word2idx and word2vec
        word2vec_dict = shared['lower_word2vec'] if config.lower_word else shared['word2vec']
        new_word2idx_dict = {word: idx for idx, word in enumerate(word for word in word2vec_dict.keys() if word not in shared['word2idx'])}
        shared['new_word2idx'] = new_word2idx_dict
        offset = len(shared['word2idx'])
        word2vec_dict = shared['lower_word2vec'] if config.lower_word else shared['word2vec']
        new_word2idx_dict = shared['new_word2idx']
        idx2vec_dict = {idx: word2vec_dict[word] for word, idx in new_word2idx_dict.items()}
        # print("{}/{} unique words have corresponding glove vectors.".format(len(idx2vec_dict), len(word2idx_dict)))
        new_emb_mat = np.array([idx2vec_dict[idx] for idx in range(len(idx2vec_dict))], dtype='float32')
        shared['new_emb_mat'] = new_emb_mat

    data_set = DataSet(data, data_type, shared=shared, valid_idxs=valid_idxs)
    return data_set


def read_data(config, data_type, ref, data_filter=None):
    data_path = os.path.join(config.data_dir, "data_{}.json".format(data_type))
    shared_path = os.path.join(config.data_dir, "shared_{}.json".format(data_type))
    with open(data_path, 'r') as fh:
        data = json.load(fh)
    with open(shared_path, 'r') as fh:
        shared = json.load(fh)

    num_examples = len(next(iter(data.values())))
    if data_filter is None:
        valid_idxs = range(num_examples)
    else:
        mask = []
        keys = data.keys()
        values = data.values()
        for vals in zip(*values):
            each = {key: val for key, val in zip(keys, vals)}
            mask.append(data_filter(each, shared))
        valid_idxs = [idx for idx in range(len(mask)) if mask[idx]]

    print("Loaded {}/{} examples from {}".format(len(valid_idxs), num_examples, data_type))

    shared_path = config.shared_path or os.path.join(config.out_dir, "shared.json")
    if not ref:
        word2vec_dict = shared['lower_word2vec'] if config.lower_word else shared['word2vec']
        word_counter = shared['lower_word_counter'] if config.lower_word else shared['word_counter']
        char_counter = shared['char_counter']
        if config.finetune:
            shared['word2idx'] = {word: idx + 2 for idx, word in
                                  enumerate(word for word, count in word_counter.items()
                                            if count > config.word_count_th or (config.known_if_glove and word in word2vec_dict))}
        else:
            assert config.known_if_glove
            assert config.use_glove_for_unk
            shared['word2idx'] = {word: idx + 2 for idx, word in
                                  enumerate(word for word, count in word_counter.items()
                                            if count > config.word_count_th and word not in word2vec_dict)}
        shared['char2idx'] = {char: idx + 2 for idx, char in
                              enumerate(char for char, count in char_counter.items()
                                        if count > config.char_count_th)}
        NULL = "-NULL-"
        UNK = "-UNK-"
        shared['word2idx'][NULL] = 0
        shared['word2idx'][UNK] = 1
        shared['char2idx'][NULL] = 0
        shared['char2idx'][UNK] = 1
        json.dump({'word2idx': shared['word2idx'], 'char2idx': shared['char2idx']}, open(shared_path, 'w'))
    else:
        new_shared = json.load(open(shared_path, 'r'))
        for key, val in new_shared.items():
            shared[key] = val

    if config.use_glove_for_unk:
        # create new word2idx and word2vec
        word2vec_dict = shared['lower_word2vec'] if config.lower_word else shared['word2vec']
        new_word2idx_dict = {word: idx for idx, word in enumerate(word for word in word2vec_dict.keys() if word not in shared['word2idx'])}
        shared['new_word2idx'] = new_word2idx_dict
        offset = len(shared['word2idx'])
        word2vec_dict = shared['lower_word2vec'] if config.lower_word else shared['word2vec']
        new_word2idx_dict = shared['new_word2idx']
        idx2vec_dict = {idx: word2vec_dict[word] for word, idx in new_word2idx_dict.items()}
        # print("{}/{} unique words have corresponding glove vectors.".format(len(idx2vec_dict), len(word2idx_dict)))
        new_emb_mat = np.array([idx2vec_dict[idx] for idx in range(len(idx2vec_dict))], dtype='float32')
        shared['new_emb_mat'] = new_emb_mat

    data_set = DataSet(data, data_type, shared=shared, valid_idxs=valid_idxs)
    return data_set


def get_squad_data_filter(config):
    def data_filter(data_point, shared):
        assert shared is not None
        rx, rcx, q, cq, y = (data_point[key] for key in ('*x', '*cx', 'q', 'cq', 'y'))
        x, cx = shared['x'], shared['cx']
        if len(q) > config.ques_size_th:
            return False

        # x filter
        xi = x[rx[0]][rx[1]]
        if config.squash:
            for start, stop in y:
                stop_offset = sum(map(len, xi[:stop[0]]))
                if stop_offset + stop[1] > config.para_size_th:
                    return False
            return True

        if config.single:
            for start, stop in y:
                if start[0] != stop[0]:
                    return False

        if config.data_filter == 'max':
            for start, stop in y:
                    if stop[0] >= config.num_sents_th:
                        return False
                    if start[0] != stop[0]:
                        return False
                    if stop[1] >= config.sent_size_th:
                        return False
        elif config.data_filter == 'valid':
            if len(xi) > config.num_sents_th:
                return False
            if any(len(xij) > config.sent_size_th for xij in xi):
                return False
        elif config.data_filter == 'semi':
            """
            Only answer sentence needs to be valid.
            """
            for start, stop in y:
                if stop[0] >= config.num_sents_th:
                    return False
                if start[0] != start[0]:
                    return False
                if len(xi[start[0]]) > config.sent_size_th:
                    return False
        else:
            raise Exception()

        return True
    return data_filter


def update_config(config, data_sets):
    config.max_num_sents = 0
    config.max_sent_size = 0
    config.max_ques_size = 0
    config.max_word_size = 0
    config.max_para_size = 0
    for data_set in data_sets:
        data = data_set.data
        shared = data_set.shared
        for idx in data_set.valid_idxs:        
            rx = data['*x'][idx]
            q = data['q'][idx]
            # print("rx: {}".format(rx))
            # if len(shared['x']) <= rx[0]:
            #     print("shared['x'] has len {} and rx[0] is {}".format(len(shared['x']), rx[0]))
            #     assert False
            # if len(shared['x'][rx[0]]) <= rx[1]:
            #     print("shared['x'][rx[0]] has len {} and rx[1] is {}".format(len(shared['x'][rx[0]]), rx[1]))
            #     assert False
            sents = shared['x'][rx[0]][rx[1]]
            # print(sents)
            config.max_para_size = max(config.max_para_size, sum(map(len, sents)))
            config.max_num_sents = max(config.max_num_sents, len(sents))
            config.max_sent_size = max(config.max_sent_size, max(map(len, sents)))
            config.max_word_size = max(config.max_word_size, max(len(word) for sent in sents for word in sent))
            if len(q) > 0:
                config.max_ques_size = max(config.max_ques_size, len(q))
                config.max_word_size = max(config.max_word_size, max(len(word) for word in q))

    if config.mode == 'train':
        config.max_num_sents = min(config.max_num_sents, config.num_sents_th)
        config.max_sent_size = min(config.max_sent_size, config.sent_size_th)
        config.max_para_size = min(config.max_para_size, config.para_size_th)

    config.max_word_size = min(config.max_word_size, config.word_size_th)

    config.char_vocab_size = len(data_sets[0].shared['char2idx'])
    config.word_emb_size = len(next(iter(data_sets[0].shared['word2vec'].values())))
    config.word_vocab_size = len(data_sets[0].shared['word2idx'])

    if config.single:
        config.max_num_sents = 1
    if config.squash:
        config.max_sent_size = config.max_para_size
        config.max_num_sents = 1

if __name__ == "__main__":

        
    flags = tf.app.flags

    sys.path.insert(0,'/data/users/sstauden/dev/nn_robustness/datasets')
    import DatasetPaths

    # Names and directories
    flags.DEFINE_string("model_name", "basic", "Model name [basic]")
    flags.DEFINE_string("data_dir", "data/squad", "Data dir [data/squad]")
    flags.DEFINE_string("run_id", "0", "Run ID [0]")
    flags.DEFINE_string("out_base_dir", "out", "out base dir [out]")
    flags.DEFINE_string("forward_name", "single", "Forward name [single]")
    flags.DEFINE_string("answer_path", "", "Answer path []")
    flags.DEFINE_string("eval_path", "", "Eval path []")
    flags.DEFINE_string("load_path", "", "Load path []")
    flags.DEFINE_string("shared_path", "", "Shared path []")

    # Device placement
    flags.DEFINE_string("device", "/cpu:0", "default device for summing gradients. [/cpu:0]")
    flags.DEFINE_string("device_type", "gpu", "device for computing gradients (parallelization). cpu | gpu [gpu]")
    flags.DEFINE_integer("num_gpus", 1, "num of gpus or cpus for computing gradients [1]")

    # Essential training and test options
    flags.DEFINE_string("mode", "test", "trains | test | forward [test]")
    flags.DEFINE_boolean("load", True, "load saved data? [True]")
    flags.DEFINE_bool("single", False, "supervise only the answer sentence? [False]")
    flags.DEFINE_boolean("debug", False, "Debugging mode? [False]")
    flags.DEFINE_bool('load_ema', True, "load exponential average of variables when testing?  [True]")
    flags.DEFINE_bool("eval", True, "eval? [True]")

    # Training / test parameters
    flags.DEFINE_integer("batch_size", 60, "Batch size [60]")
    flags.DEFINE_integer("val_num_batches", 100, "validation num batches [100]")
    flags.DEFINE_integer("test_num_batches", 0, "test num batches [0]")
    flags.DEFINE_integer("num_epochs", 12, "Total number of epochs for training [12]")
    flags.DEFINE_integer("num_steps", 20000, "Number of steps [20000]")
    flags.DEFINE_integer("load_step", 0, "load step [0]")
    flags.DEFINE_float("init_lr", 0.5, "Initial learning rate [0.5]")
    flags.DEFINE_float("input_keep_prob", 0.8, "Input keep prob for the dropout of LSTM weights [0.8]")
    flags.DEFINE_float("keep_prob", 0.8, "Keep prob for the dropout of Char-CNN weights [0.8]")
    flags.DEFINE_float("wd", 0.0, "L2 weight decay for regularization [0.0]")
    flags.DEFINE_integer("hidden_size", 100, "Hidden size [100]")
    flags.DEFINE_integer("char_out_size", 100, "char-level word embedding size [100]")
    flags.DEFINE_integer("char_emb_size", 8, "Char emb size [8]")
    flags.DEFINE_string("out_channel_dims", "100", "Out channel dims of Char-CNN, separated by commas [100]")
    flags.DEFINE_string("filter_heights", "5", "Filter heights of Char-CNN, separated by commas [5]")
    flags.DEFINE_bool("finetune", False, "Finetune word embeddings? [False]")
    flags.DEFINE_bool("highway", True, "Use highway? [True]")
    flags.DEFINE_integer("highway_num_layers", 2, "highway num layers [2]")
    flags.DEFINE_bool("share_cnn_weights", True, "Share Char-CNN weights [True]")
    flags.DEFINE_bool("share_lstm_weights", True, "Share pre-processing (phrase-level) LSTM weights [True]")
    flags.DEFINE_float("var_decay", 0.999, "Exponential moving average decay for variables [0.999]")

    # Optimizations
    flags.DEFINE_bool("cluster", False, "Cluster data for faster training [False]")
    flags.DEFINE_bool("len_opt", False, "Length optimization? [False]")
    flags.DEFINE_bool("cpu_opt", False, "CPU optimization? GPU computation can be slower [False]")

    # Logging and saving options
    flags.DEFINE_boolean("progress", True, "Show progress? [True]")
    flags.DEFINE_integer("log_period", 100, "Log period [100]")
    flags.DEFINE_integer("eval_period", 1000, "Eval period [1000]")
    flags.DEFINE_integer("save_period", 1000, "Save Period [1000]")
    flags.DEFINE_integer("max_to_keep", 20, "Max recent saves to keep [20]")
    flags.DEFINE_bool("dump_eval", True, "dump eval? [True]")
    flags.DEFINE_bool("dump_answer", True, "dump answer? [True]")
    flags.DEFINE_bool("vis", False, "output visualization numbers? [False]")
    flags.DEFINE_bool("dump_pickle", True, "Dump pickle instead of json? [True]")
    flags.DEFINE_float("decay", 0.9, "Exponential moving average decay for logging values [0.9]")

    # Thresholds for speed and less memory usage
    flags.DEFINE_integer("word_count_th", 10, "word count th [100]")
    flags.DEFINE_integer("char_count_th", 50, "char count th [500]")
    flags.DEFINE_integer("sent_size_th", 400, "sent size th [64]")
    flags.DEFINE_integer("num_sents_th", 8, "num sents th [8]")
    flags.DEFINE_integer("ques_size_th", 30, "ques size th [32]")
    flags.DEFINE_integer("word_size_th", 16, "word size th [16]")
    flags.DEFINE_integer("para_size_th", 256, "para size th [256]")

    # Advanced training options
    flags.DEFINE_bool("lower_word", True, "lower word [True]")
    flags.DEFINE_bool("squash", False, "squash the sentences into one? [False]")
    flags.DEFINE_bool("swap_memory", True, "swap memory? [True]")
    flags.DEFINE_string("data_filter", "max", "max | valid | semi [max]")
    flags.DEFINE_bool("use_glove_for_unk", True, "use glove for unk [False]")
    flags.DEFINE_bool("known_if_glove", True, "consider as known if present in glove [False]")
    flags.DEFINE_string("logit_func", "tri_linear", "logit func [tri_linear]")
    flags.DEFINE_string("answer_func", "linear", "answer logit func [linear]")
    flags.DEFINE_string("sh_logit_func", "tri_linear", "sh logit func [tri_linear]")

    # Ablation options
    flags.DEFINE_bool("use_char_emb", True, "use char emb? [True]")
    flags.DEFINE_bool("use_word_emb", True, "use word embedding? [True]")
    flags.DEFINE_bool("q2c_att", True, "question-to-context attention? [True]")
    flags.DEFINE_bool("c2q_att", True, "context-to-question attention? [True]")
    flags.DEFINE_bool("dynamic_att", False, "Dynamic attention [False]")

    config = flags.FLAGS

    
    prepro_base = "/nethome/sstauden/bi-att-flow/prepro_data/"

    attack_dict = {
        'original': {
            'train': prepro_base + "original", 
            'test': prepro_base + "original"
        },
        'AddSent': {
            'train': prepro_base + "AddSent", 
            'test': prepro_base + "AddSent"
        },
        'AddSentDiv': {
            'train': prepro_base + "AddSentDiv", 
            'test': prepro_base + "AddSentDiv"
        },
        'WordSwap': {
            'train': prepro_base + "WordSwap", 
            'test': prepro_base + "WordSwap"
        },
    }

    attack_name = "WordSwap"
    data_path = attack_dict[attack_name]['train']
    config.out_dir = os.path.join("x_training_" + config.out_base_dir, attack_name)


    dataset = read_data_extended(config, 'dev', False, data_path=data_path, shared_path=data_path)
    print(attack_name, " has ", str(dataset.num_examples), " samples")
    print(len(dataset.data['ids']))
