import os
import difflib
import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
from tqdm import tqdm
from scipy.fftpack import fft
from python_speech_features import mfcc
from random import shuffle
from keras import backend as K

def data_hparams():
    params = tf.contrib.training.HParams(
        # vocab
        data_path='nano/',        
        batch_size=1,
        data_length=10,
        shuffle=True)
    return params


class get_data():
    def __init__(self, args):
        self.data_path = args.data_path
        self.data_length = args.data_length
        self.batch_size = args.batch_size
        self.shuffle = args.shuffle
        self.source_init()

    def source_init(self):
        print('get source list...')
        read_files = []
        wav_file=self.data_path+'sig.txt'
        pny_file=self.data_path+'sq.txt'
        self.wav_lst = []
        self.pny_lst = []        

        with open(wav_file, 'r', encoding='utf8') as f:
            data = f.readlines()
            for line in tqdm(data):
                self.wav_lst.append(line.split())

        with open(pny_file, 'r', encoding='utf8') as f:
            data = f.readlines()
            for line in tqdm(data):
                self.pny_lst.append(line.split())

        if self.data_length:
            self.wav_lst = self.wav_lst[:self.data_length]
            self.pny_lst = self.pny_lst[:self.data_length]
            
        print('make am vocab...')
        self.am_vocab = self.mk_am_vocab(self.pny_lst)
         
        
        
    def get_am_batch(self):
        shuffle_list = [i for i in range(len(self.wav_lst))]
        while 1:
            if self.shuffle == True:
                shuffle(shuffle_list)
            for i in range(len(self.wav_lst) // self.batch_size):
                wav_data_lst = []
                label_data_lst = []
                begin = i * self.batch_size
                end = begin + self.batch_size
                sub_list = shuffle_list[begin:end]
                for index in sub_list:
                    fbank = np.array(self.wav_lst[index])
                    pad_fbank = np.zeros((fbank.shape[0] ))
                    pad_fbank[:fbank.shape[0]] = fbank
                    #Sigmean=np.mean(pad_fbank)
                    #Sigvar=np.std(pad_fbank)
                    #pad_fbank=[(float(i) - Sigmean) / Sigvar for i in pad_fbank]
                    #pad_fbank=np.array(pad_fbank)
                    #pad_fbank.reshape(fbank.shape[0])
                    label = self.pny2id(self.pny_lst[index], self.am_vocab)
                    label_ctc_len = self.ctc_len(label)
                    if pad_fbank.shape[0]//4  >= label_ctc_len:
                        wav_data_lst.append(pad_fbank)
                        label_data_lst.append(label)
                pad_wav_data, input_length = self.wav_padding(wav_data_lst)
                pad_label_data, label_length = self.label_padding(label_data_lst)
                inputs = {'the_inputs': pad_wav_data,
                          'the_labels': pad_label_data,
                          'input_length': input_length,
                          'label_length': label_length,
                          }
                outputs = {'ctc': np.zeros(pad_wav_data.shape[0], )}
                yield inputs, outputs

    

    def pny2id(self, line, vocab):
        return [vocab.index(pny) for pny in line]

    def wav_padding(self, wav_data_lst):
        wav_lens = [len(data) for data in wav_data_lst]
        wav_max_len = max(wav_lens)
        wav_lens = np.array([leng//4  for leng in wav_lens])
        new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 1))
        for i in range(len(wav_data_lst)):
            new_wav_data_lst[i, :wav_data_lst[i].shape[0],0] = wav_data_lst[i]
            
        return new_wav_data_lst, wav_lens

    def label_padding(self, label_data_lst):
        label_lens = np.array([len(label) for label in label_data_lst])
        max_label_len = max(label_lens)
        new_label_data_lst = np.zeros((len(label_data_lst), max_label_len))
        for i in range(len(label_data_lst)):
            new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i]
        return new_label_data_lst, label_lens

    def mk_am_vocab(self, data):
        vocab = []
        for line in tqdm(data):
            line = line
            for pny in line:
                if pny not in vocab:
                    vocab.append(pny)
        vocab.append('_')
        return vocab

    

    def ctc_len(self, label):
        add_len = 0
        label_len = len(label)
        for i in range(label_len - 1):
            if label[i] == label[i + 1]:
                add_len += 1
        return label_len + add_len

# 定义解码器------------------------------------
def decode_ctc(num_result, num2word):
    result = num_result[:, :, :]
    in_len = np.zeros((1), dtype = np.int32)
    in_len[0] = result.shape[1]
    r = K.ctc_decode(result, in_len, greedy = True, beam_width=10, top_paths=1)
    r1 = K.get_value(r[0][0])
    r1 = r1[0]
    text=[]
    for i in r1:
        text.append(num2word[i])
    return r1, text
