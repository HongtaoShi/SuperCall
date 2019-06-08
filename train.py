import os
import tensorflow as tf
import numpy as np
from utils import get_data, data_hparams
from keras.callbacks import ModelCheckpoint


# 0.准备训练所需数据------------------------------
data_args = data_hparams()
data_args.data_path = 'nano/'
data_args.batch_size = 23
data_args.data_length = 46
# data_args.data_length = None
data_args.shuffle = True
train_data = get_data(data_args)


# 0.准备验证所需数据------------------------------
data_args = data_hparams()
data_args.data_path = 'nano/'

data_args.batch_size = 23
# data_args.data_length = None
data_args.data_length = 46
data_args.shuffle = True
dev_data = get_data(data_args)

# 1.声学模型训练-----------------------------------
from model_speech.cnn_ctc import Am, am_hparams
am_args = am_hparams()
am_args.vocab_size = len(train_data.am_vocab)
am_args.gpu_nums = 1
am_args.lr = 0.0008
am_args.is_training = True
am = Am(am_args)

'''if os.path.exists('logs_am/model.h5'):
    print('load acoustic model...')
    am.ctc_model.load_weights('logs_am/model.h5')'''

#epochs = 10
batch_num = len(train_data.wav_lst) // train_data.batch_size

'''# checkpoint
ckpt = "model_{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(os.path.join('./checkpoint', ckpt), monitor='val_loss', save_weights_only=False, verbose=1, save_best_only=True)'''

#
# for k in range(epochs):
#     print('this is the', k+1, 'th epochs trainning !!!')
#     batch = train_data.get_am_batch()
#     dev_batch = dev_data.get_am_batch()
#     am.ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=10, callbacks=[checkpoint], workers=1, use_multiprocessing=False, validation_data=dev_batch, validation_steps=200)

batch = train_data.get_am_batch()
'''inputs, _ = next(batch)
print('the_inputs',np.shape(inputs['the_inputs']))
print('input_length',inputs['input_length'])
print('the_labels',np.shape(inputs['the_labels']))
print('label_length',inputs['label_length'])
print(batch_num)'''
dev_batch = dev_data.get_am_batch()

#am.ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=10, callbacks=[checkpoint], workers=1, use_multiprocessing=False, validation_data=dev_batch, validation_steps=200)
am.ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=10,  workers=50, use_multiprocessing=True, validation_data=dev_batch, validation_steps=200)
am.ctc_model.save_weights('logs_am/nano.h5')


