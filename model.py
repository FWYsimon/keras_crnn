from keras.layers.convolutional import Conv2D,MaxPooling2D,ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape,Masking,Lambda,Permute
from keras.layers import Input,Dense,Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.layers.recurrent import GRU,LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras import backend as K
from keras.preprocessing import image
from keras.optimizers import Adam,SGD,Adadelta
from keras import losses
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard
from keras.utils import plot_model

import tensorflow as tf  
import keras.backend.tensorflow_backend as K

class CRNN(object):
	"""docstring for CRNN"""
	def __init__(self, args):
		self.img_height = args.img_height
		self.img_width = args.img_width

		self.max_label_length = args.max_label_length

		self.run_unit = args.run_unit
		self.nclass = args.nclass

		self.inputs = Input(shape=(self.img_height, self.img_width, 1),name='the_input')
		self.labels = Input(name='the_labels',shape=[max_label_length],dtype='float32')
		self.input_length = Input(name='input_length', shape=[1], dtype='int64')
		self.label_length = Input(name='label_length', shape=[1], dtype='int64')
	
	def model(self):
		inputs = Input(shape=(self.img_height, 100, 1),name='the_input')
		m = Conv2D(64,kernel_size=(3,3),activation='relu',padding='same',name='conv1')(inputs)
	    m = MaxPooling2D(pool_size=(2,2),strides=(2,2),name='pool1')(m)
	    m = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same',name='conv2')(m)
	    m = MaxPooling2D(pool_size=(2,2),strides=(2,2),name='pool2')(m)
	    m = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same',name='conv3')(m)
	    m = BatchNormalization(axis=3)(m)
	    m = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same',name='conv4')(m)

	    m = ZeroPadding2D(padding=(0,1))(m)
	    m = MaxPooling2D(pool_size=(2,2),strides=(2,1),padding='valid',name='pool3')(m)

	    m = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same',name='conv5')(m)
	    m = BatchNormalization(axis=3)(m)
	    m = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same',name='conv6')(m)

	    m = ZeroPadding2D(padding=(0,1))(m)
	    m = MaxPooling2D(pool_size=(2,2),strides=(2,1),padding='valid',name='pool4')(m)
	    m = Conv2D(512,kernel_size=(2,2),activation='relu',padding='valid',name='conv7')(m)
	    m = BatchNormalization(axis=3)(m)

	    m = Permute((2,1,3),name='permute')(m)
	    m = TimeDistributed(Flatten(),name='timedistrib')(m)

	    m = Bidirectional(GRU(rnnunit,return_sequences=True,implementation=2),name='blstm1')(m)
	    #m = Bidirectional(LSTM(rnnunit,return_sequences=True),name='blstm1')(m)
	    m = Dense(rnnunit,name='blstm1_out',activation='linear',)(m)
	    #m = Bidirectional(LSTM(rnnunit,return_sequences=True),name='blstm2')(m)
	    m = Bidirectional(GRU(self.rnnunit,return_sequences=True,implementation=2),name='blstm2')(m)
	    y_pred = Dense(self.nclass,name='blstm2_out',activation='softmax')(m)
	    
	    basemodel = Model(inputs=inputs,outputs=y_pred)
	    basemodel.summary()
	    
	    return y_pred

	def get_loss(self, y_pred):
		return Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, self.labels, self.input_length, self.label_length])

	def ctc_lambda_func(self, args):
    	y_pred,labels,input_length,label_length = args
    	return K.ctc_batch_cost(labels, y_pred, input_length, label_length)