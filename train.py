import argparse
import h5py

from keras.optimizers import Adam,SGD,Adadelta
from keras.models import Model

import dataset
import model

parser = argparse.ArgumentParser(description='crnn model to recognise the words in image')

# Location of data
parser.add_argument('--image_path', type=str, default='ICPR_text_train_part2_20180313/image_9000',
                    help='image location')
parser.add_argument('--text_path', type=str, default='ICPR_text_train_part2_20180313/txt_9000',
                    help='label location')
parser.add_argument('--json_path', type=str, default='json/meta.json',
                    help='json file location')
parser.add_argument('--json_val_path', type=str, default='json/val.json',
                    help='json validation file location')
parser.add_argument('--save_path', type=str, default='data',
                    help='image after preprocess')
parser.add_argument('--key_path', type=str, default='key/keys.txt',
                    help='all char in labels')
parser.add_argument('--model_path', type=str, default='model',
                    help='model path')

parser.add_argument('--image_height', type=int, default=32,
                    help='image height')
parser.add_argument('--image_width', type=int, default=100,
                    help='image width')
parser.add_argument('--max_label_length', type=int, default=20,
                    help='max label length')
# model options
parser.add_argument('--rnn_unit', type=int, default=32,
                    help='rnn unit')

# Training options
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')

args = parser.parse_args()

# data pre-processing
dataset = Dataset(args)
dataset.data_preprocess()
dataset.rescale()
dataset.generate_key()
dataset.random_get_val()

train = dataset.generate(args.json_path, args.save_path, args.key_path, args.batch_size, args.max_label_length, (args.image_height, args.image_width))
val = dataset.generate(args.json_val_path, args.save_path, args.key_path, args.batch_size, args.max_label_length, (args.image_height, args.image_width))

crnn = CRNN(args)
y_pred = crnn.model()
loss = crnn.get_loss(y_pred)
inputs = crnn.inputs
labels = crnn.labels
input_length = crnn.input_length
label_length = crnn.label_length
model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss)
adam = Adam()
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam,metrics=['accuracy'])
checkpoint = ModelCheckpoint(args.model_path + (r'weights-{epoch:02d}.hdf5'),
                           save_weights_only=True)
earlystop = EarlyStopping(patience=10)
tensorboard = TensorBoard(args.model_path + '/tflog',write_graph=True)

res = model.fit_generator(train,
                    steps_per_epoch = dataset.train_length // args.batch_size,
                    epochs = args.epochs,
                    validation_data = val ,
                    validation_steps = dataset.val_length // args.batch_size,
                    callbacks =[earlystop,checkpoint,tensorboard],
                    verbose=1
                    )