import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

np.random.seed(3)

BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 30

# Two volume: 1000, 10000
# Two resolution: 28*28, 64*64
volume = 1000
# volume = 10000
img_rows = 28 
img_cols = 28 
# img_rows = 64
# img_cols = 64

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

train_rand_idxs = np.random.choice(60000, int(volume*0.8))
test_rand_idxs = np.random.choice(10000, int(volume*0.2))

X_train = X_train[train_rand_idxs]
Y_train = Y_train[train_rand_idxs]
X_test = X_test[test_rand_idxs]
Y_test = Y_test[test_rand_idxs]

X_train = np.asarray([cv2.resize(image, (img_rows, img_cols)) for image in X_train])
X_test = np.asarray([cv2.resize(image, (img_rows, img_cols)) for image in X_test])

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
ax1.imshow(X_train[0], cmap='gray', vmin=0, vmax=255)
ax2.imshow(X_train[1], cmap='gray', vmin=0, vmax=255)
ax3.imshow(X_train[2], cmap='gray', vmin=0, vmax=255)
ax4.imshow(X_train[3], cmap='gray', vmin=0, vmax=255)
ax5.imshow(X_train[4], cmap='gray', vmin=0, vmax=255)

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

Y_train = keras.utils.to_categorical(Y_train, NUM_CLASSES)
Y_test = keras.utils.to_categorical(Y_test, NUM_CLASSES)

input_shape = [img_rows, img_cols, 1]
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

train_start = time.time()
hist = model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
train_end = time.time()
print('Train time:', '%s seconds' % (train_end - train_start))
print('Train loss:', hist.history['loss'][-1])
print('Train accuracy:', hist.history['acc'][-1])

test_start = time.time()
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
test_end = time.time()
print('Test time:', '%s seconds' % (test_end - test_start))
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

model.save('./models/mnist_model_' + str(volume) + '_' + str(img_rows*img_cols))

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
loss_ax.plot(hist.history['loss'], 'y', label='train loss')
acc_ax.plot(hist.history['acc'], 'b', label='train acc')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()
