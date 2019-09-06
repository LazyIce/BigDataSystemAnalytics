import keras
from keras.datasets import cifar10, cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

np.random.seed(3)

BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS = 100

# Two volume: 1000, 10000
# Two resolution: 32*32, 64*64
volume = 1000
# volume = 10000
img_rows = 32 
img_cols = 32 
# img_rows = 64
# img_cols = 64

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

train_rand_idxs = np.random.choice(50000, int(volume*0.8))
test_rand_idxs = np.random.choice(10000, int(volume*0.2))

X_train = X_train[train_rand_idxs]
Y_train = Y_train[train_rand_idxs]
X_test = X_test[test_rand_idxs]
Y_test = Y_test[test_rand_idxs]

X_train = np.asarray([cv2.resize(image, (img_rows, img_cols)) for image in X_train])
X_test = np.asarray([cv2.resize(image, (img_rows, img_cols)) for image in X_test])

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
ax1.imshow(X_train[0])
ax2.imshow(X_train[1])
ax3.imshow(X_train[2])
ax4.imshow(X_train[3])
ax5.imshow(X_train[4])

(X_train_outlier, Y_train_outlier), (X_test_outlier, Y_test_outlier) = cifar100.load_data()
idx = [i for (i, y) in enumerate(Y_test_outlier) if y == 88][:10]
X_test_outlier = X_test_outlier[idx]
Y_test_outlier = Y_test_outlier[idx] 
X_test_outlier = np.asarray([cv2.resize(image, (img_rows, img_cols)) for image in X_test_outlier])
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
ax1.imshow(X_test_outlier[0])
ax2.imshow(X_test_outlier[1])
ax3.imshow(X_test_outlier[2])
ax4.imshow(X_test_outlier[3])
ax5.imshow(X_test_outlier[4])

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_test_outlier = X_test_outlier.astype('float32') / 255.0

Y_train = keras.utils.to_categorical(Y_train, NUM_CLASSES)
Y_test = keras.utils.to_categorical(Y_test, NUM_CLASSES)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES))
model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

train_start = time.time()
hist = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True)
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

outlier_results = model.predict(X_test_outlier)
outlier_results = [np.argmax(x) for x in outlier_results]
print(outlier_results)

model.save('./models/cifar10_model_' + str(volume) + '_' + str(img_rows*img_cols))

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
