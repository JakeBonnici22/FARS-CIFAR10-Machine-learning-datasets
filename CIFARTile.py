import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Input, Dense
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import KFold

x_train = np.load('train_x.npy')
y_train = np.load('train_y.npy')
x_valid = np.load('valid_x.npy')
y_valid = np.load('valid_y.npy')
x_test = np.load('test_x.npy')
y_test = np.load('test_y.npy')
# print(x_train.shape)
# print(y_train.shape)
# print(x_valid.shape)
# print(y_valid.shape)
# print(x_test.shape)
# print(y_test.shape)

def image_normalization(arr):
    return (arr - arr.min())/(arr.max()-arr.min())


def disable_ax_ticks(ax):
    ax.set_xticks([])
    ax.set_xticks([], minor=True)
    ax.set_yticks([])
    ax.set_yticks([], minor=True)


def show_example(x, y):
    fig = plt.figure()
    main_ax = fig.add_subplot()
    fig.suptitle('label = ' + str(y))
    main_ax.imshow(image_normalization(np.moveaxis(x, 0, -1)))
    disable_ax_ticks(main_ax)
    plt.show()


x_train = image_normalization(x_train)
x_valid = image_normalization(x_valid)
x_test = image_normalization(x_test)
count = x_train.shape[0]
ri = random.randrange(count)
# show_example(x_train[0], y_train[0])
# print(x_train.shape)

x_train = np.moveaxis(x_train, 1, -1)
x_valid = np.moveaxis(x_valid, 1, -1)
x_test = np.moveaxis(x_test, 1, -1)
# print(x_train.shape)
# print(x_valid.shape)

num_classes = len(np.unique(y_train))
print((num_classes))
batch_size = 512
epochs = 50
# Define per-fold score containers <-- these are new
acc_per_fold = []
loss_per_fold = []

# Merge inputs and targets
inputs = np.concatenate((x_train, x_valid), axis=0)
targets = np.concatenate((y_train, y_valid), axis=0)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=5, shuffle=True)

#Data Augmentation
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(64, 64, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)


# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):
    cnn = models.Sequential([

      # data_augmentation,

      layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(64, 64, 3)),
      layers.MaxPooling2D(),
      # layers.Dropout(0.2),

      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Dropout(0.5),

      layers.Conv2D(128, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),


      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes, activation='softmax')
    ])

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    cnn.compile(optimizer='AdaDelta',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                patience=5,
                                                verbose=1,
                                                factor=0.2,
                                                min_lr=0.0001)

    history = cnn.fit(x_train, y_train, epochs=epochs, validation_data=(x_valid, y_valid), batch_size=batch_size,
            callbacks=[learning_rate_reduction]) #, callbacks=[learning_rate_reduction]

    # Generate generalization metrics
    scores = cnn.evaluate(inputs[test], targets[test], verbose=0)
    print(
        f'Score for fold {fold_no}: {cnn.metrics_names[0]} of {scores[0]}; {cnn.metrics_names[1]} of {scores[1] * 100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

epochs_range = range(epochs)
test_loss, test_acc = cnn.evaluate(x_test, y_test, verbose=2)

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.savefig("acc_loss.png", bbox_inches='tight')
plt.show()


test_loss, test_acc = cnn.evaluate(x_test, y_test, verbose=2)
print(test_acc, test_loss)
