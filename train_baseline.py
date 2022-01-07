import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

import matplotlib.pyplot as plt
import random

from models.unet_model import UNet
from utils.config import config
from utils.loader import DataLoader
from utils.tools import generator_img_baseline_data, dice_binary

batch_size = config['unet_batch_size']
batch_size_val = config['unet_batch_size']

num_train = config['num_train']
num_val = config['num_val']

cross_validation_folds = config['cross_validation_folds']

num_epochs = 4

loss_per_fold = []
dice_binary_per_fold = []
accuracy_per_fold = []

val_loss_per_fold = []
val_dice_binary_per_fold = []
val_accuracy_per_fold = []

model_histories = []
for fold in range(cross_validation_folds):
    print(f"-------------------- start cross val {fold + 1} --------------------")
    # CrossVal: 0 = no CV, 1 = training set, 2 = val set
    loader1 = DataLoader(batch_size=batch_size, batch_size_val=batch_size_val, CrossVal=1, CV_iteration=fold)
    loader2 = DataLoader(batch_size=batch_size, batch_size_val=batch_size_val, CrossVal=2, CV_iteration=fold)

    # get training data
    train_imgs = loader1.get_image_ds().repeat()
    train_masks = loader1.get_mask_ds().repeat()

    # get validation data
    val_imgs = loader2.get_image_ds().repeat()
    val_masks = loader2.get_mask_ds().repeat()

    # Instantiate and build model
    input_shape = config['input_shape']
    model = UNet(input_shape=input_shape).build_model()

    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=[{'Accuracy': 'accuracy', 'dice': dice_binary}])
    model.summary()

    # fit training and validation data
    # we have a checkpointer that saves the best unet model
    checkpointer = ModelCheckpoint(f'model_weights/best_UNet_model_{fold}.h5', verbose=1, save_best_only=True)
    history = model.fit(generator_img_baseline_data(train_imgs, train_masks), validation_data=generator_img_baseline_data(val_imgs, val_masks), batch_size=batch_size, epochs=num_epochs, callbacks=[checkpointer], steps_per_epoch=num_train//batch_size, validation_steps=num_val//batch_size_val)
    loss_per_fold.append(history.history['loss'])
    accuracy_per_fold.append(history.history['accuracy'])
    dice_binary_per_fold.append(history.history['dice_binary'])

    val_loss_per_fold.append(history.history['val_loss'])
    val_accuracy_per_fold.append(history.history['val_accuracy'])
    val_dice_binary_per_fold.append(history.history['val_dice_binary'])

    model_histories.append(history)

print('Scores per fold:\n')
for i in range(cross_validation_folds):
    print(f'For fold {i}:\n')
    print(f'\tTraining:')
    print(f'\t\tAccuracy: {accuracy_per_fold[i]*100} %')
    print(f'\t\tLoss: {loss_per_fold[i]*100} %')
    print(f'\t\tDice Binary: {dice_binary_per_fold[i]*100} %\n')

    print(f'\tValidation:')
    print(f'\t\tAccuracy: {val_accuracy_per_fold[i]*100} %')
    print(f'\t\tLoss: {val_loss_per_fold[i]*100} %')
    print(f'\t\tDice Binary: {val_dice_binary_per_fold[i]*100} %')

print('Average Scores for UNET:\n')
print(f'\tTraining:')
print(f'\t\tAccuracy: {np.mean(accuracy_per_fold) * 100} %')
print(f'\t\tLoss: {np.mean(loss_per_fold) * 100} %')
print(f'\t\tDice Binary: {np.mean(dice_binary_per_fold) * 100} %\n')

print(f'\tValidation:')
print(f'\t\tAccuracy: {np.mean(val_accuracy_per_fold) * 100} %')
print(f'\t\tLoss: {np.mean(val_loss_per_fold) * 100} %')
print(f'\t\tDice Binary: {np.mean(val_dice_binary_per_fold) * 100} %')

# pick best model out of the 3 - used loss we can use some other factor
best_model_idx = np.argmin(loss_per_fold)
# load best model
model = load_model(f'model_weights/best_UNet_model_{best_model_idx}.h5')

best_model_history = model_histories[best_model_idx]
epochs = np.arange(0, num_epochs)
# plot all metrics
plt.plot(epochs, best_model_history.history['loss'], label='loss')
plt.plot(epochs, best_model_history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(epochs, best_model_history.history['accuracy'], label='accuracy')
plt.plot(epochs, best_model_history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

plt.plot(epochs, best_model_history.history['dice_binary'], label='dice_binary')
plt.plot(epochs, best_model_history.history['val_dice_binary'], label='val_dice_binary')
plt.legend()
plt.show()

# fixme not sure if this is right
loader = DataLoader(batch_size=batch_size, batch_size_val=batch_size_val, CrossVal=0, CV_iteration=0)

# get test data
img_ds_test = loader.get_image_ds(test_mode=True)
masks_ds_test = loader.get_mask_ds(test_mode=True)

test_preds = model.predict(img_ds_test, batch_size=batch_size)
formatted_test_preds = (test_preds >= 0.5).astype(np.uint8)

seg_acc = np.sum(formatted_test_preds == masks_ds_test)/(masks_ds_test.shape[0]*(255*255))
print(f"Accuracy {seg_acc*100}%")

# print 3 ground truth/prediction masks
idx = list(range(img_ds_test.shape[0]))
random.shuffle(idx)
for i in range(3):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(tf.keras.utils.array_to_img( masks_ds_test[idx[i]]))
    ax1.set_title('Truth')
    ax2.imshow(tf.keras.utils.array_to_img(formatted_test_preds[idx[i]]))
    ax2.set_title('Prediction')

    plt.show()


