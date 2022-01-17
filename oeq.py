import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random
from utils.loader import DataLoader
from models.effnet_encoder import EffnetEncoder
from models.mtl_framework import MTLFramework
from utils import tools, config

def generator_img():
    ''' Merges together datasets into a unified generator to pass for training '''
    a = img_ds.as_numpy_iterator()
    b = masks_ds.as_numpy_iterator()
    c = label_ds.as_numpy_iterator()
    d = bbox_ds.as_numpy_iterator()
    
    while True:
        X = a.next()
        Y1 = b.next()
        Y2 = c.next()
        Y3 = d.next()
        
        # Regularisation and shuffling
        X, Y1, Y2, Y3 = tools.get_randomised_data([X, Y1, Y2, Y3])
        X, Y1, Y3 = tools.data_augmentation(X, Y1, Y3) # Fix augmentation
        
        yield X, (Y1, Y2, Y3)

def generator_img_val():
    ''' Merges together datasets into a unified generator to pass for training '''
    a = img_ds_val.as_numpy_iterator()
    b = masks_ds_val.as_numpy_iterator()
    c = label_ds_val.as_numpy_iterator()
    d = bbox_ds.as_numpy_iterator()
    
    while True:
        X = a.next()
        Y1 = b.next()
        Y2 = c.next()
        Y3 = d.next()
        
        yield X, (Y1, Y2, Y3)

# Set configs
batch_size = config.config['batch_size']
batch_size_val = config.config['batch_size']
num_train, num_val, num_test = config.config['num_train'], config.config['num_val'], config.config['num_test']
img_height, img_width, channels = config.config['input_shape']

# Load our data pipeline
print('Loading data...')
loader = DataLoader(batch_size=batch_size, batch_size_val=batch_size_val)

# Train set
img_ds = loader.get_image_ds().repeat()
masks_ds = loader.get_mask_ds().repeat()
label_ds = loader.get_binary_ds().repeat()
bbox_ds = loader.get_bboxes_ds().repeat()

# Validation set
img_ds_val = loader.get_image_ds(val=True).repeat()
masks_ds_val = loader.get_mask_ds(val=True).repeat()
label_ds_val = loader.get_binary_ds(val=True).repeat()
bbox_ds_val = loader.get_bboxes_ds(val=True).repeat()

# Build our MTL
### CLEARS OLD MODELS IN CACHE
tf.keras.backend.clear_session()

# Get encoder with attention unit
print('Building MTL model with attention...')
base_model_name = 'B0'
encoder = EffnetEncoder(base_model_name, (img_height, img_width, channels)).build_encoder_with_attention(trainable=True) # EfficientNet with attention
encoder.summary()

print('Building MTL model...')
mtl_builder = MTLFramework(encoder, (img_height, img_width, channels))
print('Adding Segmentation head to model...')
mtl_builder.add_segmentation_head()
print('Adding Binary Classification head to model...')
mtl_builder.add_binary_classification_head(base_model_name, trainable=True)
print('Adding Bounding Box Regression head to model...')
mtl_builder.add_bbox_classification_head(base_model_name, trainable=True)
model = mtl_builder.build_mtl_model()
model.trainable = True
model.summary()

# Train model
# Initial training
print('\nBeginning training...')
epochs = 15
model.compile(optimizer=keras.optimizers.Adam(),
              loss={'segnet_out' : tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    'bin_class_out' : tf.keras.losses.BinaryCrossentropy(),
                    'bbox_out' : tf.keras.losses.MeanAbsoluteError()},
              loss_weights=[1,1,1/100], # Scale MAE to BC range
              metrics=['accuracy'])
history = model.fit(generator_img(), validation_data=generator_img_val(), epochs=epochs, steps_per_epoch=num_train//batch_size, validation_steps=num_val//batch_size_val)

# Fine-tuning at lower LR
print('Beginning fine-tuning...')
model.compile(optimizer=keras.optimizers.Adam(1e-4),
              loss={'segnet_out' : tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    'bin_class_out' : tf.keras.losses.BinaryCrossentropy(),
                    'bbox_out' : tf.keras.losses.MeanAbsoluteError()},
              loss_weights=[1,1,1/100], # Scale MAE to BC range
              metrics=['accuracy'])
history_sec = model.fit(generator_img(), validation_data=generator_img_val(), epochs=10, steps_per_epoch=num_train//batch_size, validation_steps=num_val//batch_size_val)

print('Saving plot of training...')
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(list(range(epochs)), history.history['segnet_out_accuracy'], 'r-', label='Segmentation - Training Accuracy')
ax.plot(list(range(epochs)), history.history['val_segnet_out_accuracy'], 'r--', label='Segmentation - Validation Accuracy')
ax.plot(list(range(epochs)), history.history['bin_class_out_accuracy'], 'c-', label='Classification - Training Accuracy')
ax.plot(list(range(epochs)), history.history['val_bin_class_out_accuracy'], 'c--', label='Classification - Validation Accuracy')
ax2 = ax.twinx()
ax.plot(list(range(epochs)), history.history['bbox_out_accuracy'], 'm-', label='Bounding Box - Training Accuracy')
ax.plot(list(range(epochs)), history.history['val_bbox_out_accuracy'], 'm--', label='Bounding Box - Validation Accuracy')
ax.legend()
ax.set_xlabel('Epochs')
ax.set_ylabel('Segmentation/Classification Accuracy')
ax2.set_ylabel('Bounding Box Accuracy')
fig.savefig('main_mtl_training_history_plot.png')

print('Saving model...')
model.save('model_weights/EffishingNetAtt_Eff')

# ## Test on test-set
print('Loading saved model...')
model = tf.keras.models.load_model('model_weights/EffishingNetAtt_Eff')

# Load test-set
print('Testing model performance...')
img_ds_test = loader.get_image_ds(test_mode=True)
masks_ds_test = loader.get_mask_ds(test_mode=True)
label_ds_test = loader.get_binary_ds(test_mode=True)
bbox_ds_test = loader.get_bboxes_ds(test_mode=True)

# Predict on test-set
seg_pred, bin_pred, bbox_pred = model.predict(img_ds_test, batch_size=10)
seg_pred = tf.where(seg_pred >= 0, 1, 0) # Convert to {0,1} binary classes
bin_pred = np.round(bin_pred) # Round confidence score

bin_acc = np.sum(bin_pred == label_ds_test)/label_ds_test.shape[0]
seg_acc = np.sum(seg_pred == masks_ds_test)/(masks_ds_test.shape[0]*(img_height*img_width))
iou = np.mean(tools.calculate_iou(bbox_ds_test, bbox_pred))
print(f'Binary Acc: {round(bin_acc*100, 3)}%,   Seg Acc: {round(seg_acc*100, 3)}%,    BBox IOU: {round(iou*100, 3)}%')

# Get precision
m = tf.keras.metrics.Precision()
m.update_state(seg_pred, masks_ds_test)
print(f'Precision of network: {m.result().numpy()}')

# Get recall
m = tf.keras.metrics.Recall()
m.update_state(seg_pred, masks_ds_test)
print(f'Recall of network: {m.result().numpy()}')

# Get Dice metric
m = tf.keras.metrics.MeanIoU(num_classes=2)
m.update_state(seg_pred, masks_ds_test)
print(f'Dice score of network: {m.result().numpy()}')

print('\nSaving visualisation of predictions...')
# Visualise predictions
idx = list(range(img_ds_test.shape[0]))
random.shuffle(idx)
for i in range(3):
    tools.show_seg_pred(img_ds_test[idx[i]], masks_ds_test[idx[i]], seg_pred[idx[i]][tf.newaxis, ...], bbox_ds_test[idx[i]], bbox_pred[idx[i]])

print('Getting attention maps from model...')
# # Get feature maps
input_img = next(img_ds.as_numpy_iterator()) # Get test image

# Build new model with intermediate layer as output
XX = model.input 
YY = model.layers[-12].output
new_model = tf.keras.Model(XX, YY)
Xresult = new_model.predict(input_img) # Get feature map

# Plot featuremap
fig, ax = plt.subplots(5, 10, figsize=(15, 7))

for j in range(5):
    for i in range(10):
        ax[j][i].imshow(Xresult[0, :, :, j*20 + i + 10])
        ax[j][i].axis('off')
        
fig.savefig('attention_maps.png')


