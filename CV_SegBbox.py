import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
import random
import matplotlib.pyplot as plt
import json

from utils.loader import DataLoader
from models.effnet_encoder import EffnetEncoder
from models.mtl_framework import MTLFramework
from utils import tools, config


def build_model(img_height, img_width, channels):
    ### CLEARS OLD MODELS IN CACHE
    tf.keras.backend.clear_session()

    # Get encoder
    base_model_name = 'B0'
    encoder = EffnetEncoder(base_model_name, (img_height, img_width, channels)).build_encoder(trainable=True)

    # Use our MTL framework to custom build a model
    mtl_builder = MTLFramework(encoder, (img_height, img_width, channels))
    mtl_builder.add_segmentation_head()
    # mtl_builder.add_binary_classification_head(base_model_name, trainable=True)
    mtl_builder.add_bbox_classification_head(base_model_name, trainable=True)
    model = mtl_builder.build_mtl_model()

    return model

def generator_img(img_ds, masks_ds, bbox_ds):
    ''' Merges together datasets into a unified generator to pass for training '''
    a = img_ds.as_numpy_iterator()
    b = masks_ds.as_numpy_iterator()
    # c = label_ds.as_numpy_iterator()
    d = bbox_ds.as_numpy_iterator()
    
    while True:
        X = a.next()
        Y1 = b.next()
        # Y2 = c.next()
        Y3 = d.next()
        
        # Regularisation and shuffling
        X, Y1, Y3 = tools.get_randomised_data([X, Y1, Y3])
        # X, Y1, Y3 = tools.data_augmentation(X, Y1, Y3) # Fix augmentation
        
        yield X, (Y1, Y3)

def generator_img_val(img_ds_val, masks_ds_val, bbox_ds):
    ''' Merges together datasets into a unified generator to pass for training '''
    a = img_ds_val.as_numpy_iterator()
    b = masks_ds_val.as_numpy_iterator()
    # c = label_ds_val.as_numpy_iterator()
    d = bbox_ds.as_numpy_iterator()
    
    while True:
        X = a.next()
        Y1 = b.next()
        # Y2 = c.next()
        Y3 = d.next()
        
        yield X, (Y1, Y3)


if __name__ == "__main__":
    # Set configs
    batch_size = 8
    batch_size_val = 8
    num_train, num_val, num_test = config.config['num_train'], config.config['num_val'], config.config['num_test']
    img_height, img_width, channels = config.config['input_shape']

    ##### Change this to modify the folds #####
    CVfolds = 5
    ep = 10
    ###########################################
    cv_history = []
    num_train_cv = num_train * (CVfolds-1)/CVfolds
    num_val_cv = num_train * 1/CVfolds

    for cv_iter in range(CVfolds):
        # CrossVal: 0 = no CV, 1 = training set, 2 = val set
        loader1 = DataLoader(batch_size=batch_size, batch_size_val=batch_size_val, CrossVal=1, CV_iteration=cv_iter, fold=CVfolds)
        loader2 = DataLoader(batch_size=batch_size, batch_size_val=batch_size_val, CrossVal=2, CV_iteration=cv_iter, fold=CVfolds)

        # Train set
        img_ds = loader1.get_image_ds().repeat()
        masks_ds = loader1.get_mask_ds().repeat()
        label_ds = loader1.get_binary_ds().repeat()
        bbox_ds = loader1.get_bboxes_ds().repeat()

        # Validation set
        img_ds_val = loader2.get_image_ds().repeat()
        masks_ds_val = loader2.get_mask_ds().repeat()
        label_ds_val = loader2.get_binary_ds().repeat()
        bbox_ds_val = loader2.get_bboxes_ds().repeat()
        
        model = build_model(img_height, img_width, channels)

        model.compile(optimizer=keras.optimizers.Adam(1e-4),
                    loss={'segnet_out' : tf.keras.losses.BinaryCrossentropy(from_logits=True),
                            'bbox_out' : tf.keras.losses.MeanAbsoluteError()},
                    loss_weights=[1,1/100], # Scale MAE to BC range
                    metrics=['accuracy'])

        print(f"-------------------- start cross val {cv_iter+1}/{CVfolds} --------------------")

        history = model.fit(generator_img(img_ds, masks_ds, bbox_ds), 
                            validation_data=generator_img_val(img_ds_val, masks_ds_val, bbox_ds), 
                            epochs=ep, 
                            steps_per_epoch=num_train_cv//batch_size, 
                            validation_steps=num_val_cv//batch_size_val)

        # save the final val_acc of each cv step
        cv_history.append(history.history)

    
    # save all the history 
    with open('CV_SegBbox.txt', 'w') as file:
        file.write(json.dumps(cv_history))

    # extract the wanted metrics to plot and print
    result = []
    for i in range(CVfolds):
        result.append([cv_history[i]['segnet_out_accuracy'],
                        cv_history[i]['val_segnet_out_accuracy']])

    # analyze the results
    print("set \tsegmen ")
    for i in range(CVfolds):
        print(f"CV {i+1} \t{np.round(result[i][1][-1], 4)}")

    print("-----------------------------")
    avg = np.round(np.mean(result, axis=0), 4)
    std = np.round(np.std(result, axis=0), 4)
    print(f"Avg  \t{avg[1][-1]} ")
    print(f"Std  \t{std[1][-1]} ")

    # visualize CV
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(CVfolds):
        ax.plot(result[i][0], linestyle=":", label = f"segment acc, cv {i}")
    for i in range(CVfolds):
        ax.plot(result[i][1], linestyle="-", label = f"val segment acc, cv {i}")
    ax.set_xlabel('epochs')
    ax.set_ylabel('accuracy')
    ax.set_title('MTL with segmentation and bbox')
    plt.legend(loc='lower right')
    plt.ylim([0.75, 1])
    plt.grid()
    fig.savefig('CV_SegBbox_plot.png')
