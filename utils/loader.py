# Ensure you have downloaded the OxPet dataset and unzipped it to datasets folder
import h5py
import os
import tensorflow as tf


class Generator:
    ''' 
        Generator yields inputs from file efficiently. File is opened once
        and yields until outputs are exhausted without loading entire ds into memory.
    '''
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

    def __call__(self):
        with h5py.File(self.path, 'r') as f:  # With scope for safe file exit incase memleaks
            d_name = list(f.keys())[0]
            num_images = len(f[d_name])

            for i in range(self.batch_size, num_images, self.batch_size):  # Batched sliced indexing
                yield f[d_name][i-self.batch_size:i]


class DataLoader:
    def __init__(self, batch_size, batch_size_val=64):
        # Paths relative to working directory
        self.img_path = r'images.h5'
        self.mask_path = r'masks.h5'
        self.bbox_path = r'bboxes.h5'
        self.bin_path = r'binary.h5'
        self.train_path = r'../datasets/train'
        self.val_path = r'../datasets/val'
        self.test_path = r'../datasets/test'

        # Configs
        self.batch_size = batch_size
        self.batch_size_val = batch_size_val

    def load_ds_generator(self, path, val=False):
        ''' 
            Loads and returns batched tf.Dataset generator object from passed path. 
            If val: loads validation set. 
        '''
        ds = tf.data.Dataset.from_generator(
            Generator(os.path.join(self.train_path if not val else self.val_path, path), self.batch_size if not val else self.batch_size_val), tf.float32)

        return ds

    def load_ds_test_set(self, path):
        ''' Returns the entire test set for the passed path'''

        with h5py.File(os.path.join(self.test_path, path), 'r') as f:
            d_name = list(f.keys())[0]

            return f[d_name][:]

    def get_image_ds(self, val=False, test_mode=False):
        ''' 
            Returns batched tf.Dataset generator object for images.h5 dataset 
            If test_mode asserted returns entire test-set for images.h5 dataset
        '''
        if test_mode:
            return self.load_ds_test_set(self.img_path)

        return self.load_ds_generator(self.img_path, val=val)

    def get_mask_ds(self, val=False, test_mode=False):
        ''' 
            Returns batched tf.Dataset generator object for masks.h5 dataset 
            If test_mode asserted returns entire test-set for masks.h5 dataset
        '''
        if test_mode:
            return self.load_ds_test_set(self.mask_path)

        return self.load_ds_generator(self.mask_path, val=val)

    def get_binary_ds(self, val=False, test_mode=False):
        ''' 
            Returns batched tf.Dataset generator object for binary.h5 dataset 
            If test_mode asserted returns entire test-set for binary.h5 dataset
        '''
        if test_mode:
            return self.load_ds_test_set(self.bin_path)

        return self.load_ds_generator(self.bin_path, val=val)

    def get_bboxes_ds(self, val=False, test_mode=False):
        ''' 
            Returns batched tf.Dataset generator object for bboxes.h5 dataset 
            If test_mode asserted returns entire test-set for bboxes.h5 dataset
        '''
        if test_mode:
            return self.load_ds_test_set(self.bbox_path)

        return self.load_ds_generator(self.bbox_path, val=val)
