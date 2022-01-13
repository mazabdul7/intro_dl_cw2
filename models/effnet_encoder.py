from tensorflow.keras import Model, layers, Sequential
from tensorflow.keras.applications import efficientnet
from tensorflow import random_normal_initializer
from models.modified_efficientnet_attention import EfficientNetB0
from typing import Tuple


class EffnetEncoder:
    ''' EfficientNet based encoder builder '''

    def __init__(self, base_name: str, input_shape: Tuple[int]) -> None:
        self.encoder_name = f'Effnet{base_name}Encoder'
        self.input_shape = input_shape
        self.layer_names = [
            'block2a_expand_activation',
            'block3a_expand_activation',
            'block4a_expand_activation',
            'block6a_expand_activation',
            'block7a_project_conv'
        ]

        # Download our pretrained base
        self.base_model = self.load_pretrained_base(base_name)

    def load_pretrained_base(self, base_name: str) -> Model:
        ''' Loads and returns a pre-trained EfficientNet model for a specified base name '''
        model_mapping = {
            'B0': efficientnet.EfficientNetB0(include_top=False, input_shape=self.input_shape, weights='imagenet'),
            'B1': efficientnet.EfficientNetB1(include_top=False, input_shape=self.input_shape, weights='imagenet')
        }

        return model_mapping[base_name]

    def build_encoder(self, trainable: bool = False) -> Model:
        ''' Builds and returns the EfficientNet encoder as a Keras model '''
        base_model_outputs = [self.base_model.get_layer(
            name).output for name in self.layer_names]
        encoder = Model(inputs=self.base_model.input,
                        outputs=base_model_outputs, name=self.encoder_name)
        encoder.trainable = trainable

        return encoder

    def build_encoder_with_attention(self, trainable: bool = False) -> Model:
        ''' 
            Builds and returns the EfficientNet encoder with a Sigmoid gated Attention Unit as a Keras model.
            We modified the TensorFlow EfficientNet build script to incorporate the autoencoder attention (see modified_efficientnet_attention.py)
        '''
        layer_names = [
            'block2a_att_multiply',
            'block3a_att_multiply',
            'block4a_att_multiply',
            'block6a_att_multiply',
            'block7a_project_conv'
        ]
        base_model = EfficientNetB0(
            input_shape=self.input_shape, include_top=False)
        base_model_outputs = [base_model.get_layer(
            name).output for name in layer_names]
        encoder = Model(inputs=base_model.input, outputs=base_model_outputs,
                        name=self.encoder_name+'_With_Attention')
        encoder.trainable = trainable

        return encoder

def desample_block(filters: int, size: int):
    ''' Desample block '''
    initializer = random_normal_initializer(0., 0.02)

    block = Sequential()
    block.add(layers.Conv2D(filters, size, strides=2, padding='same',
                kernel_initializer=initializer, use_bias=False, name='Desample_conv'))
    block.add(layers.BatchNormalization(name='Desample_bn'))
    block.add(layers.ReLU(name='Desample_relu'))

    return block

def upsample_block(filters: int, size: int):
    ''' Upsample block '''
    initializer = random_normal_initializer(0., 0.02)

    block = Sequential()
    block.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                kernel_initializer=initializer, use_bias=False, name='upsample_conv'))
    block.add(layers.BatchNormalization(name='upsample_bn'))
    block.add(layers.ReLU(name='upsample_relu'))

    return block