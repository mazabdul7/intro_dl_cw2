from tensorflow.keras import Model, layers, Sequential
from tensorflow.keras.applications import efficientnet
from tensorflow import random_normal_initializer
from models.modified_efficientnet_attention import EfficientNetB0


class EffnetEncoder:
    ''' EfficientNet based encoder builder '''

    def __init__(self, base_name: str, input_shape: tuple[int]) -> None:
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
        base_model = EfficientNetB0(
            input_shape=self.input_shape, include_top=False)
        base_model_outputs = [base_model.get_layer(
            name).output for name in self.layer_names]
        encoder = Model(inputs=base_model.input, outputs=base_model_outputs,
                        name=self.encoder_name+'_With_Attention')
        encoder.trainable = trainable

        return encoder

    def build_autoencoder_attention(self):
        ''' Builds and returns a basic encoder model with attention (NO EFFICIENTNET BASE)'''
        inputs = layers.Input(shape=self.input_shape)
        x = inputs
        output_tensors = []
        encoder_outpus = []

        up_stack = [
            upsample_block(672, 3),
            upsample_block(240, 3),
            upsample_block(144, 3),
            upsample_block(96, 3),
        ]

        down_stack = [
            desample_block(320, 3),  # 8x8 bottleneck
            desample_block(672, 3),
            desample_block(240, 3),
            desample_block(144, 3),
            desample_block(96, 3)  # 128
        ]

        down_stack.reverse()

        for down in down_stack:
            x = down(x)
            encoder_outpus.append(x)

        encoder_stage = Model(
            inputs=inputs, outputs=encoder_outpus, name='auto-encoder-encode')
        encoder_outpus = []

        inputs2 = layers.Input(shape=self.input_shape)
        features = encoder_stage(inputs2)
        x = features[-1]
        skip_connections = reversed(features[:-1])

        for up, skip in zip(up_stack, skip_connections):
            x = up(x)
            output_tensors.append(x)
            concat = layers.Concatenate()
            x = concat([x, skip])

        output_tensors.reverse()

        x = inputs2
        for i in range(4):
            x = down_stack[i](x)
            sh = x.shape[-1]
            x = self.se_block(x, sh)
            x = layers.Multiply()([x, output_tensors[i]])
            x = self.se_block(x, sh)
            encoder_outpus.append(x)
        x = down_stack[-1](x)
        encoder_outpus.append(x)

        return Model(inputs=inputs2, outputs=encoder_outpus, name='auto-encoder-final')

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