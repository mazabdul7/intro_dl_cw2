from urllib.request import build_opener
from tensorflow.keras import Model, layers, Sequential
from tensorflow.keras.applications import efficientnet
import tensorflow as tf
import re
from oeq.modified_efficientnet_attention import EfficientNetB0


class EffnetEncoder:
    ''' EfficientNet based encoder '''

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
        self.base_model = self.load_pretrained_base()

    def load_pretrained_base(self) -> Model:
        ''' Loads and returns a pre-trained EfficientNet model for a specified base name '''
        return EfficientNetB0(input_shape=self.input_shape, include_top=False)
    
    def build_autoencoder_attention(self):
        ''' Builds and returns an autoencoder model'''
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        x = inputs
        output_tensors = []
        encoder_outpus = []
        
        up_stack = [
            self.upsample_block(672, 3),
            self.upsample_block(240, 3),
            self.upsample_block(144, 3),
            self.upsample_block(96, 3),
        ]
        
        down_stack = [
            self.desample_block(320, 3),#8x8 bottleneck
            self.desample_block(672, 3),
            self.desample_block(240, 3),
            self.desample_block(144, 3),
            self.desample_block(96, 3) # 128
        ]
        
        down_stack.reverse()
        
        for down in down_stack:
            x = down(x)
            encoder_outpus.append(x)
        
        encoder_stage = Model(inputs=inputs, outputs=encoder_outpus, name='auto-encoder-encode')
        encoder_outpus = []
        
        inputs2 = tf.keras.layers.Input(shape=self.input_shape)
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
        
    def multiply_factory(self, x, y):
        return layers.Multiply()([x, y])

    def build_effnet_with_attention(self, trainable: bool = False) -> Model:
        ''' Builds and returns the encoder as a Keras model '''
        base_model_outputs = [self.base_model.get_layer(name).output for name in self.layer_names]
        encoder = Model(inputs=self.base_model.input, outputs=base_model_outputs, name=self.encoder_name)
        encoder.trainable = trainable

        return encoder
    
    def se_block(self, in_block, ch, ratio=16):
        x = layers.GlobalAveragePooling2D()(in_block)
        x = layers.Dense(ch//ratio, activation='relu')(x)
        x = layers.Dense(ch, activation='sigmoid')(x)
        return layers.Multiply()([in_block, x])
    
    def upsample_block(self, filters: int, size: int):
        ''' Upsample block '''
        initializer = tf.random_normal_initializer(0., 0.02)

        block = Sequential()
        block.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                  kernel_initializer=initializer, use_bias=False, name='upsample_conv'))
        block.add(layers.BatchNormalization(name='upsample_bn'))
        block.add(layers.ReLU(name='upsample_relu'))

        return block
    
    def desample_block(self, filters: int, size: int):
        ''' Desample block '''
        initializer = tf.random_normal_initializer(0., 0.02)

        block = Sequential()
        block.add(layers.Conv2D(filters, size, strides=2, padding='same',
                  kernel_initializer=initializer, use_bias=False, name='Desample_conv'))
        block.add(layers.ReLU(name='Desample_relu'))
        block.add(layers.BatchNormalization(name='Desample_bn'))

        return block

    def insert_layer_nonseq(self, model, layer_regex, insert_layer_factory,
        insert_layer_name=None, position='after'):

        # Auxiliary dictionary to describe the network graph
        network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

        # Set the input layers of each layer
        for layer in model.layers:
            for node in layer._outbound_nodes:
                layer_name = node.outbound_layer.name
                if layer_name not in network_dict['input_layers_of']:
                    network_dict['input_layers_of'].update(
                            {layer_name: [layer.name]})
                else:
                    network_dict['input_layers_of'][layer_name].append(layer.name)

        # Set the output tensor of the input layer
        network_dict['new_output_tensor_of'].update(
                {model.layers[0].name: model.input})

        # Iterate over all layers after the input
        model_outputs = []
        for layer in model.layers[1:]:

            # Determine input tensors
            layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                    for layer_aux in network_dict['input_layers_of'][layer.name]]
            if len(layer_input) == 1:
                layer_input = layer_input[0]

            # Insert layer if name matches the regular expression
            if re.match(layer_regex, layer.name):
                if position == 'replace':
                    x = layer_input
                elif position == 'after':
                    x = layer(layer_input)
                elif position == 'before':
                    pass
                else:
                    raise ValueError('position must be: before, after or replace')

                new_layer = insert_layer_factory
                if insert_layer_name:
                    new_layer.name = insert_layer_name
                else:
                    new_layer.name = '{}_{}'.format(layer.name, 
                                                    new_layer.name)
                x = new_layer(x)
                print('New layer: {} Old layer: {} Type: {}'.format(new_layer.name,
                                                                layer.name, position))
                if position == 'before':
                    x = layer(x)
            else:
                x = layer(layer_input)

            # Set new output tensor (the original one, or the one of the inserted
            # layer)
            network_dict['new_output_tensor_of'].update({layer.name: x})

            # Save tensor in output list if it is output in initial model
            if layer_name in model.output_names:
                model_outputs.append(x)

        return Model(inputs=model.inputs, outputs=model_outputs)
