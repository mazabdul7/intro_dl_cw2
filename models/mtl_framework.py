import tensorflow as tf
from tensorflow.keras import Model, Sequential, layers
from models.effnet_encoder import EffnetEncoder

class MTLFramework:
    def __init__(self, encoder: Model, input_shape: tuple[int]) -> None:
        self.encoder = encoder
        self.input_shape = input_shape
        self.outputs = []
        self.inputs = []
        
        self.skips = self.get_encoder_features()
        self.encoder_output = self.skips[-1]
        
    def get_encoder_features(self) -> list[tf.Tensor]:
        ''' Get inter and end feature map outputs from encoder '''
        inputs = tf.keras.layers.Input(shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        self.inputs.append(inputs)
        features = self.encoder(inputs)
        
        return features
    
    def upsample_block(self, filters: int, size: int):
        ''' Upsample block '''
        initializer = tf.random_normal_initializer(0., 0.02)

        block = Sequential()
        block.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False, name='upsample_conv'))
        block.add(layers.BatchNormalization(name='upsample_bn'))
        block.add(layers.ReLU(name='upsample_relu'))

        return block
    
    def add_segmentation_head(self) -> tf.Tensor:
        ''' Builds and returns output tensor of segmentation head '''
        skip_connections = reversed(self.skips[:-1])
        x = self.encoder_output # Encoder output tensor

        up_stack = [
            self.upsample_block(512, 3),
            self.upsample_block(256, 3),
            self.upsample_block(128, 3),
            self.upsample_block(64, 3), 
        ]

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skip_connections):
            x = up(x)
            concat = layers.Concatenate()
            x = concat([x, skip])

        # Segmentation output tensor
        seg_out = layers.Conv2DTranspose(filters=2, kernel_size=3, strides=2, padding='same', name='segnet_out')(x)
        self.outputs.append(seg_out)
        
        return seg_out
    
    def add_binary_classification_head(self, base_name: str, trainable: bool = False) -> tf.Tensor:
        x = self.encoder_output
        base_model = EffnetEncoder('B0', self.input_shape).load_pretrained_base(base_name)
        
        # Build final classification output layers
        final_layers = base_model.layers[-4:] # Get pre-trained final classification layers
        
        # Freeze layers
        for layer in final_layers:
            layer.trainable = trainable
        
        x = final_layers[0](x)
        for out_layer in final_layers[-3:]:
            x = out_layer(x)
        x = layers.GlobalAveragePooling2D(name='bin_class_pooling')(x)
        bin_class_out = layers.Dense(2, activation='softmax', name='bin_class_out')(x)
        self.outputs.append(bin_class_out)
        
        return bin_class_out
    
    def flush_output_heads(self) -> None:
        self.outputs.clear()
    
    def build_mtl_model(self) -> Model:
        ''' Build and return MTL model '''
        return Model(inputs=self.inputs, outputs=self.outputs)