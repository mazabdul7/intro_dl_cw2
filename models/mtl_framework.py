import tensorflow as tf
from tensorflow.keras import Model, Sequential, layers
from models.effnet_encoder import EffnetEncoder, upsample_block, desample_block


class MTLFramework:
    ''' Modulare MTL framework that lets us quickly add auxillary tasks to an encoder in a plug-and-play fashion '''

    def __init__(self, encoder: Model, input_shape: tuple[int]) -> None:
        self.encoder = encoder
        self.input_shape = input_shape
        self.outputs = []
        self.inputs = []

        self.skips = self.get_encoder_features()
        self.encoder_output = self.skips[-1]

    def get_encoder_features(self) -> list[tf.Tensor]:
        ''' Get inter and end feature map outputs from encoder '''
        inputs = tf.keras.layers.Input(shape=self.input_shape, name='input')
        self.inputs.append(inputs)
        features = self.encoder(inputs)

        return features

    def add_segmentation_head(self) -> tf.Tensor:
        ''' Builds and returns output tensor of segmentation head '''
        skip_connections = reversed(self.skips[:-1])
        x = self.encoder_output  # Encoder output tensor

        up_stack = [
            upsample_block(512, 3),
            upsample_block(256, 3),
            upsample_block(128, 3),
            upsample_block(64, 3),
        ]

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skip_connections):
            x = up(x)
            concat = layers.Concatenate()
            x = concat([x, skip])

        # Segmentation output tensor
        seg_out = layers.Conv2DTranspose(
            filters=1, kernel_size=3, strides=2, padding='same', name='segnet_out')(x)
        self.outputs.append(seg_out)

        return seg_out

    def add_binary_classification_head(self, base_name: str, trainable: bool = False, output_size: int = 1, activation: str = 'sigmoid') -> tf.Tensor:
        ''' Builds and returns output tensor of binary classification head '''
        x = self.encoder_output
        base_model = EffnetEncoder(
            'B0', self.input_shape).load_pretrained_base(base_name)

        # Build final classification output layers
        # Get pre-trained final classification layers
        final_layers = base_model.layers[-4:]

        # Freeze layers
        for layer in final_layers:
            layer.trainable = trainable

        x = final_layers[0](x)
        for out_layer in final_layers[-3:]:
            x = out_layer(x)
        x = layers.GlobalAveragePooling2D(name='bin_class_pooling')(x)
        bin_class_out = layers.Dense(
            output_size, activation=activation, name='bin_class_out')(x)
        self.outputs.append(bin_class_out)

        return bin_class_out

    def add_bbox_classification_head(self, base_name: str, trainable: bool = False) -> tf.Tensor:
        ''' Builds and returns output tensor of bounding box regression head '''
        x = self.encoder_output
        base_model = EffnetEncoder(
            'B0', self.input_shape).load_pretrained_base(base_name)

        # Build final classification output layers
        # Get pre-trained final classification layers
        final_layers = base_model.layers[-4:]

        for layer in final_layers:
            layer.trainable = trainable
            layer._name = layer.name + str("_bbox")

        x = final_layers[0](x)
        for out_layer in final_layers[-3:]:
            x = out_layer(x)
        x = layers.GlobalAveragePooling2D(name='bbox_class_pooling')(x)
        x = layers.BatchNormalization()(x)
        bbox_out = layers.Dense(4, activation='linear', name='bbox_out')(x)
        self.outputs.append(bbox_out)

        return bbox_out

    def flush_output_heads(self) -> None:
        ''' Clear output heads of the model '''
        self.outputs.clear()

    def build_mtl_model(self) -> Model:
        ''' Build and return MTL model '''
        return Model(inputs=self.inputs, outputs=self.outputs)
