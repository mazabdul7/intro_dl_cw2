from tensorflow.keras import Model
from tensorflow.keras.models import Model, Sequential, clone_model
from tensorflow.keras.layers import Input
from tensorflow.python.keras.layers.core import Dropout, Lambda
from tensorflow.python.keras.layers.convolutional import Conv2D, Conv2DTranspose
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.layers.merge import concatenate


class UNet:
    ''' UNet model '''
    def __init__(self,  input_shape: tuple) -> None:
        self.input_shape = input_shape
        self.downsized_images = []

    def downsample_block(self, filters: int, size: tuple):
        ''' Downsample block '''
        block = Sequential()
        block.add(Conv2D(filters, size, activation='elu', kernel_initializer='he_normal', padding='same'))
        block.add(Dropout(0.1))
        block.add(Conv2D(filters, size, activation='elu', kernel_initializer='he_normal', padding='same'))

        return block

    def upsampled_image(self, filters: int):
        block = Sequential()
        block.add(Conv2DTranspose(filters, (2, 2),  strides=(2, 2), padding='same'))

        return block

    def upsample_block(self, filters: int, size: tuple):
        ''' Upsample block '''
        block = Sequential()

        block.add(Conv2D(filters, size, activation='elu', kernel_initializer='he_normal', padding='same'))
        block.add(Dropout(0.2))
        block.add(Conv2D(filters, size, activation='elu', kernel_initializer='he_normal', padding='same'))

        return block

    def build_model(self) -> Model:
        inputs = Input(self.input_shape)
        filters = [16, 32, 64, 128]
        depth = len(filters)
        last_filter = 256
        kernel_size = (3, 3)
        down_stack = [self.downsample_block(filter, kernel_size) for filter in filters]

        x = Lambda(lambda x: x / 255)(inputs)
        for block in down_stack:
            x = block(x)
            self.downsized_images.append(x)
            x = MaxPooling2D((2, 2))(x)

        # base of the model - basically bottom of the "U"
        x = Conv2D(last_filter, kernel_size, activation='elu', kernel_initializer='he_normal', padding='same')(x)
        x = Dropout(0.3)(x)
        x = Conv2D(last_filter, kernel_size, activation='elu', kernel_initializer='he_normal', padding='same')(x)

        upsampled_images = [self.upsampled_image(filter) for filter in filters]
        # we reverse the range of depth because the input of the first outsampling will have a filter of 256
        # and we want to convert it to 128, then 64 ... so we're going in the reverse order to the downsampling
        up_stack = [self.upsample_block(filters[i], kernel_size) for i in reversed(range(depth))]

        # we reverse them because we want to match the 1st downsized image to the last upsized etc
        # see U diagram of UNet
        self.downsized_images.reverse()
        for i, block in enumerate(up_stack):
            x = upsampled_images[i](x)
            x = concatenate([x, self.downsized_images[i]])
            x = block(x)

        outputs = Conv2D(1, (1, 1))(x)

        return Model(inputs=[inputs], outputs=[outputs])