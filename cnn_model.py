from typing import Tuple, Optional
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, LeakyReLU, Add, Dropout, \
    GlobalAveragePooling2D, Input, MaxPooling2D
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor


class SmallImageNet:
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, num_filters: int,
                 regularization: Optional[float], input_name: str, output_name: str):
        """
        Custom implementation of the lighted ResNet for small image classification task.
        :param input_shape: input shape (height, width, channels).
        :param num_classes: number of classes.
        :param num_filters: network expansion factor, determines the number of filters in start layer.
        :param regularization: pass float < 1 (good is 0.0005) to make regularization or None to ignore it.
        :param input_name: name of the input tensor.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.input_name = input_name
        self.output_name = output_name
        self.init_filters = num_filters
        self.ker_reg = None if regularization is None else tf.keras.regularizers.l2(regularization)

    def build(self) -> tf.keras.models.Model:
        """
        Building CNN model.
        :return: Model() object.
        """
        inputs = Input(shape=self.input_shape, name=self.input_name)

        x = Conv2D(self.init_filters, (5, 5), strides=1, kernel_regularizer=self.ker_reg, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D((2, 2), strides=1)(x)
        x = self.conv_res_block(x, self.init_filters * 2)
        x = self.conv_res_block(x, self.init_filters * 2)
        x = Dropout(0.1)(x)
        x = self.conv_res_block(x, self.init_filters * 2 ** 2, res=True)
        x = self.conv_res_block(x, self.init_filters * 2 ** 2, res=True)
        x = GlobalAveragePooling2D()(x)
        x = Dense(self.num_classes, kernel_regularizer=self.ker_reg, activation='softmax', name=self.output_name)(x)

        return Model(inputs=inputs, outputs=x)

    def conv_res_block(self, x: Tensor, filters: int, res: bool = False) -> Tensor:
        """
        Convolutional residual block.
        :param x: input tensor.
        :param filters: number of filters in output tensor.
        :param res: determines whether the given block is residual or not.
        :return: output tensor.
        """
        conv_kwargs = {'strides': 1, 'use_bias': False, 'padding': 'same', 'kernel_regularizer': self.ker_reg}
        x1 = Conv2D(filters, (3, 3), **conv_kwargs)(x)
        x1 = BatchNormalization()(x1)
        x1 = LeakyReLU()(x1)
        x1 = Conv2D(filters, (3, 3), **conv_kwargs)(x1)
        if res:
            x2 = Conv2D(filters, (1, 1),  **conv_kwargs)(x)
            x1 = Add()([x1, x2])
        x1 = BatchNormalization()(x1)
        x1 = LeakyReLU()(x1)
        x_out = MaxPooling2D(pool_size=(2, 2))(x1)
        return x_out
