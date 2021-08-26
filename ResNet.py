from typing import Tuple, Optional
from tensorflow.keras.layers import Layer, BatchNormalization, Conv2D, Dense, LeakyReLU, Add, Dropout, \
    GlobalAveragePooling2D, Input, MaxPooling2D
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor


class ResidualBlock(Layer):

    def __init__(self, num_filters, regularization, stride=1, **kwargs):
        """
        Residual block. If stride == 1, then there are no any transformations in one of the branches.
        If stride > 1, then there are convolution with 1x1 filters in one of the branches.
        :param filters_num: number of filters in output tensor.
        :param stride: convolution stride.
        :param regularization: pass float < 1 (good is 0.0005) to make regularization or None to ignore it.
        """
        super(ResidualBlock, self).__init__(**kwargs)
        self.strides = stride
        self.ker_reg = None if regularization is None else regularization
        self.num_filters = num_filters

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'strides': self.strides,
            'ker_reg': self.ker_reg,
            'num_filters': self.num_filters,
        })
        return config

    def build(self, input_shape):
        self.BatchNorm = BatchNormalization(input_shape=input_shape)
        self.conv_1 = Conv2D(self.num_filters, (3, 3), strides=self.strides, activation=None,
                             padding='same', kernel_regularizer=self.ker_reg)
        self.conv_2 = Conv2D(self.num_filters, (3, 3), activation=None, padding='same', kernel_regularizer=self.ker_reg)
        self.conv_3 = Conv2D(self.num_filters, (1, 1), strides=self.strides, activation=None,
                             padding='same', kernel_regularizer=self.ker_reg)
        self.activation = LeakyReLU()

    def call(self, inputs, training=False):
        x = self.conv_1(inputs)
        x = self.BatchNorm(x)
        x = self.activation(x)
        x = self.conv_2(x)
        if self.strides == 1:
            x_1 = inputs
        else:
            x_1 = self.conv_3(inputs)
        x = Add()([x, x_1])
        x = self.BatchNorm(x)
        output = self.activation(x)
        return output


class ResNet:
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, num_filters: int,
                 regularization: Optional[float], input_name: str, output_name: str):
        """
        Custom implementation of the lighted ResNet18 for cf synthesis task.
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
        Building CNN model for cf synthesis task.
        :return: Model() object.
        """
        inputs = Input(shape=self.input_shape, name=self.input_name)

        x = Conv2D(self.init_filters, (7, 7), strides=1, kernel_regularizer=self.ker_reg, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(x)

        for _ in range(2):
            x = ResidualBlock(num_filters=self.init_filters, regularization=self.ker_reg)(x)

        x = ResidualBlock(num_filters=self.init_filters * 2, regularization=self.ker_reg, stride=2)(x)
        x = ResidualBlock(num_filters=self.init_filters * 2, regularization=self.ker_reg)(x)

        x = ResidualBlock(num_filters=self.init_filters * 4, regularization=self.ker_reg, stride=2)(x)
        x = ResidualBlock(num_filters=self.init_filters * 4, regularization=self.ker_reg)(x)

        x = ResidualBlock(num_filters=self.init_filters * 8, regularization=self.ker_reg, stride=2)(x)
        x = ResidualBlock(num_filters=self.init_filters * 8, regularization=self.ker_reg)(x)

        x = GlobalAveragePooling2D()(x)
        x = Dense(self.num_classes, kernel_regularizer=self.ker_reg, activation='softmax', name=self.output_name)(x)

        return Model(inputs=inputs, outputs=x)