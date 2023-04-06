#f0rom tensorflow.keras import BatchNormalizaiotion
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Model, Sequential
from keras import layers, optimizers, losses
import keras
#VGGNET
class Model():
    def VGGnet_model(targetSize):
        VGGNET_model=Sequential([
            #first
            layers.Conv2D(input_shape = (targetSize[0], targetSize[1], 3), filters = 64, kernel_size = (3, 3), padding = "same", activation = "relu"),
            layers.Conv2D(filters = 64,kernel_size = (3, 3), padding = "same", activation = "relu"),
            layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2)),
            #second
            layers.Conv2D(filters = 128, kernel_size = (3, 3), padding = "same", activation = "relu"),
            layers.Conv2D(filters = 128, kernel_size = (3, 3), padding = "same", activation = "relu"),
            layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2)),
            #third
            layers.Conv2D(filters = 256, kernel_size = (3, 3), padding = "same", activation = "relu"),
            layers.Conv2D(filters = 256, kernel_size = (3, 3), padding = "same", activation = "relu"),
            layers.Conv2D(filters = 256, kernel_size = (3, 3), padding = "same", activation = "relu"),
            layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2)),
            #fourth
            layers.Conv2D(filters = 512, kernel_size = (3, 3), padding = "same", activation = "relu"),
            layers.Conv2D(filters = 512, kernel_size = (3, 3), padding = "same", activation = "relu"),
            layers.Conv2D(filters = 512, kernel_size =(3, 3), padding = "same", activation = "relu"),
            layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2)),
            #fifth
            layers.Conv2D(filters = 512, kernel_size = (3, 3), padding = "same", activation = "relu"),
            layers.Conv2D(filters = 512, kernel_size = (3, 3), padding = "same", activation = "relu"),
            layers.Conv2D(filters = 512, kernel_size = (3, 3), padding = "same", activation = "relu"),
            layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2)),
            #FC
            layers.Flatten(),
            layers.Dense(units = 4096, activation = "relu"),
            layers.Dense(units = 4096, activation = "relu"),
            layers.Dense(units = 4, activation = "softmax"),
        ])
        VGGNET_model.compile(optimizer = optimizers.Adam(learning_rate = 0.0001), loss = losses.categorical_crossentropy, metrics = ['accuracy'])
        return VGGNET_model