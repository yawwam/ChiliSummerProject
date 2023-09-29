import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, MaxPool2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input, InputLayer
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras.regularizers import l2

def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1, reg=False):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu",kernel_initializer = "he_normal")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu",kernel_initializer = "he_normal")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    if reg : 
        p = Dropout(0.3)(p)
    return c, p
    
def upsample_block(x, conv_features, filters, kernel_size=(3,3),padding="same",strides=1, reg=False):
    # upsample
    x = keras.layers.Conv2DTranspose(filters, kernel_size,padding=padding)(x)
    
    conv_features = keras.layers.Conv2D(filters, (1, 1), padding=padding, activation="relu")(conv_features)
    # concatenate
    x = keras.layers.concatenate([x, conv_features])
    # dropout
    if reg :
        x = Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu",kernel_initializer = "he_normal")(x)
    x = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu",kernel_initializer = "he_normal")(x)
    return x
   
def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1, reg=False):
    c1 = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    if reg :
        c1 = Dropout(0.3)(c1)
        c1 = BatchNormalization()(c1)
    c2 = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c1)
    return c1, c2
    

def create_skeleton_gesture_reg2dmodel(height, width, depth, filters=(64, 128, 256, 512, 1024)) :
    inputShape = (height, width, depth)

    chanDim = -1
    inputs = Input(shape=inputShape)

    p0 = inputs
    _, p1 = down_block(p0, filters[0],reg=True)
    _, p2 = down_block(p1, filters[1],reg=True)
    _, p3 = down_block(p2, filters[2],reg=True)
    _, p4 = down_block(p3, filters[3],reg=True)
    bn1, _ = bottleneck(p4, filters[4],reg=True)

    x = Flatten()(bn1)
    x = Dense(256,kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Activation("relu")(x)
    fc = Dense(128, activation='relu',kernel_regularizer=l2(0.001))(x)
    fc = Dropout(0.3)(fc)

    x_gesture = Dense(4, activation='softmax', name='gesture')(fc)
    x_skeleton = Dense(42, activation="linear", name='skeleton')(fc)
    concat_output = tf.keras.layers.Concatenate(axis=-1,trainable=False)([x_skeleton,x_gesture])
    model = Model(inputs=inputs, outputs=[concat_output])
    return model
    
def create_skeleton_gesture_reg3dmodel(height, width, depth, filters=(64, 128, 256, 512, 1024)) :
    inputShape = (height, width, depth)

    chanDim = -1
    inputs = Input(shape=inputShape)

    p0 = inputs
    _, p1 = down_block(p0, filters[0],reg=True)
    _, p2 = down_block(p1, filters[1],reg=True)
    _, p3 = down_block(p2, filters[2],reg=True)
    _, p4 = down_block(p3, filters[3],reg=True)
    _, bn2 = bottleneck(p3, filters[4],reg=True)

    x = Flatten()(bn2)
    x = Dense(256, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)
    fc = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    fc = Dropout(0.3)(fc)

    x_gesture = Dense(4, activation='softmax', name='gesture')(fc)
    x_skeleton = Dense(63, activation="linear", name='skeleton')(fc)
    concat_output = tf.keras.layers.Concatenate(axis=-1,trainable=False)([x_skeleton,x_gesture])
    model = Model(inputs=inputs, outputs=[concat_output])
    return model
    

def create_skeleton_gesture_3dmodel(height, width, depth, filters=(32, 64, 128, 256)) :
    inputShape = (height, width, depth)

    chanDim = -1
    inputs = Input(shape=inputShape)

    p0 = inputs
    _, p1 = down_block(p0, filters[0])
    _, p2 = down_block(p1, filters[1])
    _, p3 = down_block(p2, filters[2])
    bn1, _ = bottleneck(p3, filters[3])

    x = Flatten()(bn1)
    x = Dense(256)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)
    fc = Dense(128, activation='relu')(x)

    x_gesture = Dense(4, activation='softmax', name='gesture')(fc)
    x_skeleton = Dense(63, activation="linear", name='skeleton')(fc)
    concat_output = tf.keras.layers.Concatenate(axis=-1,trainable=False)([x_skeleton,x_gesture])
    model = Model(inputs=inputs, outputs=[concat_output])
    return model

def create_skeleton_gesture_2dmodel(height, width, depth, filters=(32, 64, 128, 256)) :
    inputShape = (height, width, depth)

    chanDim = -1
    inputs = Input(shape=inputShape)

    p0 = inputs
    _, p1 = down_block(p0, filters[0])
    _, p2 = down_block(p1, filters[1])
    _, p3 = down_block(p2, filters[2])
    bn1, _ = bottleneck(p3, filters[3])

    x = Flatten()(bn1)
    x = Dense(256)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)
    fc = Dense(128, activation='relu')(x)

    x_gesture = Dense(4, activation='softmax', name='gesture')(fc)
    x_skeleton = Dense(42, activation="linear", name='skeleton')(fc)
    concat_output = tf.keras.layers.Concatenate(axis=-1,trainable=False)([x_skeleton,x_gesture])
    model = Model(inputs=inputs, outputs=[concat_output])
    return model

