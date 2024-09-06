from tensorflow.keras.models import Model, Sequential, clone_model, load_model
from tensorflow.keras.layers import Input, Dense, add, concatenate, Conv2D,Dropout,\
BatchNormalization, Flatten, MaxPooling2D, AveragePooling2D, Activation, Dropout, Reshape
import tensorflow as tf


def cnn_3layer_fc_model(n_classes,n1 = 16, n2=16, n3=32,n4=32, dropout_rate = 0.2,input_shape = (144,32)):#神经网络函数定义；n_classes是分类的类数，光刻热区蒸馏只用了这个模型
    model_A, x = None, None
     
    x = Input(input_shape)#输入数据的形状(144,32)，32个通道
    if len(input_shape)==2:
        y = Reshape((12, 12, 32))(x)#输入图片的形状(144,32)——>(12,12,32)
    
    y = Conv2D(filters = n1, kernel_size = (3,3), strides = 1, padding = "same", #第一次卷积操作
            activation = None)(y)
    y = BatchNormalization()(y)#防止过拟合
    y = Activation("relu")(y)#激活操作
    y = Dropout(dropout_rate)(y)#防止过拟合

    y = Conv2D(filters = n2, kernel_size = (3,3), strides = 1, padding = "same", #第2次卷积操作
            activation = None)(y)
    y = BatchNormalization()(y)#防止过拟合
    y = Activation("relu")(y)#激活操作
    y = Dropout(dropout_rate)(y)#防止过拟合
    y = AveragePooling2D(pool_size = (2,2), strides = 1, padding = "same")(y)#池化

    y = Conv2D(filters = n3, kernel_size = (3,3), strides = 1, padding = "same", #第3次卷积操作
            activation = None)(y)
    y = BatchNormalization()(y)#防止过拟合
    y = Activation("relu")(y)#激活操作
    y = Dropout(dropout_rate)(y)#防止过拟合

    y = Conv2D(filters = n4, kernel_size = (2,2), strides = 2, padding = "valid", #第4次卷积操作
            activation = None)(y)
    y = BatchNormalization()(y)#防止过拟合
    y = Activation("relu")(y)#激活操作
    y = Dropout(dropout_rate)(y)#防止过拟合
    y = AveragePooling2D(pool_size = (2,2), strides = 2, padding = "valid")(y)#池化


    y = Flatten()(y)
    y = Dense(units=250, activation=tf.nn.relu)(y)
    y = Dense(units = n_classes, activation = None, use_bias = False,
            kernel_regularizer=tf.keras.regularizers.l2(1e-3))(y)
    y = Activation("softmax")(y)


    model_A = Model(inputs = x, outputs = y)

    model_A.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-3), 
                        loss = "sparse_categorical_crossentropy",
                        metrics = ["accuracy"])
    return model_A#最后返回一个模型
  
def cnn_2layer_fc_model(n_classes,n1 = 128, n2=256, dropout_rate = 0.2,input_shape = (28,28)):#神经网络函数定义；n_classes是分类的类数
    model_A, x = None, None
    
    x = Input(input_shape)#输入数据的形状(28,28)
    if len(input_shape)==2: y = Reshape((input_shape[0], input_shape[1], 1))(x)##输入图片的形状(28,28)——>(28,28,1)
    
    y = Conv2D(filters = n1, kernel_size = (3,3), strides = 1, padding = "same", 
            activation = None)(y) #卷积操作
    y = BatchNormalization()(y)  #防止过拟合
    y = Activation("relu")(y)#激活操作
    y = Dropout(dropout_rate)(y)#防止过拟合
    y = AveragePooling2D(pool_size = (2,2), strides = 1, padding = "same")(y)#池化


    y = Conv2D(filters = n2, kernel_size = (3,3), strides = 2, padding = "valid", 
            activation = None)(y)#卷积操作
    y = BatchNormalization()(y)#防止过拟合
    y = Activation("relu")(y)#激活操作
    y = Dropout(dropout_rate)(y)#防止过拟合
    #y = AveragePooling2D(pool_size = (2,2), strides = 2, padding = "valid")(y)

    y = Flatten()(y)
    y = Dense(units = n_classes, activation = None, use_bias = False,
            kernel_regularizer=tf.keras.regularizers.l2(1e-3))(y)
    y = Activation("softmax")(y)


    model_A = Model(inputs = x, outputs = y)

    model_A.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-3), 
                        loss = "sparse_categorical_crossentropy",
                        metrics = ["accuracy"])
    return model_A #最后返回一个模型


def remove_last_layer(model, loss = "mean_absolute_error"):
    """
    Input: Keras model, a classification model whose last layer is a softmax activation
    Output: Keras model, the same model with the last softmax activation layer removed,
        while keeping the same parameters
        输入：Keras 模型，最后一层是 softmax 激活的分类模型
     输出：Keras 模型，删除最后一个 softmax 激活层的相同模型，
         同时保持相同的参数
    """
    
    new_model = Model(inputs = model.inputs, outputs = model.layers[-2].output)
    new_model.set_weights(model.get_weights())
    new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-3), 
                      loss = loss)
    
    return new_model