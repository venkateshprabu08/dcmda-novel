import tensorflow as tf
from tensorflow.keras import layers
from clr import cyclic_learning_rate
import numpy as np
import random
from keras.layers import Dense, Input, dot
from keras.models import Model
from tensorflow.keras.optimizers import Adam

def get_all_samples(conjunction):
    pos = []
    neg = []
    for index in range(conjunction.shape[0]):
        for col in range(conjunction.shape[1]):
            if conjunction[index, col] == 1:
                pos.append([index, col, 1])
            else:
                neg.append([index, col, 0])
    pos_len = len(pos)
    new_neg = random.sample(neg, pos_len)
    samples = pos + new_neg
    samples = random.sample(samples, len(samples))
    samples = np.array(samples)
    return samples



#def BuildModel(train_gae,train_nmf,label):
#
#    l1 = len(train_gae[1])
#    l2=len(train_nmf[1])
#    inputs_gae = Input(shape=(l1,))
#    inputs_nmf = Input(shape=(l2,))
#
#    x = Dense(128, activation='relu')(inputs_gae)
#    x = Dense(64, activation='relu')(x)
#    y = Dense(128, activation='relu')(inputs_nmf)
#    y = Dense(64, activation='relu')(y)
#    attention = layers.Attention()
#    result = attention([y, x])
#    predictions = Dense(1, activation='sigmoid')(result)
#
#    model = Model(inputs=[inputs_gae,inputs_nmf], outputs=predictions)
#    model.compile(optimizer='rmsprop',
#                  loss='binary_crossentropy',
#                  metrics=['accuracy'])
#    model.fit([train_gae, train_nmf], label)
#    return model


# 然后在 Dense 层中使用这个自定义激活函数

def BuildModel(train_gae,train_nmf,label):

    l1 = len(train_gae[1])
    l2=len(train_nmf[1])
    inputs_gae = Input(shape=(l1,))
    inputs_nmf = Input(shape=(l2,))

    x = Dense(128, activation='relu')(inputs_gae)
    x = Dense(64, activation='relu')(x)
    y = Dense(128, activation='relu')(inputs_nmf)
    y = Dense(64, activation='relu')(y)
    attention = layers.Attention()
    result = attention([y,x])
#    predictions = Dense(1, activation='sigmoid')(result)
    predictions =Dense(1, activation='sigmoid')(result)


    # 创建模型
    model = Model(inputs=[inputs_gae, inputs_nmf], outputs=predictions)

    # 编译模型并设置学习率
    optimizer = Adam(learning_rate=0.0005)  # 设置学习率
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # 训练模型并设置 epoch 数
    model.fit([train_gae, train_nmf], label,
              epochs=80)  # 可选：设置批次大小l)
    return model