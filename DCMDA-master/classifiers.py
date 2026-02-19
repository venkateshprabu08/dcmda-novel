import tensorflow as tf
from tensorflow.keras import layers
from clr import cyclic_learning_rate
import numpy as np
import random
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


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


# Novelty: Multi-Head Cross-Attention layer for improved feature fusion
class MultiHeadCrossAttention(layers.Layer):
    """Multi-Head Cross-Attention for fusing GAE and NMF features.

    Uses multiple attention heads to capture diverse interaction patterns
    between the two feature streams, improving robustness over single-head attention.
    """

    def __init__(self, num_heads=4, key_dim=16, **kwargs):
        super(MultiHeadCrossAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim

    def build(self, input_shape):
        embed_dim = input_shape[0][-1]
        self.head_dim = self.key_dim
        self.W_q = []
        self.W_k = []
        self.W_v = []
        for _ in range(self.num_heads):
            self.W_q.append(self.add_weight(
                shape=(embed_dim, self.head_dim),
                initializer='glorot_uniform', trainable=True))
            self.W_k.append(self.add_weight(
                shape=(embed_dim, self.head_dim),
                initializer='glorot_uniform', trainable=True))
            self.W_v.append(self.add_weight(
                shape=(embed_dim, self.head_dim),
                initializer='glorot_uniform', trainable=True))
        self.W_o = self.add_weight(
            shape=(self.num_heads * self.head_dim, embed_dim),
            initializer='glorot_uniform', trainable=True)
        super(MultiHeadCrossAttention, self).build(input_shape)

    def call(self, inputs):
        query, key_value = inputs[0], inputs[1]
        head_outputs = []
        for i in range(self.num_heads):
            Q = tf.matmul(query, self.W_q[i])
            K = tf.matmul(key_value, self.W_k[i])
            V = tf.matmul(key_value, self.W_v[i])
            scale = tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
            attn_scores = tf.matmul(Q, K, transpose_b=True) / scale
            attn_weights = tf.nn.softmax(attn_scores, axis=-1)
            head_out = tf.matmul(attn_weights, V)
            head_outputs.append(head_out)
        concat = tf.concat(head_outputs, axis=-1)
        output = tf.matmul(concat, self.W_o)
        return output

    def get_config(self):
        config = super(MultiHeadCrossAttention, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
        })
        return config


def BuildModel(train_gae, train_nmf, label):

    l1 = len(train_gae[1])
    l2 = len(train_nmf[1])
    inputs_gae = Input(shape=(l1,))
    inputs_nmf = Input(shape=(l2,))

    # Novelty: Added BatchNormalization and Dropout for regularization
    x = Dense(128, activation='relu')(inputs_gae)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)

    y = Dense(128, activation='relu')(inputs_nmf)
    y = BatchNormalization()(y)
    y = Dropout(0.3)(y)
    y = Dense(64, activation='relu')(y)

    # Novelty: Multi-Head Cross-Attention replaces single-head attention
    x_reshaped = layers.Reshape((1, 64))(x)
    y_reshaped = layers.Reshape((1, 64))(y)
    cross_attn = MultiHeadCrossAttention(num_heads=4, key_dim=16)
    result = cross_attn([y_reshaped, x_reshaped])
    result = layers.Flatten()(result)

    # Novelty: Additional regularization before output
    result = Dropout(0.2)(result)
    predictions = Dense(1, activation='sigmoid')(result)

    model = Model(inputs=[inputs_gae, inputs_nmf], outputs=predictions)

    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Novelty: EarlyStopping to prevent overfitting
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit([train_gae, train_nmf], label,
              epochs=80, callbacks=[early_stop])
    return model