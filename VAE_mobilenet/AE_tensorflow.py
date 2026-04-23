import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import keras
from keras import Sequential, layers
from PIL import Image

tf.random.set_seed(22)
np.random.seed(22)

assert tf.__version__.startswith('2.')


# 合并多张图片
def save_images(imgs, name):
    new_im = Image.new('L', (280, 280))
    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1
    new_im.save(name)


h_dim = 32
batch_size = 10000
lr = 1e-3

# 一、获取数据集
(X_train, Y_train), (X_val, Y_val) = keras.datasets.fashion_mnist.load_data()

# 二、数据处理:处理训练集为batch模型(不需要目标值)
X_train = X_train.astype(np.float32) / 255.
db_train = tf.data.Dataset.from_tensor_slices(X_train)
db_train = db_train.shuffle(batch_size * 5).batch(batch_size)
X_val = X_val.astype(np.float32) / 255.
db_val = tf.data.Dataset.from_tensor_slices(X_val)
db_val = db_val.batch(batch_size)
print('X_train.shpae = {0}，Y_train.shpae = {1}，tf.reduce_max(Y_train) = {2}，tf.reduce_min(Y_train) = {3}'.format(X_train.shape, Y_train.shape, tf.reduce_max(Y_train), tf.reduce_min(Y_train)))
print('X_val.shpae = {0}，Y_val.shpae = {1}，tf.reduce_max(Y_val) = {2}，tf.reduce_min(Y_val) = {3}'.format(X_val.shape, Y_val.shape, tf.reduce_max(Y_val), tf.reduce_min(Y_val)))


# 三、创建AutoEncoder神经网络模型
class AutoEncoder(keras.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Encoders
        self.encoder = Sequential([
            layers.Dense(256, activation=tf.nn.relu),  # [b, 784] => [b, 256]
            layers.Dense(128, activation=tf.nn.relu),  # [b, 256] => [b, 128]
            layers.Dense(h_dim)  # [b, 128] => [b, 10]
        ])
        # Decoders
        self.decoder = Sequential([
            layers.Dense(128, activation=tf.nn.relu),  # [b, 10] => [b, 128]
            layers.Dense(256, activation=tf.nn.relu),  # [b, 128] => [b, 256]
            layers.Dense(784)  # [b, 256] => [b, 784]
        ])

    def call(self, inputs, training=None):
        # [b, 784] => [b, 10]
        h_out = self.encoder(inputs)
        # [b, 10] => [b, 784]
        x_hat = self.decoder(h_out)

        return x_hat


# 四、实例化AutoEncoder神经网络模型
model = AutoEncoder()
model.build(input_shape=(None, 28, 28, 1))
model.summary()
optimizer = tf.optimizers.Adam(learning_rate=lr)


# 五、训练模型：整体数据集进行一次梯度下降来更新模型参数，整体数据集迭代一次，一般用epoch。每个epoch中含有batch_step_no个step，每个step中样本的数量就是设置的每个batch所含有的样本数量。
def train_epoch(epoch_no):
    for batch_step_no, X_batch in enumerate(db_train):  # 每次计算一个batch的数据，循环结束则计算完毕整体数据的一次前向传播；每个batch的序号一般用step表示(batch_step_no)
        X_batch = tf.reshape(X_batch, [-1, 784])  # [b, 28, 28] => [b, 784]
        print('\tX_batch.shape = {0}'.format(X_batch.shape))
        with tf.GradientTape() as tape:
            X_batch_hat_logits = model(X_batch)
            print('\tX_batch_hat_logits.shape = {0}'.format(X_batch_hat_logits.shape))
            CrossEntropy_Loss = tf.losses.binary_crossentropy(X_batch, X_batch_hat_logits, from_logits=True)
            CrossEntropy_Loss = tf.reduce_mean(CrossEntropy_Loss)
        grads = tape.gradient(CrossEntropy_Loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print('\t第{0}个epoch-->第{1}个batch step的初始时的：CrossEntropy_Loss = {2}'.format(epoch_no, batch_step_no + 1, CrossEntropy_Loss))


# 六、模型评估 test/evluation
def evluation(epoch_no):
    X_batch = next(iter(db_val))
    X_batch_784 = tf.reshape(X_batch, [-1, 784])
    print('\tX_batch_784.shape = {0}'.format(X_batch_784.shape))
    X_batch_logits_hat = model(X_batch_784)    # model包含了encoder、decoder两大步骤
    X_batch_prob_hat = tf.sigmoid(X_batch_logits_hat)
    X_batch_prob_hat = tf.reshape(X_batch_prob_hat, [-1, 28, 28])  # [b, 784] => [b, 28, 28]
    print('\tX_batch_prob_hat.shape = {0}'.format(X_batch_prob_hat.shape))
    # X_batch_concat = tf.concat([X_batch, X_batch_prob_hat], axis=0)  # [b, 28, 28] => [2b, 28, 28]
    # print('\tX_batch_concat.shape = {0}'.format(X_batch_concat.shape))
    # X_batch_concat = X_batch_prob_hat
    X_batch_hat = X_batch_prob_hat.numpy() * 255.
    X_batch_hat = X_batch_hat.astype(np.uint8)
    save_images(X_batch_hat, 'AutoEncoder_images/rec_epoch_%d.png' % epoch_no)


# 六、整体数据迭代多次梯度下降来更新模型参数
def train():
    epoch_count = 10  # epoch_count为整体数据集迭代梯度下降次数
    for epoch_no in range(1, epoch_count + 1):
        print('\n\n利用整体数据集进行模型的第{0}轮Epoch迭代开始:**********************************************************************************************************************************'.format(epoch_no))
        train_epoch(epoch_no)
        evluation(epoch_no)
        print('利用整体数据集进行模型的第{0}轮Epoch迭代结束:**********************************************************************************************************************************'.format(epoch_no))


if __name__ == '__main__':
    train()
