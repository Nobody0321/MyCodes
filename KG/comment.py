import tensorflow as tf
import numpy as np
import os
import pickle
import text_helpers
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from tensorflow.python.framework import ops
ops.reset_default_graph()

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# 开始计算图会话
sess = tf.Session()

# 确定CROW模型参数，后面要自己训练word embedding
embedding_size = 200
vocabulary_size = 2000
batch_size = 100
max_words = 100

# 加载nltk库中的英文停顿词表
stops = stopwords.words('english')

# 载入数据
print('Loading Data')
data_folder_name = 'temp'
texts, target = text_helpers.load_movie_data()

# 使用text_helpers加载和转换文本数据集
print('Normalizing Text Data')
texts = text_helpers.normalize_text(texts, stops)

# 一句评论至少包含3个词
target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) > 2]
texts = [x for x in texts if len(x.split()) > 2]

# 将数据分为测试集和训练集
## 下面两步，随机选择target中80%的索引作为训练集标签的索引，剩下的20%作为测试集
train_indices = np.random.choice(len(target), round(0.8*len(target)), replace=False)  # round()函数，对浮点数四舍五入
test_indices = np.array(list(set(range(len(target))) - set(train_indices)))  # set() 函数创建一个无序不重复元素集

## 根据上一步选出的索引构造训练集和测试集
texts_train = [x for ix, x in enumerate(texts) if ix in train_indices]
texts_test = [x for ix, x in enumerate(texts) if ix in test_indices]
target_train = np.array([x for ix, x in enumerate(target) if ix in train_indices])
target_test = np.array([x for ix, x in enumerate(target) if ix in test_indices])

# 加载词典和Embedding矩阵
dict_file = os.path.join( '..', '05_Working_With_CBOW_Embeddings', 'temp', 'movie_vocab.pkl')
word_dictionary = pickle.load(open(dict_file, 'rb'))

# 通过字典将加载的句子转化为数值型numpy数组
text_data_train = np.array(text_helpers.text_to_numbers(texts_train, word_dictionary))
text_data_test = np.array(text_helpers.text_to_numbers(texts_test, word_dictionary))

# 由于影评长度不一样，规定一句影评为100个单词，不足用0填充
text_data_train = np.array([x[0:max_words] for x in [y+[0]*max_words for y in text_data_train]])
text_data_test = np.array([x[0:max_words] for x in [y+[0]*max_words for y in text_data_test]])

print('Creating Model')

# Embedding层（Word2Vec相关，其实tf中可以直接调用google开发的Word2Vec库）
# 初始化
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))


# 定义Embedding层模型:

# 声明逻辑回归的模型变量和占位符
W = tf.Variable(tf.random_normal(shape=[embedding_size,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))
x_data = tf.placeholder(shape=[None, max_words], dtype=tf.int32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 在计算图中  假如嵌套查找操作。计算句子中所有单词的平均嵌套(?)
embed = tf.nn.embedding_lookup(embeddings, x_data)
embed_avg = tf.reduce_mean(embed, 1)  # 求第二维（行）的平均值

# 声明模型操作和损失函数
model_output = tf.add(tf.matmul(embed_avg, W), b)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))

# 预测函数和准确度函数
prediction = tf.round(tf.sigmoid(model_output))
predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)  # 预测正确为1，错误为0
accuracy = tf.reduce_mean(predictions_correct)  # 总体正确率

# 定义优化器和学习速率
my_opt = tf.train.AdagradOptimizer(0.005)
train_step = my_opt.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()
sess.run(init)

# 随机初始化单词嵌套，导入CBOW模型
model_checkpoint_path = os.path.join( '..', '05_Working_With_CBOW_Embeddings',
                                      'temp','cbow_movie_embeddings.ckpt')
saver = tf.train.Saver({"embeddings": embeddings})
saver.restore(sess, model_checkpoint_path)


# 开始训练，每迭代100次保存训练集和测试集的损失和准确度
# 每500次打印一次模型状态
print('Start Model Training')
train_loss = []
test_loss = []
train_acc = []
test_acc = []
i_data = []
for i in range(10000):
    # 通过下标来取数据
    rand_index = np.random.choice(text_data_train.shape[0], size=batch_size)
    rand_x = text_data_train[rand_index]
    rand_y = np.transpose([target_train[rand_index]])

    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y}) # feed_dict给placeholder赋值

    # record loss and accuracy every 100 rounds
    if (i+1)%100==0:
        i_data.append(i+1)
        train_loss_temp = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        train_loss.append(train_loss_temp)

        test_loss_temp = sess.run(loss, feed_dict={x_data: text_data_test, y_target: np.transpose([target_test])})
        test_loss.append(test_loss_temp)

        train_acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y})
        train_acc.append(train_acc_temp)

        test_acc_temp = sess.run(accuracy, feed_dict={x_data: text_data_test, y_target: np.transpose([target_test])})
        test_acc.append(test_acc_temp)

    if (i+1)%500==0:
        acc_and_loss = [i+1, train_loss_temp, test_loss_temp, train_acc_temp, test_acc_temp]
        acc_and_loss = [np.round(x,2) for x in acc_and_loss]
        print('Generation # {}. Train Loss (Test Loss): {:.2f} ({:.2f}). Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))


# 绘制损失函数
plt.plot(i_data, train_loss, 'k-', label='Train Loss')
plt.plot(i_data, test_loss, 'r--', label='Test Loss', linewidth=4)
plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.legend(loc='upper right')
plt.show()

# 绘制训练和测试函数
plt.plot(i_data, train_acc, 'k-', label='Train Set Accuracy')
plt.plot(i_data, test_acc, 'r--', label='Test Set Accuracy', linewidth=4)
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()



