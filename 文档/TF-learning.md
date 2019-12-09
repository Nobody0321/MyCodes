[toc]
# TensorFlow 常用函数
## tf.shape()
    tf.shape(
        input,
        name=None,
        out_type=tf.int32
    )
是一个Tensor instance
输出input矩阵的维度矩阵，返回一个tensor

## tensor.shape
variable自带*属性*，用于输出一个variable的shape，返回值类型是TensorShape

## tensor.get_shape()
返回一个tuple

## tf.reshape()
    tf.reshape(
        tensor,
        shape,
        name=None
    )
对输入的tensor进行维度重排，返回重排后新的tensor, -1是默认值


## tf.nn.bidirectional_dynamic_rnn()
    tf.nn.bidirectional_dynamic_rnn(
        cell_fw,
        cell_bw,
        inputs,
        sequence_length=None,
        initial_state_fw=None,
        initial_state_bw=None,
        dtype=None,
        parallel_iterations=None,
        swap_memory=False,
        time_major=False,
        scope=None
    )

返回值为tuple (outputs, output_states)，

其中outputs是(output_fw, output_bw) 包含了前后向lstm的所有状态的tuple，

output_fw/bw 形如 [batch_size, max_time, cell_fw/bw.output_size]

而output_states 包含了输出时前后向tuple的最后一个隐层状态


## TF初始化
在TensorFlow的世界里，变量的定义和初始化是分开的，所有关于图变量的赋值和计算都要通过tf.Session的run来进行。

想要将所有图变量进行集体初始化时应该使用tf.global_variables_initializer。

## tf.py_func()
    tf.py_func(
        func,
        inp,
        Tout,
        stateful=True,
        name=None
    )
用途：将一个参数为nparray，返回值为nparray的函数func包装成一个tensorflow操作.

首先，tf.py_func接收的是tensor，然后将其转化为nparray送入func函数，最后再将func函数输出的numpy array转化为tensor返回

func:一个自定义的函数，输入是nparray，输出也是nparray，在自定义的函数func内部，可以使用np自由地进行nparray操作

inp: 标志func函数接收输入的一个list

Tout：指定了*func*函数返回的nparray转化成tensor后的*格式*，如果是返回多个值，就是一个列表或元组；如果只有一个返回值，就是一个单独的dtype类型(当然也可以用列表括起来)。

返回值： 输出是一个tensor列表或单个tensor。


**func是脱离Graph的。在func中不能定义可训练的参数参与网络训练(反传)**

## tf.convert_to_tensor()

## tf.gather()
可以把向量中某些索引值提取出来，得到新的向量，适用于要提取的索引为不连续的情况。这个函数似乎只适合在一维的情况下使用

## tf.where()
返回一个boolean tensor中true值的位置（索引）
常用一些条件判断函数tf.equal(),tf.not_equal(),tf.greater()连用，用于提取矩阵中特定值的
```
label = tf.Variable([1,0,1])
a_keep = tf.equal(label, 1)
print(tf.where(a_keep).eval())
>>>[[0]
 [2]]
```

## tf.reduce_sum(), tf.reduce_mean()
对一个多维矩阵的所有元素相加求和/求平均数


## tf.random_normal()
tf.random_normal(shape,mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)
random_normal: 正态分布随机数，均值mean,标准差stddev

## tf.nn.softmax()
对每一个行向量进行softmax

维度相同

##tf.control_dependencies()


## 在tf中使用BN层

# Numpy 常用函数
## np.stack()
将一个nparray的list（N个） 合成一个nparray（1*N，增加一维，），前提是list中所有array维度一致

## np.vstack()
将两个nparray 在垂直方向（veriacl）上 上下排列

## np.hstack()
将两个nparray 在水平方向（horizonal）上左右排列

## np.meshgrid(x_list,y_list)
根据给定的x，y坐标，延伸出平行于y轴、x轴的直线，组成一个网格，返回这些直线的所有交点坐标（x坐标y坐标分组）返回X，Y

## np.ascontiguousarray()
返回一个在内存中以c order排序的数组/矩阵
### 额外知识： C order vs Fortran order
所谓C order，指的是行优先的顺序（Row-major Order)，即内存中同行的元素存在一起，而Fortran Order则指的是列优先的顺序（Column-major Order)，即内存中同列的元素存在一起。Pascal, C，C++，Python都是行优先存储的，而Fortran，MatLab是列优先存储的。

## 对numpy中axis的理解
例如
a = np.array([[1,2,3,4,5],[6,7,8,9,0]])

print(a.argmax(axis = 0))  # output : [1 1 1 1 0]
print(a.argmax(axis = 1))  # output: [4 3]
axis 相当于在哪个维度上变化，在行上变化取argmax就是取每一列最大的arg的索引

另一种理解：
axis相当于定义返回的是行向量还是列向量，返回行向量的话，行中每一个数字都是该列最大数字的行号
返回列向量的话，列中每一个数字都是该行最大数字的列号

## np.polyfit()
np.polyfit(x, y, deg, rcond=None, full=False, sw=None, cov=False)
根据xy坐标，和需要拟合几次曲线。


## np.polyfit1d()
根据一组系数[w_0, w_1, w_2, ...,_n]，组成曲线的表表达式
