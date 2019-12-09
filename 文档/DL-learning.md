[toc]
## 交叉熵损失
对于离散分布p和q，交叉熵形势如下：
$$H(p,q)= -\sum_{x}p(x)\log q(x))$$
其中p是数据中的真实分布，q是我们模型预测的分布
在机器学习中，由于真实分布是未知的，所以我们只能估计交叉熵，通过下式：
$$H(T,q)=-\sum_{i=1}^{N}\frac{1}{N}\log_2q(x_i)$$
其中N是数据容量（测试集大小），q(x)是在训练集上估计的（预测的）事件x的发生概率。


## Batch Normalization
### 先导问题： 
IDD独立同分布假设：假设训练数据和测试数据的满足相同分布。
Internal Covariate Shift问题就是描述在神经网络训练过程中，自变量的分布总是变化的问题。
在网络的前向传播中，随着网络深度的增加，因为各层参数的不断变化，每个隐层都会改变自变量的分布。（每一层输入的自变量是同分布，层之间不是同分布）

### 主要思想
神经网络之所以训练慢，收敛慢，一般是因为分布会逐渐向非线性函数定义域的两端靠近。对于sigmoid激活函数来说，就是激活值会离原点较远。
而根据激活函数的性质，这些区域的梯度也较小，从而导致了反向传播的梯度消失或者梯度弥散。
所以我们引入batchNorm机制，对每一隐层的输入都进行**正则化**，把不同分布的隐层输入拉回到同一标准正态分布。（受启发于图像处理，在图像处理中，对输入图像进行白化，总是能加速神经网络的收敛）。使得非线性的激活函数的输入值重新落入梯度较大的区域，以此避免了梯度消失问题，也使得网络训练速度加快。

BN之后，大部分激活值落入非线性函数的线性区间（对比标准正太分布和sigmoid可知，95%的数据在[-2,2]区间，这个区域sigmoid是接近线性的），因此降低了网络的表达能力（*纯线性的函数无法表示复杂的非线性函数*）。

为了维持网络的表达能力，BN对变换后的满足均值为0，方差为1的x又进行了一次线性操作
$$y=scale*x+shift$$
把标准正态分布又偏移扩增，希望能找到一个线性和非线性较好的平衡点，做到既能非线性表达，又能靠近线性区，使得网络收敛足够快。
shift和scale参数通过网络训练得到。

### 具体实践
在训练过程中，bn可以根据mini batch的若干实例计算均值和方差，然后对x进行白化。
在推理过程中，一次输入只有一个instance，无法求出均值和方差。所以需要在训练的时候，保存每一层所有mini batch 对应的均值和标准层，然后求出整体的期望和方差，用于推理过程。

### 优点
1. 极大提升了训练速度，收敛过程大大加快；
2. 增加分类效果，一种解释是这是类似于Dropout的一种防止过拟合的正则化表达方式，所以不用Dropout也能达到相当的效果；
3. 简化了调参过程，对于初始化要求没那么高，而且可以使用大的学习率等。



## 反向传播

#### 1. DNN 的反向传播
我们使用最常见的均方差作为损失函数，那么网络最后的输出为
$$J(W,b,x,y)=\frac{1}{2}||a^L−y||_2^2$$
其中，$a^L$和$y$是特征维度为$d_{out}$的向量，$||S||_2$是s的l2范数

首先是输出的第L层，我们注意到在DNN中

$$a^L=\sigma(z^L)=\sigma(W^L a^{L-1}+b^L)$$
代入损失函数可得

$$J(W,b,x,y)=\frac{1}{2}||a^L−y||_2^2=\frac{1}{2}||\sigma(W^L a^{L-1}+b^L)-y||_2^2 = \frac{1}{2}\sum_i^n [\sigma(W_i^L a_i^{L-1}+b_i^L)-y]^2$$

求导可得
$$
\frac{\partial{J(W,b,x,y)}}{\partial W^L}=
\frac{\partial{J(W,b,x,y)}}{\partial a^L} \frac{\partial a^L}{\partial z^L}\frac{\partial z^L}{\partial W^L}     \\
=\sum_i^n (a_i^L-y) \sigma^′(z^L) a_i^{L-1}     \\
=[(a^L−y)\odotσ^′(z^L)](a^{L−1})^\mathrm{T}
$$
同理可得 
$$\frac{\partial{J(W,b,x,y)}}{\partial {b^L}}=(a^L−y)\odot σ^′(z^L)$$

由此可得最后一层的梯度，可以推导出中间的梯度
记第$L$层损失函数对$z^L$的导数为 $\delta^L$
$$\delta^L=\frac{\partial{J(W,b,x,y)}}{\partial z^L}
=(a^L−y)\odotσ^′(z^L)$$
对第$l$层，这里的损失函数对于未激活输出$z^l$的梯度是
$$\delta^l=\frac{\partial J(W,b,x,y)}{\partial z^l}=
(\frac{\partial z^L}{\partial z^{L-1}}\frac{\partial z^{L-1}}{\partial z^{L-2}}···\frac{\partial z^{l+1}}{\partial z^{l}})^ {\mathrm{T}} \delta^L$$
由上式及$z^l= W^la^{l-1} + b^l$，可以推出第$l$层的$W^l,b^l$的梯度:
$$\frac{\partial J(W,b,x,y)}{\partial W^l} = \delta^{l}(a^{l-1})^T$$

$$\frac{\partial J(W,b,x,y)}{\partial b^l} = \delta^{l}$$
## 滑动平均
###定义
滑动平均也叫做指数加权平均，用来估计变量的局部均值，使得变量的更新于一段时间内的历史取值有关。
变量v在t时刻记为$v_t$，v在t时刻的真实值是$\theta_t$，使用滑动模型对v_t更新如下：
$$v_t=\beta\cdot v_{t-1}+(1-\beta)\cdot \theta_t$$
式中$\beta\in[0,1)$

t时刻变量v的滑动平均大致等于过去$\frac{1}{1-\beta}$ 个时刻真值$\theta$的平均。偏差在滑动平均起始时候较大，所以引入$\frac{1}{1-\beta^t}$因子修正偏差。

$$vt=β⋅vt−1+(1−β)⋅θt$$
$$v\_biased_t=\frac{v_t}{1-\beta^t}$$

###优点
占内存少，有点像lstm，一直在传递历史平均值，而不需要额外空间保存所有历史值

##矩阵求导
###向量对向量求导的链式法则
设多个向量存在依赖关系，比如三个向量$x\to y\to z$存在依赖关系，则我们有下面的链式求导法则：
$$\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = \frac{\partial \mathbf{z}}{\partial \mathbf{y}}\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$$
假设x,y,z分别是m,n,p维向量,则 $\frac{\partial \mathbf{z}}{\partial \mathbf{y}}$结果是一个$p\times n$矩阵，每一行是[$\frac{\partial \mathbf{z_i}}{\partial \mathbf{y_{1-m}}}$]；
同理可得$\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$是一个$n\times m$矩阵，
因此求导结果$\frac{\partial \mathbf{z}}{\partial \mathbf{x}}$是一个$p\times m$的雅各比矩阵。

###标量对向量的链式求导法则
对于依赖关系$\mathbf{x}\to \mathbf{y}\to z$，向上一节一样进行链式求导，就会发现维度不相容，$\frac{\partial z}{\partial \mathbf{x}}$结果是一个$m\times 1$的向量，而$\frac{\partial z}{\partial \mathbf{y}}$是一个$n\times 1$的向量，$\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$是一个$n\times m$的雅克比矩阵,右边的向量和矩阵是没法直接乘的.

如果是标量对更多的向量求导,比如$\mathbf{y_1}\to \mathbf{y_2}\to ...\to \mathbf{y_n}\to z$，
则其链式求导表达式可以表示为：
$$\frac{\partial z}{\partial \mathbf{y_1}} = (\frac{\partial \mathbf{y_n}}{\partial \mathbf{y_{n-1}}} \frac{\partial \mathbf{y_{n-1}}}{\partial \mathbf{y_{n-2}}} ...\frac{\partial \mathbf{y_2}}{\partial \mathbf{y_1}})^T\frac{\partial z}{\partial \mathbf{y_n}}$$

##TODO
###RELU
todo
### residual connection
todo

### dropout
dropout 层服从概率为p的伯努利分布
每一个位置上的数字以p的概率为1，以1-p的概率为0

dropout层通过随机地将参数置0，避免了两个有相关性的神经元总是一起变化。（这种技术减少了神经元之间复杂的共适性。因为一个神经元不能依赖其他特定的神经元。因此，不得不去学习随机子集神经元间的鲁棒性的有用连接。）

每次前向传播都是一个不同的模型，这也使得每次优化的目标都不完全相同，相当于分别训练多个不同模型，得到这些模型的平均输出。dropout处理很像是平均一个大量不同网络的平均结果。不同的网络在不同的情况下过拟合。因此，很大程度上。dropout将会减少这种过拟合。

由于训练时每次只保留了 n*(1-p) 个神经元，这样就使得网络的平均输出是未dropout网络的 (1-p) 倍。
也就是说训练和测试的网络输出是两个分布。
在推理/测试的时候，我们就需要将输出同样扩增为1/（1-p）倍。

现在主流框架的实现是在训练时增大梯度，在推理/测试的时候就不用管了。