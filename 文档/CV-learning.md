# CV学习笔记(OCR)
主要记录我在亚马逊实习期间做的ocr项目的学习笔记


## CTPN的训练过程
1. 人工标注图片文字，准备训练集，就有了GT
2. 训练网络：在三个方向上训练网络

    2.1 对文字部分的识别能力（文本框的可信度），这部分主要由VGG+卷积+BiLstm完成

    2.2 减少文本框的垂直偏移，这也就是文中基于anchor mechanism的bbox regression完成的

    2.3 文本框连接成文本线（text line），并减少连接后的文本线在水平防线的偏移，这又是由论文的side-refinement部分完成

也因此，论文的Loss Function如下：
$$L(s_i,v_j,o_k)=\frac{1}{N_s}\sum_i L_{s}^{cl}(s_i,s_i^*)+\frac{\lambda_1}{N_v}\sum_jL_{v}^{re}(v_j,v_j^*)+\frac{\lambda_2}{N_o}\sum_kL_{o}^{re}(o_k,o_k^*)$$

loss由三部分组成，第一部分是交叉熵，其中$s$代表一个anchor的网络预测输出，$s^*$代表anchor的真是标签；

第二部分是anchor竖直方向偏移的回归损失，使用smooth L1 Loss， 其中$v=[t_y, t_h]$代表网络预测的文本框垂直偏差，v^*代表真实偏差；

*这里运用了统计学的思想，既然我们大部分原始的预测总是有偏差的，那么就运用回归的方法，从统计上减少这个偏差*

第三部分是anchor水平方向偏移的回归损失，依然使用smooth L1 Loss，其中$o=t_x$代表网络预测的文本线水平偏差，$o^*=t_x^*$代表真实偏差

## 2.bounding box regression
bounding box(bbox) 指的是目标检测中需要生成的准确包括目标的矩形框。

一个bounding box是否有效衡量标准是IoU指数，即交并集之比

### why regression？
因为在训练神经网络的时候，我们往往生成的bounding box 与 ground truth(GT)在位置（使用两个方框的中心点衡量）和尺寸（使用两个框对应长、宽的比例衡量）上有误差。

所以我们在神经网络生成文本框之后又加入了一个新的任务，那就是bounding box regression， 希望以回归的方法进一步对网络参数进行更新。

我们用 [x,y,h,w]表示一个神经网络生成的proposal（可以理解为预测框），其中，x, y 是proposal的中心点坐标，h，w是proposal的高和宽，同理也可以表示GT box(G)

回归的目的就是尽可能将proposal通过一系列*变换*得到一个与G非常接近的方框$\hat{G}$

上文提到的变换，R-CNN论文中定义为平移变换和尺度放缩变换
    
1.平移变换 
$$\Delta X = P_w d_x(P)\tag{1}$$
$$\Delta Y=P_w d_y(P) \tag{2}$$
*平移变换中使用指数函数是因为神经网络的输出有正有负，而放缩量是正数*

2.放缩变换
$$S_w=exp(d_w(P))\tag{3}$$
$$S_h=exp(d_h(P))\tag{4}$$

所以我们回归就是要学习$d_x, d_y, d_w, d_h$这四个线性变换

在训练阶段，regression部分流程如下：
输入：神经网络的编码结果（卷积结果），Ground Truth 坐标 
输出：真正需要的平移量$(t_x, t_y)$和放缩量$(t_w, t_h)$
由公式(1)~(4)， 可以推导出
$$d_x(P;w_x)=t_x=(G_x-P_x)/P_w$$
$$d_y(P;w_y)=t_x=(G_y-P_y)/P_h$$
$$d_w(P;w_w)=t_x=\log{(G_w/P_w)}$$
$$d_h(P;w_h)=t_x=\log{(G_h/P_h)}$$
*为什么要除以宽高？  大的box可能绝对偏移量比小的box大，所以要除以尺寸以归一化偏移*

因此回归的目标函数可以表示为：
$$d_*(P)=w_*^T \Phi_5(P)$$
其中$\Phi_5(P)$ 是输入proposal 的特征向量，$w_*$是要学习的参数
损失函数可以表示为：
$$Loss = \Sigma_i^N{(t_*^i-w_*^T\Phi_5(P))^2}$$
优化目标为：
$$w_*=argmax_{w_*}{(loss+\lambda||\hat{w_*}||^2)}$$

*为什么说IoU较大时才是线性变换（因为指数变换本就不是线性变换）？？*
$$t_w=\log{(G_w/P_w)}=\log{(1+(G_w-P_w)/P_w)}$$
又已知  $\lim_{x->0}\log{(1+x)} = x$
所以在$(G_w-P_w)$较小时，$t_w$可以近似为一个线性函数

## 3. NMS
非极大值抑制：剔除同一份目标上的重叠建议框（proposal box），最终一个目标只会被一个得分最大的建议框标出，NMS就是抑制冗余的矩形框，保留最优框的过程。

实现步骤：
1. 假设图片中有20类目标，而每种类别的建议框有2000个，可以组成一个2000*20的矩阵，每列是同一类别对应的proposal box 的score，从上往下，从大到小；
2. 从每次最大的得分建议框开始，向下检索，与每一个pbox进行IoU计算，若IoU>阈值，就认为当前两个pbox标记的是同一个目标，那就删去score较小的pbox；
3. 从score次大的pbox开始，继续重复步骤2；
4. 重复2、3直到遍历完所有行。

## 4. Smooth L1 Loss
首先介绍L1 Loss、L2 Loss：
$$L1 = |f(x)-Y|,     L1'=\pm f'(x)$$
$$L2 = |f(x)-Y|^2,   L2'=2(f(x)-Y)f'(x)$$
L1范数的缺点是在误差接近0时不平滑，所以很少使用

L2范数的缺点是没有对离群点做很好的归一化，当存在一个很大的离群点时，会极大的影响loss。由于L2的梯度往往大于L1，所以收敛速度也快得多

针对L1、L2范数的缺点宽，产生了huber loss
$$huber\_loss = 
\begin{cases}
\frac{1}{2}\sum_{i=1}^{n}(f(x)-Y)^2 & |f(x)-Y|\le1 \\
\sum_{i=1}^{n}|f(x)-Y|-0.5 & otherwise
\end{cases}
$$
综合了L1， L2的特征。

smooth L1是L1的变形，被广泛用于Faster RNN、 SSD等网络计算
$$smooth\_L1 = 
\begin{cases}
0.5x^2 & |x| \lt 1 \\
|x|-0.5&otherwise
\end{cases}
$$
其中x就是模型的delta（即$f(x)-Y$）。
$$\frac{\mathrm{d}smooth\_L1}{\mathrm{d}x} = 
\begin{cases}
x & |x|<1\\
\pm1 & otherwise
\end{cases}
$$
相对于L1, L2，其对离群点没有那么敏感，也可以控制梯度的量级。

因为在x较小时，梯度也会变小，以免震荡。在x较大时，梯度被限制，所以离群点不会过分影响参数更新


## anchor的作用：
1. 将原始图片切割成16*16的小区域；
2. 在每个区域的中心点上，生成10个宽为16，高为[11,...,287]的anchor；
3. 将图中所有的anchor与标记的gt_box进行IoU计算；
4. 根据IoU的结果，将图片分为【正样例，负样例，无关样例】三类
5. 神经网络对每个conv5_3的feature map，每一个像素点输出10组预测anchor与实际anchor的偏移（dy，dh），我们根据label，去掉无关样例，分别对正样例和负样例计算损失

##思考
1. 为什么anchor那么大，甚至超出了感受野的范围？
    因为我们希望在学习的时候，对当前16*16区域学习，不仅能够预测当前区域是否是目标，更可以窥一斑而知全豹，从当前部分区域判断出整个目标的尺寸


2. 为什么预测的proposal宽度固定为16？
   1. 我的理解：因为conv5_3对应的是原图上一个228*228，stride = 16的卷积核，我们在conv5_3上卷积，就相当于用这个卷积核在原图上以16个像素移动，所以我们可以认为每个区域只预测原图上16\*16的小格子
   

使用了aliicpr数据集，随机三张拼接图片，1400训练集， 600测试集，23轮就开始收敛