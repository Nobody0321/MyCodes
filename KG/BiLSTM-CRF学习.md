# BiLSTM-CRF学习笔记（原理和理解）

## 
BiLSTM-CRF 被提出用于NER或者词性标注，效果比单纯的CRF或者lstm或者bilstm效果都要好。

根据pytorch官方指南(https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#bi-lstm-conditional-random-field-discussion)，实现了BiLSTM-CRF一个toy级别的源码。下面是我个人的学习理解过程。

## 1. LSTM
LSTM的原理前人已经解释的非常清楚了：https://zhuanlan.zhihu.com/p/32085405
BiLSTM-CRF中，BiLSTM部分主要用于，根据一个单词的上下文，给出当前单词对应标签的概率分布，可以把BiLSTM看成一个编码层。
比如，对于标签集{N, V, O}和单词China，BiLSTM可能输出形如(0.88,-1.23,0.03)的非归一化概率分布。
这个分布我们看作是crf的特征分布输入，那么在CRF中我们需要学习的就是特征转移概率。

## 2. CRF
主要讲一下代码中要用到的CRF的预测（维特比解码）
维特比算法流程:
1.求出位置1的各个标记的非规范化概率$δ_1 (j)$
    $$δ_1 (j)=w∗F_1 (y_0=START,y_i=j,x),  j=1,2,…,m$$

2.由递推公式（前后向概率计算）
    $$δ_i (l)=max_{(1≤j≤m)} \{δ_{i−1} (j)+w∗F_i (y_{i−1}=j,y_i=l,x)\},   l=1,2,…,l$$
    每一步都保留当前所有可能的状态$l$ 对应的最大的非规范化概率，
    并将最大非规范化概率状态对应的路径（当前状态得到最大概率时上一步的状态$y_i$）记录
    $Ψ_i (l)=arg ⁡max_{(1≤j≤m)} \{δ_{i−1} (j)+w∗F_i (y_{i−1}=j,y_i=l,x)\} =arg max⁡{δ_i (l)},  l=1,2,…,m$
    就是$P_{ij}$的取值有m*m个，对每一个$y_j$，都确定一个（而不是可能的m个）能最大化概率的$y_i$状态

3.递推到$i=n$时终止
    这时候求得非规范化概率的最大值为
    $$max_y⁡\{w∗F(y,x)\}=max_{(1≤j≤m)} δ_n (j) =max_{(1≤j≤m)}\{⁡δ_{n−1}(j)+w∗F_n (y_{n−1}=Ψ_{n−1} (k),y_{i=l},x)\},   l=1,2,…,m$$
    最优路径终点
    $$y_n^∗=arg⁡max_{(1≤j≤m)}⁡{δ_n (j)}$$
    
4.递归路径
    由最优路径终点递归得到的最优路径（由当前最大概率状态状态对应的上一步状态，然后递归）
    $$y_i^∗=Ψ_{i+1} (y_{i+1}^∗ ),  i=n−1,n−2,…,1$$
    求得最优路径：
    $$y^∗=(y_1^∗,y_2^∗,…,y_n^∗ )^T$$

## 3. 损失函数
最后由CRF输出，损失函数的形式主要由CRF给出
在BiLSTM-CRF中，给定输入序列X，网络输出对应的标注序列y，得分为
$$S(X,y)=∑_{i=0}^n A_{y_i,y_{i+1} } +∑_{i=1}^n P_{i,y_i }$$
（转移概率和状态概率之和）
利用softmax函数，我们为每一个正确的tag序列y定义一个概率值
$$p(y│X)=\frac{e^S(X,y)}{∑_{y′∈Y_X} e^{S(X,y′)}  }$$

在训练中，我们的目标就是最大化概率p(y│X) ，怎么最大化呢，用对数似然（因为p(y│X)中存在指数和除法，对数似然可以化简这些运算）
对数似然形式如下：
$$log⁡(p(y│X)=log⁡ \frac{e^s{(X,y)}}{∑_{y∈Y_X}e^s(X,y^′)}=S(X,y)−log⁡(∑_{y^′∈Y_X}e^s(X,y^′ )  )$$
最大化这个对数似然，就是最小化他的相反数：
￥$−log⁡(p(y│X))=log⁡(∑_{y^′∈Y_X}e^s(X,y^′ )  )-S(X,y)$$
(loss function/object function)
最小化可以借助梯度下降实现

在对损失函数进行计算的时候，前一项$S(X,y)$很容易计算，
后一项$log⁡(∑_{y^′∈Y_X}e^s(X,y^′ )  )$比较复杂，计算过程中由于指数较大常常会出现上溢或者下溢，
由公式 $log∑e^{(x_i )}=a+ log⁡∑e^{(x_i−a)}$，可以借助a对指数进行放缩，通常a取$x_i$的最大值（即$a=max⁡[X_i ]$），这可以保证指数最大不会超过0，于是你就不会上溢出。即便剩余的部分下溢出了，你也能得到一个合理的值。

又因为$log⁡(∑_y e^{log {(∑_x e^x)+y}}  )$，在$log$取$e$作为底数的情况下，可以化简为
$log⁡(∑_ye^y ∗e^{log⁡(∑_xe^x ) } )=log⁡(∑_ye^y ∗∑_xe^x )=log⁡(∑_y∑_xe^{x+y} )$。
log_sum_exp因为需要计算所有路径，那么在计算过程中，计算每一步路径得分之和和直接计算全局得分是等价的，就可以大大减少计算时间。
当前的分数可以由上一步的总得分+转移得分+状态得分得到，这也是pytorch范例中
next_tag_var = forward_var + trans_score + emit_score 
的由来






注意，由于程序中比较好选一整行而不是一整列，所以调换i,j的含义，t[i][j]表示从j状态转移到i状态的转移概率


直接分析源码的前向传播部分，其中_get_lstm_features函数调用了pytorch的BiLSTM

    def forward(self, sentence):
        """
        重写前向传播
        :param sentence: 输入的句子序列
        :return:返回分数和标记序列
        """
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
        

源码的维特比算法实现，在训练结束，还要使用该算法进行预测

    def _viterbi_decode(self, feats):
        """
        使用维特比算法预测
        :param feats:lstm的所有输出
        :return:返回最大概率和最优路径
        """ 
        backpointers = []

        # step1. 初始化
        init_vvars = torch.full((1, self.tagset_size), -1000.)
        # 初始化第一步的转移概率
        init_vvars[0][self.tag_to_idx[START_TAG]] = 0

        # 初始化每一步的非规范化概率
        forward_var = init_vvars 

        # step2. 递推
        # 遍历每一个单词通过bilstm输出的概率分布
        for feat in feats:
            # 每次循环重新统计
            bptrs_t = []
            viterbivars_t = []
            
            for next_tag in range(self.tagset_size):
                # 根据维特比算法
                # 下一个tag_i+1的非归一化概率是上一步概率加转移概率（势函数和势函数的权重都统一看成转移概率的一部分）
                next_tag_var = forward_var + self.transitions[next_tag]
                # next_tag_var = tensor([[-3.8879e-01,  1.5657e+00,  1.7734e+00, -9.9964e+03, -9.9990e+03]])

                # 计算所有前向概率（?）
                # CRF是单步线性链马尔可夫，所以每个状态只和他上1个状态有关，可以用二维的概率转移矩阵表示

                # 保存当前最大状态
                best_tag_id = argmax(next_tag_var)
                # best_tag_id = torch.argmax(next_tag_var).item()
                bptrs_t.append(best_tag_id)

                # 从一个1*N向量中取出一个值（标量），将这个标量再转换成一维向量
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))  

            # viterbivars 长度为self.tagset_size，对应feat的维度
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            # 记录每一个时间i，每个状态取值l取最大非规范化概率对应的上一步状态
            backpointers.append(bptrs_t)
        
        # step3. 终止
        terminal_var = forward_var + self.transitions[self.tag_to_idx[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # step4. 返回路径
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_idx[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

源码的损失函数计算

    def neg_log_likelihood(self, sentence, tags):
        """
        实现负对数似然函数
        :param sentence:
        :param tags:
        :return:
        """
        # 返回句子中每个单词对应的标签概率分布
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags) # 输出路径的得分（S（X,y））
        # 返回负对数似然函数的结果
        return forward_score - gold_score

    
    def _forward_alg(self, feats):
        """
        使用前向算法计算损失函数的第一项log(\sum(exp(S(X,y’))))
        :param feats: 从BiLSTM输出的特征
        :return: 返回
        """
        init_alphas = torch.full((1, self.tagset_size), -10000.)

        init_alphas[0][self.tag_to_idx[START_TAG]] = 0.

        forward_var = init_alphas

        for feat in feats:
            # 存放t时刻的 概率状态
            alphas_t = [] 
            for current_tag in range(self.tagset_size):
                # lstm输出的是非归一化分布概率
                emit_score = feat[current_tag].view(1, -1).expand(1, self.tagset_size)

                # self.transitions[current_tag] 就是从上一时刻所有状态转移到当前某状态的非归一化转移概率
                # 取出的转移矩阵的行是一维的，这里调用view函数转换成二维矩阵
                trans_score = self.transitions[current_tag].view(1, -1)

                # trans_score + emit_score 等于所有特征函数之和
                # forward 是截至上一步的得分
                current_tag_var = forward_var + trans_score + emit_score
                
                alphas_t.append(log_sum_exp(current_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1) # 调用view函数转换成1*N向量
        terminal_var = forward_var + self.transitions[self.tag_to_idx[STOP_TAG]]

        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags):
        """
        返回S(X,y)
        :param feats: 从BiLSTM输出的特征
        :param tags: CRF输出的标记路径
        :return:
        """
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_idx[START_TAG]], dtype=torch.long),tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_idx[STOP_TAG],tags[-1]]
        return score