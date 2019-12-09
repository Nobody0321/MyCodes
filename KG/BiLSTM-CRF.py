import torch
import torch.nn as nn 
# import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4


def argmax(vec):
    """
    return the argmax as python int
    pytorch内部已经实现了这个函数(torch.argmax)
    :param vec: 1*N的矩阵
    :return:
    """
    value, idx = torch.max(vec, 1)
    # 返回每一行（其实只有一行）的最大值及其对应的下标
    # 在有第二个参数（选择维度）时返回对应下标，否则只返回全局最大值
    return idx.item()


def prepare_sequence(seq, to_idx):
    """
    使用词袋模型，将单词转化为数字/向量
    :param seq:句子，默认已经分词
    :param to_idx:
    :return:
    """
    idxs = [to_idx[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def log_sum_exp(vec):
    """
    使用前向算法简化计算logsumexp
    :param vec:1*N向量，形如 tensor([[-3.8879e-01,  1.5657e+00,  1.7734e+00, -9.9964e+03, -9.9990e+03]])
    :return:返回logsumexp
    """
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
    

# create model
class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_idx, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.vocab_size = vocab_size
        self.tag_to_idx = tag_to_idx
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = len(tag_to_idx)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2, num_layers=1, bidirectional=True)

        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

        # 转移概率矩阵
        # 由于程序中比较好选一整行而不是一整列，所以调换i,j的含义，t[i][j]表示从j状态转移到i状态的转移概率
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # 给 i = start_tag ，j = Stop_tag 特殊标记
        self.transitions.data[tag_to_idx[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_idx[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        """
        初始化lstm的隐状态（归零）
        :return: 初始化后的隐状态矩阵
        """
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        """
        使用前向算法计算打分函数S(X,y)
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

    def _get_lstm_features(self, sentence):
        """
        lstm输出
        :param sentence:输入句子序列
        :return:输出BilSTM特征（认为是学习到的状态估计概率）
        """
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_features = self.hidden2tag(lstm_out)
        return lstm_features

    def _score_sentence(self, feats, tags):
        """
        自定义的BiLSTM的损失函数
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

    def _viterbi_decode(self, feats):
        """
        使用维特比算法预测
        :param feats:lstm的所有输出
        :return:返回最大概率和最优路径
        """ 
        backpointers = []

        # step1. 初始化
        init_vvars = torch.full((1, self.tagset_size), -1000.)
        #1. 初始化第一步
        init_vvars[0][self.tag_to_idx[START_TAG]] = 0

        # 初始化每一步的非规范化概率
        forward_var = init_vvars 

        # step2. 递推
        # 遍历每一个单词通过bilstm输出的概率分布
        for feat in feats:
            # 每次循环重新统计
            bptrs_t = []
            viterbivars_t = []
            
            for current_tag in range(self.tagset_size):
                # 根据维特比算法
                # 下一个tag_i+1的非归一化概率是上一步概率加转移概率（势函数和势函数的权重都统一看成转移概率的一部分）
                current_tag_var = forward_var + self.transitions[current_tag]
                # current_tag_var = tensor([[-3.8879e-01,  1.5657e+00,  1.7734e+00, -9.9964e+03, -9.9990e+03]])

                # 计算所有前向概率（?）
                # CRF是单步线性链马尔可夫，所以每个状态只和他上1个状态有关，可以用二维的概率转移矩阵表示

                # 保存当前最大状态
                best_tag_id = argmax(current_tag_var)
                # best_tag_id = torch.argmax(current_tag_var).item()
                bptrs_t.append(best_tag_id)

                # 从一个1*N向量中取出一个值（标量），将这个标量再转换成一维向量
                viterbivars_t.append(current_tag_var[0][best_tag_id].view(1))  

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

    def neg_log_likelihood(self, sentence, tags):
        """
        实现负对数似然函数
        :param sentence:
        :param tags:
        :return:
        """
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags) # 输出路径的得分（S（X,y））
        # 返回负对数似然函数的结果
        return forward_score - gold_score

    def forward(self, sentence):
        """
        重写前向传播
        :param sentence: 输入的句子序列
        :return:
        """
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


# Make up some training data
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    print(model(precheck_sent))

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))
# We got it!