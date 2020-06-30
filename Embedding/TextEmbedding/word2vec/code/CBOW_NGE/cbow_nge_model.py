import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CBOWModel(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(CBOWModel, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.u_embeddings = nn.Embedding(self.vocab_size, self.emb_size, sparse=True)
        self.w_embeddings = nn.Embedding(self.vocab_size, self.emb_size, sparse=True)
        self._init_embedding()  # 初始化

    def _init_embedding(self):
        int_range = 0.5 / self.emb_size
        self.u_embeddings.weight.data.uniform_(-int_range, int_range)
        self.w_embeddings.weight.data.uniform_(-0, 0)


    def forward(self, pos_u, pos_w, neg_w):
        pos_u_emb = []  # 上下文embedding
        for per_Xw in pos_u:
            # 上下文矩阵的第一维不同词值不同，如第一个词上下文为c，第二个词上下文为c+1，需要统一化
            per_u_emb = self.u_embeddings(torch.LongTensor(per_Xw).cuda())  # 对上下文每个词转embedding
            per_u_numpy = per_u_emb.data.cpu().numpy()  # 转回numpy，好对其求和
            per_u_numpy = np.sum(per_u_numpy, axis=0)
            per_u_list = per_u_numpy.tolist()  # 为上下文词向量Xw的值
            pos_u_emb.append(per_u_list)  # 放回数组
        pos_u_emb = torch.FloatTensor(pos_u_emb).cuda()  # 转为tensor 大小 [ mini_batch_size * emb_size ]
        pos_w_emb = self.w_embeddings(torch.LongTensor(pos_w).cuda())  # 转换后大小 [ mini_batch_size * emb_size ]
        neg_w_emb = self.w_embeddings(
            torch.LongTensor(neg_w).cuda())  # 转换后大小 [ mini_batch_size*negative_sampling_number  * emb_size ]
        # 计算梯度上升（ 结果 *（-1） 即可变为损失函数 ->可使用torch的梯度下降）
        score_1 = torch.mul(pos_u_emb, pos_w_emb)  # Xw.T * θu
        score_2 = torch.sum(score_1, dim=1)  # 点积和
        score_3 = F.logsigmoid(score_2)  # log sigmoid (Xw.T * θu)
        neg_score_1 = torch.bmm(neg_w_emb, pos_u_emb.unsqueeze(2))  # batch_size*negative_sampling_number
        neg_score_2 = F.logsigmoid((-1) * neg_score_1)
        loss = torch.sum(score_3) + torch.sum(neg_score_2)
        return -1 * loss

    # 存储embedding
    def save_embedding(self, id2word_dict, file_name):
        embedding = self.u_embeddings.weight.data.numpy()
        file_output = open(file_name, 'w')
        file_output.write('%d %d\n' % (self.vocab_size, self.emb_size))
        for id, word in id2word_dict.items():
            e = embedding[id]
            e = ' '.join(map(lambda x: str(x), e))
            file_output.write('%s %s\n' % (word, e))


def test():
    model = CBOWModel(100, 10)
    id2word = dict()
    for i in range(100):
        id2word[i] = str(i)
    pos_u = [[1, 2, 3], [0, 1, 2, 3]]
    pos_w = [0, 1]
    neg_w = [[23, 42], [32, 24]]
    model.forward(pos_u, pos_w, neg_w)
    model.save_embedding(id2word, 'test.txt')


if __name__ == '__main__':
    test()
