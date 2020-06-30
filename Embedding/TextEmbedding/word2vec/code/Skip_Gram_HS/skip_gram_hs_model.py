import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.w_embeddings = nn.Embedding(2*vocab_size-1, emb_size, sparse=True)
        self.v_embeddings = nn.Embedding(2*vocab_size-1, emb_size, sparse=True)
        self._init_emb()

    def _init_emb(self):
        initrange = 0.5 / self.emb_size
        self.w_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_w, pos_v,neg_w, neg_v):
        emb_w = self.w_embeddings(torch.LongTensor(pos_w).cuda())  # 转为tensor 大小 [ mini_batch_size * emb_dimension ]
        neg_emb_w = self.w_embeddings(torch.LongTensor(neg_w).cuda())
        emb_v = self.v_embeddings(torch.LongTensor(pos_v).cuda())
        neg_emb_v = self.v_embeddings(torch.LongTensor(neg_v).cuda())  # 转换后大小 [ negative_sampling_number * mini_batch_size * emb_dimension ]
        score = torch.mul(emb_w, emb_v).squeeze()
        score = torch.sum(score, dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = F.logsigmoid(score)
        neg_score = torch.mul(neg_emb_w, neg_emb_v).squeeze()
        neg_score = torch.sum(neg_score, dim=1)
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = F.logsigmoid(-neg_score)
        # L = log sigmoid (Xw.T * θv) + [log sigmoid (-Xw.T * θv)]
        loss = -1 * (torch.sum(score) + torch.sum(neg_score))
        return loss

    def save_embedding(self, id2word, file_name):
        embedding = self.w_embeddings.weight.data.cpu().numpy()
        fout = open(file_name, 'w')
        fout.write('%d %d\n' % (len(id2word), self.emb_size))
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))


def test():
    model = SkipGramModel(100, 10)
    id2word = dict()
    for i in range(100):
        id2word[i] = str(i)
    pos_w = [0, 0, 1, 1, 1]
    pos_v = [1, 2, 0, 2, 3]
    neg_v = [[23, 42, 32], [32, 24, 53], [32, 24, 53], [32, 24, 53], [32, 24, 53]]
    model.forward(pos_w, pos_v, neg_v)
    model.save_embedding(id2word, 'test.txt')


if __name__ == '__main__':
    test()
