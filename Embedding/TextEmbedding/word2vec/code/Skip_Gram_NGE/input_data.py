import numpy as np
from collections import deque

class InputData:
    def __init__(self,input_file_name,min_count):
        self.input_file_name = input_file_name
        self.index = 0
        self.input_file = open(self.input_file_name,"r",encoding="utf-8")
        self.min_count = min_count
        self.wordid_frequency_dict = dict()
        self.word_count = 0
        self.word_count_sum = 0
        self.sentence_count = 0
        self.id2word_dict = dict()
        self.word2id_dict = dict()
        self._init_dict()  # 初始化字典
        self.sample_table = []
        self._init_sample_table()  # 初始化负采样映射表
        self.get_wordId_list()
        self.word_pairs_queue = deque()
        # 结果展示
        print('Word Count is:', self.word_count)
        print('Word Count Sum is', self.word_count_sum)
        print('Sentence Count is:', self.sentence_count)
    def _init_dict(self):
        word_freq = dict()
        for line in self.input_file:
            line = line.strip().split()
            self.word_count_sum +=len(line)
            self.sentence_count +=1
            for i,word in enumerate(line):
                if i%1000000==0:
                    print (i,len(line))
                if word_freq.get(word)==None:
                    word_freq[word] = 1
                else:
                    word_freq[word] += 1
        for i,word in enumerate(word_freq):
            if i % 100000 == 0:
                print(i, len(word_freq))
            if word_freq[word]<self.min_count:
                self.word_count_sum -= word_freq[word]
                continue
            self.word2id_dict[word] = len(self.word2id_dict)
            self.id2word_dict[len(self.id2word_dict)] = word
            self.wordid_frequency_dict[len(self.word2id_dict)-1] = word_freq[word]
        self.word_count =len(self.word2id_dict)
    def _init_sample_table(self):
        sample_table_size = 1e8
        pow_frequency = np.array(list(self.wordid_frequency_dict.values())) ** 0.75
        word_pow_sum = sum(pow_frequency)
        ratio_array = pow_frequency / word_pow_sum
        word_count_list = np.round(ratio_array * sample_table_size)
        for word_index, word_freq in enumerate(word_count_list):
            self.sample_table += [word_index] * int(word_freq)
        self.sample_table = np.array(self.sample_table)
        np.random.shuffle(self.sample_table)
    def get_wordId_list(self):
        self.input_file = open(self.input_file_name, encoding="utf-8")
        sentence = self.input_file.readline()
        wordId_list = []  # 一句中的所有word 对应的 id
        sentence = sentence.strip().split(' ')
        for i,word in enumerate(sentence):
            if i%1000000==0:
                print (i,len(sentence))
            try:
                word_id = self.word2id_dict[word]
                wordId_list.append(word_id)
            except:
                continue
        self.wordId_list = wordId_list
    def get_batch_pairs(self,batch_size,window_size):
        while len(self.word_pairs_queue) < batch_size:
            for _ in range(1000):
                if self.index == len(self.wordId_list):
                    self.index = 0
                wordId_w = self.wordId_list[self.index]
                for i in range(max(self.index - window_size, 0),
                                         min(self.index + window_size + 1,len(self.wordId_list))):

                    wordId_v = self.wordId_list[i]
                    if self.index == i:  # 上下文=中心词 跳过
                        continue
                    self.word_pairs_queue.append((wordId_w, wordId_v))
                self.index+=1
        result_pairs = []  # 返回mini-batch大小的正采样对
        for _ in range(batch_size):
            result_pairs.append(self.word_pairs_queue.popleft())
        return result_pairs


    # 获取负采样 输入正采样对数组 positive_pairs，以及每个正采样对需要的负采样数 neg_count 从采样表抽取负采样词的id
    # （假设数据够大，不考虑负采样=正采样的小概率情况）
    def get_negative_sampling(self, positive_pairs, neg_count):
        neg_v = np.random.choice(self.sample_table, size=(len(positive_pairs), neg_count)).tolist()
        return neg_v

    # 估计数据中正采样对数，用于设定batch
    def evaluate_pairs_count(self, window_size):
        return self.word_count_sum * (2 * window_size) - self.sentence_count * (
                    1 + window_size) * window_size

    # 测试所有方法
if __name__=="__main__":
    test_data = InputData('../data/text8.txt', 1)
    test_data.evaluate_pairs_count(2)
    pos_pairs = test_data.get_batch_pairs(10, 2)
    print('正采样:')
    print(pos_pairs)
    pos_word_pairs = []
    for pair in pos_pairs:
        pos_word_pairs.append((test_data.id2word_dict[pair[0]], test_data.id2word_dict[pair[1]]))
    print(pos_word_pairs)
    neg_pair = test_data.get_negative_sampling(pos_pairs, 3)
    print('负采样:')
    print(neg_pair)
    neg_word_pair = []
    for pair in neg_pair:
        neg_word_pair.append(
            (test_data.id2word_dict[pair[0]], test_data.id2word_dict[pair[1]], test_data.id2word_dict[pair[2]]))
    print(neg_word_pair)



