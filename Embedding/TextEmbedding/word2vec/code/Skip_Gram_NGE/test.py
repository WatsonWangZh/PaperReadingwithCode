import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
def read_file():
    data = open("../data/word-test.v1.txt").readlines()[1:]
    test = {}
    now = ""
    for line in data:
        line = line.lower()
        line = line.split()
        if len(line)==2:
            now = line[1]
        elif len(line)==4:
            if test.get(now)==None:
                test[now] = [line]
            else:
                test[now].append(line)


    test_datas = []
    for key,value in test.items():
        test_datas.extend(value)
    print (len(test_datas))
    return test_datas
def test(test_datas):
    data = open("../results/skip_gram_neg.txt").readlines()[1:]
    word2id = {}
    id2word = {}
    word2embed={}
    embedding = []
    for line in data:
        line = line.split()
        word2id[line[0]] = len(word2id)
        id2word[len(id2word)] = line[0]
        e = np.array([float(x) for x in line[1:]])
        word2embed[line[0]] = e
        embedding.append(e)
    embedding = np.array(embedding)
    true_sample = 0
    for i,data in enumerate(test_datas):
        if i%100==0:
            print (i,len(test_datas),true_sample)
        if word2id.get(data[0])!=None and word2id.get(data[1])!=None and word2id.get(data[2])!=None and word2id.get(data[3])!=None:
            vec_tmp = word2embed[data[1]]-word2embed[data[0]]+word2embed[data[2]]

            cos_result = cosine_similarity(np.reshape(vec_tmp,[1,-1]),embedding)
            pred = np.argmax(cos_result)
            if id2word[pred]==data[3]:
                true_sample+=1

    print (true_sample)
def test_2():
    data = open("../results/skip_gram_neg.txt").readlines()[1:]
    word2id = {}
    id2word = {}
    word2embed = {}
    embedding = []
    for line in data:
        line = line.split()
        word2id[line[0]] = len(word2id)
        id2word[len(id2word)] = line[0]
        e = np.array([float(x) for x in line[1:]])
        word2embed[line[0]] = e
        embedding.append(e)

    embedding = np.array(embedding)

    word = "good"
    vec_tmp = word2embed[word]
    cos_result = cosine_similarity(np.reshape(vec_tmp, [1, -1]), embedding)
    cos_result = cos_result.reshape([-1])
    result = np.argsort(cos_result)
    for i in range(10):
        print (result[len(result)-i-1])
        print (id2word[int(result[len(result)-i-1])])
test_datas = read_file()
#test(test_datas)
test_2()