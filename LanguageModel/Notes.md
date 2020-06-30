# Pre-trained Language Model
## GPT1.0
* 基于Fine-Tuning的预训练方法,与word2vec不同点在于：word2vec在fine-tuning过程中，只调整embedding矩阵而不调整网络结构，而GPT的网络结构参数也参与到Fine-Tuning过程中。
* 典型的两阶段过程，第一阶段利用语言模型进行预训练,第二阶段通过Fine-Tuning的模式解决下游任务。
* 特征抽取器采用Transformer，仅采用单向的语言模型。
* GPT的使用: 需要将下游任务的网络结构改造成与GPT网络结构相同，然后使用预训练的参数初始化网络，并结合下游任务进行Fine-Tuing,以使得网络结构更加适合下游任务。finetuning时，通常在其后加入一个线性层，而不再接入复杂的网络。
## BERT
* (Bidirectional Encode Representations from Transformer)
* 最关键有两点:特征抽取器采用Transformer;采用双向语言模型进行预训练
* BERT与GPT及ELMO的对比:如果把ELMO的特征抽取器换成Transformer，那么就得到了BERT;如果把GPT的预训练阶段换成双向语言模型，就得到了BERT。
* 