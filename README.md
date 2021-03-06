# Reading with Code
<div align="center">
    <img src="picture/rwc.png" width="259" height="195">
    
A Personal Library.  
Paper and books that I already read and will read, with notes and code.  
![License](https://img.shields.io/badge/License-MIT-green.svg) ![Update](https://img.shields.io/badge/Update-Daily-green.svg)
</div>

## Category
- [Conference Helper](#conference-helper)
- [Paper Tools](#paper-tools)
- [Competition Websites](#competition-websites)
- [GPU Platform](#gpu-platform)
- [NLP](#nlp)
  - [Backbone](#nlp-backbone)
    - [Network](#network)
    - [Discussion](#discussion)
  - [Embedding](#embedding)
    - [Text Embedding](#text-embedding)
      - [Word Embedding](#word-embedding)
      - [Sentence Embedding](#sentence-embedding)
      - [Context Embedding](#context-embedding)
      - [Paragraph Embedding](#paragraph-embedding)
      - [Relation Embedding](#relation-embedding)
    - [KG Embedding](#knowledge-graph-embedding)
      - [Framework:](#framework)
      - [Node (Vertex) Embedding](#node-vertex-embedding)
        - [Translation Based](#translation-based)
        - [Random Walk Based](#random-walk-based)
      - [Graph Embedding](#graph-embedding)
  - [Pre-trained Language Model](#pre-trained-language-model)
  - [Mechanism](#mechanism)
    - [Attention](#attention)
    - [Memorization](#memorization)
  - [Task Classification](#task-classification)
    - [Sequence Labeling](#sequence-labeling)
    - [Text Classification](#text-classification)
    - [Sentence Pair Relationship](#sentence-pair-relationship)
      - [Sentence Similarity](#sentence-similarity)
      - [Natural Language Inference](#natural-language-inference)
    - [Generalization](#generalization)
      - [Machine Translation](#machine-translation)
      - [Text Summarization](#text-summarization)
- [CV](#cv)
  - [Origin and Pioneer](#origin-and-pioneer)
  - [Backbone](#cv-backbone)
    - [Image Recoginization](#image-recoginization)
    - [Image Segmentation](#image-segmentation)
    - [Object Detection](#object-detection)

- [General Elements](#general-elements)
  - [Activation Function](#activation-function)
  - [Loss Function](#loss-function)
  - [Optimizer](#optimizer)
  - [Training Tricks](#training-tricks)
    - [Initialization](#initialization)
    - [Middle Layer Operation](#middle-layer-operation)
    - [Out Layer Operation](#out-layer-operation)
    - [Gradient Operation](#gradient-operation)

- [Graph Neural Network](#graph-neural-network)
  - [Framework:](#framework-1)
  - [Survey](#survey)
  - [Spatial Domain](#spatial-domain)
  - [Spectral Domain](#spectral-domain)

- [Machine Learning](#machine-learning)
  - [Ensemble Learning](#ensemble-learning)

- [System](#system)

- [Useful Resources](#useful-resources)
  - [Toolkits](#toolkits)
  - [Terms Explanation](#terms-explanation)
  - [Articles](#articles)
  - [Books](#books)
  - [Course](#course)
  - [Websites](#websites)
  
# Conference Helper
* [WikiCFP](http://www.wikicfp.com/cfp/)
* [Guide2Research](http://www.guide2research.com/)
* [EasyChair Smart CFP](https://easychair.org/cfp/)
* [IEEE Conferences](https://www.ieee.org/conferences/index.html)
* [Upcoming Conference List](http://www.conferencelist.info/upcoming.html)
* [AI Deadline](https://aideadlin.es)
* [Conferences-with-ccf](https://jackietseng.github.io/conference_call_for_paper/conferences-with-ccf.html)
* [AMiner](https://www.aminer.cn/ranks/conf?type=ccf)
* [Call4Papers](http://123.57.137.208/index.html)

# Paper Tools
## Search Engine
* [Semantic Scholar](https://www.semanticscholar.org/)
* [Google Scholar](https://scholar.google.com/)
* [dblp](https://dblp.uni-trier.de/)
* [ACM DL](https://dl.acm.org/)
* [IEEE Xplore DL](https://ieeexplore.ieee.org/Xplore/home.jsp)
* [Elsevier ScienceDirect](https://www.sciencedirect.com/)
* [Springer](https://www.springer.com/gp)
* [Wiley Online Lib](https://onlinelibrary.wiley.com/)
* [SCI-HUB](https://sci-hub.se/)

## Reference Management 
* [Mendeley](https://www.mendeley.com/)
* [Papers 3](https://www.papersapp.com/)
* [Endnote](https://endnote.com/)
* [Zotero](https://www.zotero.org/)
## Writing
* [Overleaf](https://www.overleaf.com/)
## Grammar Checker
* [WitheSmoke](https://www.whitesmoke.com/)
* [Linggle](https://linggle.com/)
* [Grammarly](https://www.grammarly.com/)

# Competition Websites
* [Kaggle](https://www.kaggle.com/)
* [TIANCHI天池](https://tianchi.aliyun.com/competition/gameList/activeList)
* [DataCastle](https://www.dcjingsai.com/v2/index.html)
* [DataFountain](https://www.datafountain.cn/)
* [biendata](https://www.biendata.xyz/)
* [kesci](https://www.kesci.com/home/competition)

# GPU Platform
* [Colab](https://colab.research.google.com)
* [Google Cloud](https://cloud.google.com/)
* [AWS](https://aws.amazon.com/console/)

# NLP
## NLP Backbone 
### Network
* LSTM Neural Computation 1997 [LONG SHORT-TERM MEMORY](https://www.bioinf.jku.at/publications/older/2604.pdf)   
* GRU SSST-8 Workshop 2014 [On the Properties of Neural Machine Translation: Encoder–Decoder Approaches](https://arxiv.org/pdf/1409.1259.pdf)
* TextCNN ACL 2014 [Convolutional Neural Networks for Sentence Classification](https://www.aclweb.org/anthology/D14-1181.pdf)[code]() 
* C-LSTM arXiv 2015 [A C-LSTM Neural Network for Text Classification](https://arxiv.org/pdf/1511.08630.pdf)
* TextRCNN AAAI 2015 [Recurrent Convolutional Neural Networks for Text Classification](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745/9552)
* TextRNN IJCAI 2016 [Recurrent Neural Network for Text Classification with Multi-Task Learning](https://arxiv.org/pdf/1605.05101.pdf)
* TextHAN ACL 2016 [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf)
* DMN ICML 2016 [Ask Me Anything: Dynamic Memory Networks for Natural Language Processing](https://arxiv.org/pdf/1506.07285.pdf)
* EntNet ICLR 2017 [TRACKING THE WORLD STATE WITH RECURRENT ENTITY NETWORKS](https://arxiv.org/pdf/1612.03969.pdf)
* DGCNN ICML 2017 [Convolutional Sequence to Sequence Learning](https://arxiv.org/pdf/1705.03122.pdf)
* DPCNN ACL 2017 [Deep Pyramid Convolutional Neural Networks for Text Categorization](https://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf)
* Transformer NIPS 2017 [Attention Is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)
* RNN-Capsule WWW 2018 [Sentiment Analysis by Capsules](https://dl.acm.org/doi/pdf/10.1145/3178876.3186015)
* TextGCN AAAI 2019 [Graph Convolutional Networks for Text Classification](https://arxiv.org/pdf/1809.05679.pdf)
* Enhanced-RCNN WWW 2020 [Enhanced-RCNN: An Efficient Method for Learning Sentence Similarity](https://dl.acm.org/doi/abs/10.1145/3366423.3379998)
### Discussion
* LSTM vs GRU NIPS Workshop 2014 [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/pdf/1412.3555.pdf)
* CNN Hyperparameters arXiv 2015 [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1510.03820.pdf)
* CNN vs RNN arXiv 2017 [Comparative Study of CNN and RNN for Natural Language Processing](https://arxiv.org/pdf/1803.01271.pdf)
* Generative vs Discriminative arXiv 2017 [Generative and Discriminative Text Classification with Recurrent Neural Networks](https://arxiv.org/pdf/1703.01898.pdf)
  
## Embedding
### [Text Embedding](./Notes/NLP/TextEmbedding.md)
#### Word Embedding
* LSA(LSI), Journal of the American Society for Information Science, 1990. [Indexing by Latent Semantic Analysis](http://lsa.colorado.edu/papers/JASIS.lsi.90.pdf)
* NNLM JMLR 2003 [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) 
* RNNLM InterSpeech 2010 [Recurrent neural network based language model](https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)   
* Huang Model ACL 2012 [Improving Word Representations via Global Context and Multiple Word Prototypes](https://nlp.stanford.edu/pubs/HuangACL12.pdf)  
* word2vec Architecture ICLR 2013 [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf) 
* word2vec Tricks NIPS 2013 [Distributed Representations of Words and Phrases and their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) 
    * [code](https://github.com/tmikolov/word2vec) 
* GloVe EMNLP 2014 [GloVe: Global Vectors for Word Representation](https://www.aclweb.org/anthology/D14-1162.pdf)  
    * [code](https://github.com/stanfordnlp/GloVe)
* MSSG EMNLP 2014 [Efficient Non-parametric Estimation of Multiple Embeddings per Word in Vector Space](https://www.aclweb.org/anthology/D14-1113.pdf)
* char2wordvec EMNLP 2015 [Finding Function in Form: Compositional Character Models for Open Vocabulary Word Representation](https://arxiv.org/pdf/1508.02096.pdf)    
#### Sentence Embedding
* RAE NIPS 2011 [Dynamic Pooling and Unfolding Recursive Autoencoders for Paraphrase Detection](https://papers.nips.cc/paper/4204-dynamic-pooling-and-unfolding-recursive-autoencoders-for-paraphrase-detection.pdf)   
* MV-RNN EMNLP 2012 [Semantic Compositionality through Recursive Matrix-Vector Spaces](https://ai.stanford.edu/~ang/papers/emnlp12-SemanticCompositionalityRecursiveMatrixVectorSpaces.pdf)  
* TextCNN ACL 2014 [A Convolutional Neural Network for Modelling Sentences](http://mirror.aclweb.org/acl2014/P14-1/pdf/P14-1062.pdf) 
* Skip-Thought NIPS 2015 [Skip-Thought Vectors](https://papers.nips.cc/paper/5950-skip-thought-vectors.pdf) 
* FastText EACL 2017 [Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf)
    * [code](https://github.com/facebookresearch/fastText)
* Quick-Thought ICLR 2018 [AN EFFICIENT FRAMEWORK FOR LEARNING SENTENCE REPRESENTATIONS](https://arxiv.org/pdf/1803.02893.pdf)    
#### Context Embedding
* context2vec CoNLL 2016 [context2vec: Learning Generic Context Embedding with Bidirectional LSTM](https://www.aclweb.org/anthology/K16-1006.pdf)  
#### Paragraph Embedding
* Doc2vec PMLR 2014 [Distributed Representations of Sentences and Documents](https://arxiv.org/pdf/1405.4053.pdf)   
#### Relation Embedding
* GloRE NAACL 2018 [Global Relation Embedding for Relation Extraction](https://www.aclweb.org/anthology/N18-1075.pdf)
#### Entity Embedding
* SEE AAAI 2018 [SEE: Syntax-aware Entity Embedding for Neural Relation Extraction](https://arxiv.org/pdf/1801.03603.pdf)

### [Knowledge Graph Embedding](.)
#### Framework:   
  * [PyTorch-BigGraph](https://github.com/facebookresearch/PyTorch-BigGraph)
#### Node (Vertex) Embedding
##### Translation Based
* TransE NIPS 2013 [Translating Embeddings for Modeling Multi-relational Data](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf)  
* TransH AAAI 2014 [Knowledge Graph Embedding by Translating on Hyperplanes](https://persagen.com/files/misc/wang2014knowledge.pdf) 
* TransR AAAI 2015 [Learning Entity and Relation Embeddings for Knowledge Graph Completion](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewFile/9571/9523) 
* TransD ACL 2015 [Knowledge Graph Embedding via Dynamic Mapping Matrix](https://www.aclweb.org/anthology/P15-1067.pdf) 
* TransG ACL 2016 [TransG : A Generative Model for Knowledge Graph Embedding](https://www.aclweb.org/anthology/P16-1219.pdf) 
##### Random Walk Based
* DeepWalk KDD 2014 [DeepWalk: Online Learning of Social Representations](https://arxiv.org/pdf/1403.6652.pdf)   
* LINE WWW 2015 [LINE: Large-scale Information Network Embedding](https://arxiv.org/pdf/1503.03578.pdf)  
* node2vec KDD 2016 [node2vec: Scalable Feature Learning for Networks](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf) 
* SDNE KDD 2016 [Structural Deep Network Embedding](https://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf)   
* GraphSAGE NIPS 2017 [Inductive Representation Learning on Large Graphs](https://arxiv.org/pdf/1706.02216.pdf)  
#### Graph Embedding
* graph2vec MLGWorkshop 2017 [graph2vec: Learning Distributed Representations of Graphs](https://arxiv.org/pdf/1707.05005.pdf) 
* metapath2vec KDD 2017 [metapath2vec: Scalable Representation Learning for Heterogeneous Networks](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf) 


## [Pre-trained Language Model]()
* [Pre-trained Models for Natural Language Processing: A Survey](https://arxiv.org/pdf/2003.08271.pdf)
* NNLM JMLR 2003 [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)   
* RNNLM InterSpeech 2010 [Recurrent neural network based language model](https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)
* ULMFit ACL 2018 [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/pdf/1801.06146.pdf)
* ELMO NAACL 2018 [Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf) 
* GPT 1.0 InterSpeech 2018 [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)  
* BERT InterSpeech 2018 [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)  
* Transformer-XL ACL 2019 [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/pdf/1901.02860.pdf)   
* GPT 2.0 OpenAI Report 2019 [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) [Slides](https://pdfs.semanticscholar.org/41f9/45f59bd0d345d4e355fb72110524f6fdffdb.pdf)   
* ERNIE (Tsinghua) ACL 2019 [ERNIE: Enhanced Language Representation with Informative Entities](https://arxiv.org/pdf/1905.07129.pdf)   
* ERNIE 1.0 arXiv 2019 [ERNIE: Enhanced Representation through Knowledge Integration](https://arxiv.org/pdf/1904.09223.pdf)      
* MASS ICML 2019 [MASS: Masked Sequence to Sequence Pre-training for Language Generation](https://arxiv.org/pdf/1905.02450.pdf)  
* UniLM arXiv 2019 [Unified Language Model Pre-training for Natural Language Understanding and Generation](https://arxiv.org/pdf/1905.03197.pdf) 
* XLNet NIPS 2019 [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/pdf/1906.08237.pdf)  
* ERNIE 2.0 AAAI 2020 [ERNIE 2.0: A Continual Pre-Training Framework for Language Understanding](https://arxiv.org/pdf/1907.12412.pdf)
* RoBERTa ICLR 2020 [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf) 
* ALBERT ICLR 2020 [ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS](https://arxiv.org/pdf/1909.11942.pdf) [code](https://github.com/google-research/ALBERT)  
* ELECTRA ICLR 2020 [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/pdf/2003.10555.pdf) [code](https://github.com/google-research/electra)  

## Mechanism
### Attention
* origin attention ICML 2015[Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf)
* Transformer NIPS 2017 [Attention Is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) 
* Attention is not Explanation NAACL 2019 [Attention is not Explanation](https://arxiv.org/pdf/1902.10186.pdf)
* Attention is not not Explanation EMNLP 2019 [Attention is not not Explanation](https://arxiv.org/pdf/1908.04626.pdf)
### Memorization
* E2EMN NIPS 2015 [End-To-End Memory Networks](https://arxiv.org/pdf/1503.08895.pdf)
* DMN ICML 2016 [Ask Me Anything: Dynamic Memory Networks for Natural Language Processing](https://arxiv.org/pdf/1506.07285.pdf)

## Task Classification
### [Sequence Labeling]()
* BiLSTM-CRF arXiv 2015 [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/pdf/1508.01991.pdf)  
* CNN-BiLSTM-CRF arXiv 2016 [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://arxiv.org/pdf/1603.01354.pdf) 
* Lattice LSTM ACL 2018 [Chinese NER Using Lattice LSTM](https://www.aclweb.org/anthology/P18-1144.pdf)

### [Text Classification]()
* TF-IDF
* LDA主题模型
* LSI主题模型
* CNN ACL 2014 [Convolutional Neural Networks for Sentence Classification](https://www.aclweb.org/anthology/D14-1181.pdf)
* char CNN NIPS 2015 [Character-level Convolutional Networks for Text Classification](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)
* HAN ACL 2016[Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf)
* FastText EACL 2017 [Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf)
* TextGCN AAAI 2019 [Graph Convolutional Networks for Text Classification](https://arxiv.org/pdf/1809.05679.pdf)

### [Sentence Pair Relationship]()
#### Sentence Similarity
* SV-DSSM CIKM 2013 [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data](https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf)
* SV-CNN-DSSM CIKM 2014 [A Latent Semantic Model with Convolutional-Pooling Structure for Information Retrieval](https://www.iro.umontreal.ca/~lisa/pointeurs/ir0895-he-2.pdf)
* SV-LSTM-DSSM IEEE/ACM Transactions on Audio, Speech, and Language Processing 2016 [Deep Sentence Embedding Using Long Short-Term Memory Networks: Analysis and Application to Information Retrieval](https://arxiv.org/pdf/1502.06922.pdf)
* MV-DSSM IW3C2 2015 [A Multi-View Deep Learning Approach for Cross Domain User Modeling in Recommendation Systems](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/frp1159-songA.pdf)
* Siamese RNN ACL 2016 [Learning Text Similarity with Siamese Recurrent Networks](https://www.aclweb.org/anthology/W16-1617.pdf)
#### Natural Language Inference
* ESIM ACL 2017 [Enhanced LSTM for Natural Language Inference](https://arxiv.org/pdf/1609.06038.pdf)

### [Generalization]()
#### Machine Translation
* BLEU ACL 2002 [BLEU: a Method for Automatic Evaluation of Machine Translation](https://www.aclweb.org/anthology/P02-1040.pdf)
* EMNLP 2014 [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://www.aclweb.org/anthology/D14-1179)  
* DeepNMT NIPS 2014 [Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) 
* AttNMT EMNLP 2014 [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)  
* GNMT arXiv 2016 [Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/pdf/1609.08144.pdf)
* ByteNet arXiv 2017 [Neural Machine Translation in Linear Time](https://arxiv.org/pdf/1610.10099v1.pdf) [Slides](http://llcao.net/cu-deeplearning17/pp/class8_TranslationinLinearTime.pdf) 
#### Text Summarization
* EMNLP 2019 [Text Summarization with Pretrained Encoders](https://arxiv.org/pdf/1908.08345.pdf)
 
# CV
## [Origin and Pioneer](.)
* CAT'S VISUAL CORTEX, J Physiol 1962. []
* Neocognitron, Kunihiko Fukushima 1980. [Neocognitron: A Self-Organizing Neural Network Model for a Mechanism of Visual Pattern Recognition](https://www.semanticscholar.org/paper/Neocognitron%3A-A-Self-Organizing-Neural-Network-for-Fukushima-Miyake/9b2541b8d8ca872149b4dabd2ccdc0cacc46ebf5) 
  * [Kunihiko Fukushima](http://personalpage.flsi.or.jp/fukushima/index-e.html)
  
## CV Backbone
### Image Recoginization
* LeNet-5, Lecun, IEEE 1998. [GradientBased Learning Applied to Document Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
* AlexNet, NIPS 2012. [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
* GoogLeNet, CVPR 2015. [Going deeper with convolutions](https://arxiv.org/pdf/1409.4842.pdf)
* VGG-16, ICLR 2015. [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)
* ResNet, CVPR 2016. [Deep Residual Learning for Image Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
* DenseNet, CVPR 2017. []
* SE-Net, CVPR 2017.
* EfficientNet
### Image Segmentation
### Object Detection
* YOLOv1 CVPR 2016 [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf) 
* YOLOv2 CVPR 2017 [YOLO9000: Better, Faster, Stronger](https://arxiv.org/pdf/1612.08242.pdf)
* YOLOv3 Tech report 2018 [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
* YOLOv4 CVPR 2020 [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/pdf/2004.10934.pdf)
  * [DarkNet](https://pjreddie.com/darknet/)
* YOLOv5 No paper released, only code provided. 2020 [code](https://github.com/ultralytics/yolov5)

# Reinforcement Learning
* 值函数 动作函数 状态函数
* Sara, Q-Learning， DQN， 
* AlphaGo Nature 2016 [Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961)
* AlphaZero arXiv 2017 [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/pdf/1712.01815.pdf)

# General Elements
## Activation Function
* ReLU
* ELU
* GELU
## Loss Function
* Focal Loss CVPR 2017 [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf)
* Dice Loss ACL 2020 [Dice Loss for Data-imbalanced NLP Tasks](https://www.aclweb.org/anthology/2020.acl-main.45.pdf)
* Triplet Loss
* Quadruplet Loss 
## Optimizer
* SGD
* RMSProp
* Momentum
* Adam
* Radam
## Training Tricks
### Initialization
* Xavier Initialization, AISTATS 2010. [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
* Kaiming Initialization, ICCV 2015. [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852.pdf)
### Middle Layer Operation
* Batch Normalization, ICML 2015. [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)  
* Batch Normalization Explanation, NIPS 2018. [How Does Batch Normalization Help Optimization?](http://papers.nips.cc/paper/7515-how-does-batch-normalization-help-optimization.pdf)  
* Layer Normalization, arXiv 2016. [Layer Normalization](https://arxiv.org/pdf/1607.06450v1.pdf)  
* Instance Normalization, arXiv 2016. [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/pdf/1607.08022.pdf)  
* Group Normalization, arXiv 2018. [Group Normalization](https://arxiv.org/pdf/1803.08494.pdf)  
### Out Layer Operation
* Dropout, JMLR 2014. [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)  
* Mixout, ICLR 2020. [MIXOUT: EFFECTIVE REGULARIZATION TO FINETUNE LARGE-SCALE PRETRAINED LANGUAGE MODELS](https://arxiv.org/pdf/1909.11299.pdf)  
* Disout, AAAI 2020. [Beyond Dropout: Feature Map Distortion to Regularize Deep Neural Networks](https://arxiv.org/pdf/2002.11022.pdf)  
### Gradient Operation
* Gradient Clipping, ICML 2013. [On the difficulty of training Recurrent Neural Networks](https://arxiv.org/pdf/1211.5063.pdf)
* Gradient Clipping Explanation, ICLR 2020. [WHY GRADIENT CLIPPING ACCELERATES TRAINING: A THEORETICAL JUSTIFICATION FOR ADAPTIVITY](https://arxiv.org/pdf/1905.11881.pdf)

## [Graph Neural Network](.)
### Framework:  
  * [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) 
### Survey
* arXiv 2018[Graph Neural Networks: A Review of Methods and Applications](https://arxiv.org/pdf/1812.08434.pdf)  
### Spatial Domain
* GNN arXiv 2017 [A Generalization of Convolutional Neural Networks to Graph-Structured Data](https://arxiv.org/pdf/1704.08165.pdf)  
* GraphSAGE NIPS 2017 [Inductive Representation Learning on Large Graphs](https://arxiv.org/pdf/1706.02216.pdf)  
* GAT ICLR 2018 [GRAPH ATTENTION NETWORKS](https://arxiv.org/pdf/1710.10903.pdf)  
* PGC AAAI 2018 [Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition](https://arxiv.org/pdf/1801.07455.pdf)   
* GIN ICLR 2019 [HOW POWERFUL ARE GRAPH NEURAL NETWORKS?](https://arxiv.org/pdf/1810.00826.pdf)  [
### Spectral Domain
* SCNN ICLR 2014 [Spectral Networks and Deep Locally Connected Networks on Graphs](https://arxiv.org/pdf/1312.6203.pdf)    
* ChebNet NIPS 2016 [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/pdf/1606.09375.pdf)  
* GCN ICLR 2017 [SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS](https://arxiv.org/pdf/1609.02907.pdf)   


# [Ensemble Learning](.)
## Bagging
## Boosting 
* XGBoost SIGKDD 2016 [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/pdf/1603.02754.pdf)
  * [Documentation](https://xgboost.readthedocs.io/en/latest/index.html) [Slides](https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf)
  * [Code](https://github.com/dmlc/xgboost)
* LightGBM NIPS 2017 [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf)
  * [Documentation](https://lightgbm.readthedocs.io/en/latest/index.html)
  * [Code](https://github.com/microsoft/LightGBM)
* CatBoost Yandex 2017 [CatBoost: unbiased boosting with categorical features](https://arxiv.org/pdf/1706.09516.pdf)
  * [Documentation](https://catboost.ai/docs/concepts/about.html)
  * [Code](https://github.com/catboost/catboost)
* [XGBoost, LightGBM or CatBoost — which boosting algorithm should I use?](https://medium.com/riskified-technology/xgboost-lightgbm-or-catboost-which-boosting-algorithm-should-i-use-e7fda7bb36bc)
## Stacking

# System
* GFS SOSP 2003[The Google File System](https://static.googleusercontent.com/media/research.google.com/zh-CN//archive/gfs-sosp2003.pdf)
* MapReduce OSDI 2004 [MapReduce: Simplified Data Processing on Large Clusters](https://static.googleusercontent.com/media/research.google.com/zh-CN//archive/mapreduce-osdi04.pdf)
* Bigtable OSDI 2006 [Bigtable: A Distributed Storage System for Structured Data](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/68a74a85e1662fe02ff3967497f31fda7f32225c.pdf)


# Useful Resources
## Toolkits
* [THULAC](http://thulac.thunlp.org/) by 清华大学自然语言处理与社会人文计算实验室
  * 一个高效的中文词法分析工具包
  * 支持 c++、java、python
* [NLPIR2016](http://ictclas.nlpir.org/) by 张华平
  * 汉语分词系统: 分词标注、实体抽取、 词频统计、关键词提取、Word2vec、文本分类、情感分析、依存文法、繁简编码转换、自动注音、摘要提取
  * 支持 c++、java、 python
* [LTP](http://ltp.ai/) by 哈工大社会计算与信息检索研究中心
  * 分词、词性标注、句法分析
  * 支持 c++、windows、linux、mac
* [CRFsuite](http://www.chokkan.org/software/crfsuite/) by Naoaki Okazaki
  * A fast implementation of Conditional Random Fields(CRFs)
  * 支持 c
* [fastNLP](https://github.com/FudanNLP/fnlp) by 复旦大学自然语言处理组
  * 信息检索、中文处理、结构化学习
  * 支持 python
* [hanNLP](https://github.com/hankcs/HanLP) by hankcs
  * 中文分词、命名实体识别、关键词提取、自动摘要、短语提取、拼音转换、 简繁转换、文本推荐、依存句法分析
  * 支持 Java
* [OpenNLP](https://github.com/apache/opennlp) by Apache
  * A machine learning based tookit for the processing of natural language text
  * 支持 java
* [Stanford CoreNLP](https://github.com/stanfordnlp/CoreNLP) by The Stanford Natural Language Processing Group
  * 集成了许多斯坦福的 NLP 工具，包括:词性标记、命名实体识别、语法、参数分析系统、情绪系统、自举模式学习、开放信息提取工具
  * 支持 java
* [LingPipe](http://www.alias-i.com/lingpipe/) by alias 公司
  * 命名实体识别、聚类、词性标注、情感分类、句子检测、瓶邪检测、字符串匹配、关键词抽取、数据库文本挖掘、 断字识音、语言种类识别、词义消歧
  * 支持 java
* [MALLET](http://mallet.cs.umass.edu/) by UMASS AMHERST
  * Machine Learning for Language toolkit, 提供统计学自然语言处理、 文档分类、聚类、主题建模、信息提 取和其它机器学习文本应用
  * 支持 java
* [CRF++](http://taku910.github.io/crfpp/) by Taku Kudo
  * 序列标注
  * 支持 c++
* [HTK](http://htk.eng.cam.ac.uk/) by 剑桥大学工程学院
  * 基于 HMM 模型的语音处理工具
  * 支持 c++
* [gensim](https://radimrehurek.com/gensim/) by open-source
  * 分词、关键词提取、词性标注、并行分词、word2vec
  * 支持 python
* [jieba](https://github.com/fxsjy/jieba) by Sun Junyi
  * 分词软件
  * 支持 python、java、c++ 
* [Ansj](https://github.com/NLPchina/ansj_seg) by 中国自然语言处理开源组织
  * 中文分词、中文姓名识别、用户自定义词典、关键字提取、自动摘要、关键字标记 
  * 支持 java
* [NLTK](https://www.nltk.org/)
  * 是一套基于 python 的自然语言处理工具集
  * 支持 python
* [FoolNLTK](https://github.com/rockyzhengwu/FoolNLTK)
  * 中文处理工具包
  * 支持 python
* [pkuseg](https://github.com/lancopku/pkuseg-python)
  * 一个多领域中文分词工具包
  * 支持 python
* [TextBlob](https://github.com/sloria/TextBlob) by Steven Loria
  * 情感分析、词性标注、翻译等
* [Spacy](https://github.com/explosion/spaCy) by Explosion AI
  * 命名实体识别、序列标记、功能强大、支持二十多种语言
  * 支持 python、cpython
* [word2vec](https://code.google.com/archive/p/word2vec/) by Mikolov(Google)
  * 词向量训练工具  
  * 支持 c
* [fastText](https://github.com/facebookresearch/fastText) by Mikolov(FAIR)
  * 词向量训练工具
  * 支持 c++ 
* [glove](https://nlp.stanford.edu/projects/glove/) by Stanford
  * 词向量训练工具
  * 支持 c++ 

## Resources
* [中文常用停用词表](https://github.com/goto456/stopwords)
* [Stopwords for 50 languages in JSON format](https://github.com/6/stopwords-json)

## Terms Explanation
* [Turing Test](http://www.wikiwand.com/en/Turing_test)
* [Distributional hypothesis](https://en.wikipedia.org/wiki/Distributional_semantics#Distributional_hypothesis)
* [Distributional Representation vs Distributed Representation](https://zhuanlan.zhihu.com/p/70767034)
* [Distributed Representation vs Local Representation](https://deepai.org/machine-learning-glossary-and-terms/distributed-representation)
* [Google Machine Learning Glossary](https://developers.google.cn/machine-learning/glossary)


## Articles
* [常用的 Normalization 方法：BN、LN、IN、GN](https://www.chainnews.com/articles/678463364097.htm)
* [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* [Understanding GRU Networks](https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be)
* [A Gentle Introduction to Graph Embeddings](https://medium.com/towards-artificial-intelligence/a-gentle-introduction-to-graph-embeddings-c7b3d1db0fa8)
* [The mostly complete chart of Neural Networks, explained](https://towardsdatascience.com/the-mostly-complete-chart-of-neural-networks-explained-3fb6f2367464)

## Books
* [机器学习周志华](https://github.com/Mikoto10032/DeepLearning/blob/master/books/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%91%A8%E5%BF%97%E5%8D%8E.pdf)
* [深度学习花书](https://github.com/exacity/deeplearningbook-chinese)
* [Dive into Deep Learning](http://d2l.ai/)
* [可解释的机器学习--黑盒模型可解释性理解指南](https://github.com/MingchaoZhu/InterpretableMLBook)
* [Neural Network Methods for NLP - Yoav Goldberg](./Books/NeuralNetworkMethods4NLP.pdf)
* [Introduction to Graph Neural Network](./Books/Intro2GNNs.pdf)
* [Deep Learning in Natural Language Processing]()

## Course
* [吴恩达深度学习](https://www.deeplearning.ai/)
* [李宏毅2020机器学习深度学习(完整版)国语](https://www.bilibili.com/video/BV1JE411g7XF?from=search&seid=17396144567049451218)
* [李宏毅《深度学习人类语言处理》国语(2020)](https://www.bilibili.com/video/av94310884)
* [STAT 157 at UC Berkeley](https://courses.d2l.ai/berkeley-stat-157/syllabus.html) 
  * [video](https://www.youtube.com/playlist?list=PLZSO_6-bSqHQHBCoGaObUljoXAyyqhpFW)
* [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
* [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/)
* [CS224U: Natural Language Understanding](http://web.stanford.edu/class/cs224u/)
* [CS520: Knowledge Graphs](https://web.stanford.edu/class/cs520/)

## High Quality Paper Websites
* [PaperswithCode](https://paperswithcode.com/)
* [SOTA.ai](https://www.stateoftheart.ai/)

