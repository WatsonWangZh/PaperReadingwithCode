# WordEmbedding
## LSA(LSI)
* 3个步骤:
  * 1. 选取上下文(词-文档矩阵， 词-词矩阵， n元词组)
  * 2. 确定矩阵中元素的值(词频， TF-IDF， PMI)
  * 3. 矩阵分解(高维稀疏->低维稠密)
    * 奇异值分解 (SVD)
    * 非负矩阵分解 (NMF)
    * 典型关联分析 (Canonical Correlation Analysis，CCA) 
    * Hellinger PCA(HPCA)