---
layout:     post
title:      "Statistical Language Model"
subtitle:   "N-Gram  NNLM  word2vec"
date:       2018-06-23 08:53:00
author:     "guanglinzhou"
header-img: "img/post-bg-alitrip.jpg"
comments: true
tags:
    - ML
    - NLP
---

<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>


### Statistical Language Model



----------
本篇博客对统计语言模型做个简述，列举觉得不错的博客和论文，以及自己的一些补充，作为NLP 语言模型和词向量的入门篇。


----------

强烈推荐首先看这三篇博客：

- [word2vec前世今生](https://www.cnblogs.com/iloveai/p/word2vec.html)
- [浅谈词向量](http://xiaosheng.me/2017/06/08/article69/)
- [NLP：Language Model](https://www.cnblogs.com/taojake-ML/p/6413715.html)


----------


以上三篇博客讲解真的非常好了，我做一些自己遇到的问题及解决方法补充吧。

##### 问题
NLP的常见的问题：如何计算一段文本序列在某种语言出现的概率？
统计语言模型对这类问题提供了一个基本解决框架，对于文本$$S=w_1,w_2,...,w_{t-1}$$，它的概率可以表示为：

$$P(S)=P(w_1,w_2,..,w_T)=\prod_{t=1}^{T}p(w_t\ |\ w_1,w_2,...,w_{t-1})$$

即文本中各个Word按顺序出现的联合概率，按照全概率公式可以转换为一系列条件概率的乘积。

如果直接按全概率公式求解联合概率的话，会出现参数空间过大的情况。（一开始没明白参数多少如何判断的，所以列出来说一下）

举例来看：以搜索引擎或者翻译举例，
统计语言模型将一段文本的概率表示上述条件概率的乘积，那么模型的参数就是所有的条件概率，比如计算$$P(w_5\ |\ w_4,w_3,w_2,w_1)$$，词典大小为$$|W|$$，那么每个$$w_i$$都有$$|V|$$种取值可能，则该条件概率的参数个数有$$|V|^5$$个，所以，如此通过全概率公式将联合概率转换为条件概率连乘，模型的参数个数过多。
所以在此基础上通过马尔科夫过程思想，将条件概率只与前一个word有关，这就是BiGram，即1一阶马尔科夫过程，由此引申了N-Gram模型，一般$$N\le3$$


##### 词表示
无论是英文还是中文NLP问题，语料中的word都是字符串形式，无法被计算机识别，ML模型的输入都是数值型的数据，那么Word在计算机中如何表示呢？

**1、**N-Gram模型，其中word的表示方法为One-Hot方式，将word表示为稀疏的的向量，向量大小为整个词典的大小，比如对于一句话**“I have a dog”**，那么dog这个单词的向量表示为**“0 0 0 1”**，可以看出，当语料中包含的词很多时，每个word所表示的向量是大而稀疏的，word之间的相似度为0；同时数据稀疏严重即有限的训练预料会造成联合概率为0，造成维度灾难；

**2、**以神经网络语言模型作为代表，其将Word表示为密集的向量，也称为**distributed representation for words**，此方法可以将word映射为任意维度的向量，同时可以发现word之间的相似性等特点，与当前主流的深度学习方法相结合，是目前的主流方法。


----------
#### NNLM
神经网络运用于语言模型是在[A Neural Probabilistic Language Model 2003](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)这篇论文中。
通过**word distributed representations**可以解决维度灾难这个问题
![](https://ws4.sinaimg.cn/large/006tNc79ly1fsli8cz50wj30re08rt95.jpg)


神经网络模型主要做了两件事：
![](https://ws4.sinaimg.cn/large/006tNc79ly1fslieqydxfj30rk09lwgl.jpg)
神经网络模型的结构图：
![](https://ws4.sinaimg.cn/large/006tNc79ly1fslie78jxuj30iw0ggq4b.jpg)

简而言之：

线性映射，相当于没有非线性转换的**Hidden Layer**，将$$word$$（one-hot编码的形式），使用一个**共享的矩阵$$C$$**，表示为人为设定维度大小的特征向量($$distributed vector$$)，当然矩阵的内容作为参数通过训练阶段调整，特征向量的表示方法将每个word映射为向量空间中的一个点，特征的维度可以认为设定（论文中特征向量为30、60、100，远远小于词汇集的大小17000）。

后面就是正常的**Feed-Forward Neural Network**，将特征向量作为神经网络的输入，神经网络的输出是词典中各个$$word$$的概率，相当于一个多分类，类别为词典大小。输出的第$$i$$个元素表示概率$$P(w_t=i\ |\ w_1^{t-1})$$，表示第$$i$$个word的概率。 一个词汇序列发生的联合概率表达为条件概率的连乘形式，实验中给定之前的词汇，使用多层神经网络来预测下一个word。如此概率函数的参数可以通过最大似然法迭代的调整。
word的特征向量初始化可以使用语义特征的先验知识，比如$$dog和cat初始向量比较相似$$。

具体示例结构图：
![](https://ws2.sinaimg.cn/large/006tNc79ly1fslifrx4cvj30e109x3zc.jpg)
**`这个训练过程通过MLE和SGD完成，模型的参数为矩阵C+神经网络的权重参数`**
MLE：
![](https://ws1.sinaimg.cn/large/006tNc79ly1fslipaip4zj30dz02a3yf.jpg)
SGD更新参数：
![](https://ws1.sinaimg.cn/large/006tNc79ly1fslipuozzqj30bh029dfq.jpg)

**模型训练后除了可以直接用于预测**word sequence**的概率，$$矩阵C$$也可以得到所有word对应的词向量。**

但是正如神经网络结构图所示中——$$Softmax$$层标记为**$$most\  computation\  here$$**，这里需要对词典中的所有word进行归一化计算出概率。


----------
#### word2vec
[这篇文章解释语言模型尤其是word2vec两个模型，简直超赞！！！清晰的解释了CBOW和Skip-Gram模型以及hierarchical softmax和negative sampling方法](http://www.cnblogs.com/peghoty/p/3857839.html)

NNLM模型计算复杂度表示为：

$$Q=N*D+N*D*H+H*V$$
原本复杂度最大的在于$$H*V$$，即隐层到输出层$$Softmax$$过程，但是，一些方法如$$hierarchical \ softmax$$、避免完全正则化、词汇的二叉树表示等等，减少了这部分复杂度。
因此，大部分复杂度是由$$N*D*H$$造成的，即projection-hidden layer。
本文使用$$hierarchical \ softmax$$优化隐层和输出层之间的复杂度；正如上文所说，模型的计算瓶颈为$$N*D*H$$，文章去除了隐藏层，因此复杂度基本取决于$$Softmax \ normalization$$


----------
因为不是专科搞NLP的，现在也没有看透**word2vec**源码的时间，对**word2vec**的了解止步于上述链接。
知识图谱算是NLP的一个分支，至此，对词向量的表示和语言模型有了基本的认识，也对KG中实体映射为向量的方式有了了解，KG中实体映射为向量也使用了negative sampling方法，构造错误的三元组再通过SGD更新参数，这部分可以看**TransE等模型**。


**Word2vec**的源码还是非常值得花时间来看的，这里留个**`TODO`**吧。
对**word vector 和 language model**有了了解，就进行下一章。
