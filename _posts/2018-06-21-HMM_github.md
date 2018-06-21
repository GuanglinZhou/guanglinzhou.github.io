---
layout:     post
title:      "HMM"
subtitle:   "隐马尔科夫模型  维比特算法"
date:       2018-06-21 08:53:00
author:     "guanglinzhou"
header-img: "img/post-bg-alitrip.jpg"
comments: true
tags:
    - ML
    - NLP
---

<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>

###  隐马尔科夫模型


----------
很久没有更新博客了，这两个月忙着做个地图匹配的实验，顺便要发论文，压力山大，最近导师接了个交通知识图谱的活儿，这不，又得开始做些调研，发现项目中用到的实体对齐，NER等需要很多NLP的知识点，问了些搞NLP的同学，先学学基础的HMM、CRF等概率图模型，再接触深度学习吧。
说来也巧，HMM也有用在地图匹配中的。


##### 项目地址：[Github—ML_Python，感兴趣的给个star！](https://github.com/GuanglinZhou/ML_Python/tree/master/HMM)

----------
#### 马尔科夫过程
一阶马尔科夫过程定义为：假设某一时刻状态转移的概率只依赖于它的前一个状态，比如每天的天气举例，今天的天气只依赖于昨天的天气，和前天的天气没有关系。
数学定义为：
假设序列状态为$$...,X_{t-2},X_{t-1},X_t,X_{t+1},...$$，则在时刻$$t+1$$的状态只依赖于时刻$$t$$，即

$$
	P(X_{t+1}\ |\ ...,X_{t-2},X_{t-1},X_t)=P(X_{t+1}\ |\ X_t)
$$


----------
#### 隐马尔科夫模型HMM
**隐马尔科夫模型描述由一个隐藏的马尔科夫链随机生成不可观测的状态随机序列，再由各个状态生成一个观测而产生观测随机序列的过程。**
隐马尔科夫模型广泛应用于自然语言处理中，比如词性标注，命名实体识别等问题。
NLP领域中经典的语言迷行N-gram也和马尔科夫有关系，N-gram模型对应的是N-1阶马尔科夫过程。

HMM的基本组成为3+2。

三个概率矩阵：
- 状态转移概率矩阵$$A$$：状态之间的转移概率矩阵；
- 观测概率矩阵$$B$$：状态产生观测的概率矩阵；
- 初始概率矩阵$$\pi$$：各状态的初始概率；

两个集合：
- 所有可能的状态集合$$I$$；
- 所有可能的观测集合$$O$$；


----------
HMM属于生成式模型，从之前的博客[逻辑回归更新篇](http://www.jameszhou.tech/2018/03/25/%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E6%9B%B4%E6%96%B0%E7%AF%87_github/)中，我们可知，生成式模型是通过联合概率求解

$$
P(c\ |\ x)=\frac{P(x,c)}{P(x)}
$$

![](https://ws1.sinaimg.cn/large/006tKfTcgy1fsitcarnovj30cn07t0su.jpg)

----------


HMM在NLP领域使用的比较多的是HMM的第三个问题，即解码问题，已知HMM模型
$$
\lambda=(A,B,\pi)
$$
和观测集合$$V$$，求对给定观测序列条件概率$$P(I\ |\ O)$$最大的状态序列$$I=(i_1,i_2,...,i_T)$$，即给定观测序列，求最有可能的对应的状态序列。


----------

举例来说，目前有3个盒子，每个盒子中有红，白两种球，每次随机的在某个盒子中选择了某个球，观测序列为$$O=(红，白，红)$$，HMM模型参数为：

$$A=\begin{bmatrix} 0.5 & 0.2 & 0.3 \\ 0.3 & 0.5 & 0.2 \\ 0.2 & 0.3 & 0.5 \end{bmatrix}$$

$$B=\begin{bmatrix} 0.5 & 0.5 \\ 0.4 & 0.6\\ 0.7 & 0.3 \end{bmatrix}$$

$$\pi=(0.2,0.4,0.4)$$


求最优状态序列，即最有可能的盒子顺序。
给盒子编号为（1、2、3），每个盒子中都有（红，白）两类球。
则，第一种试试暴力法来解，

$$\begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 2 \\ 1 & 1 & 3 \\ \vdots & \vdots & \vdots \\  3 & 3 & 3\end{bmatrix}$$

总共27种方案，分别求其概率，取最大概率对应的状态序列。
我们可以看出，暴力法求解的可能性为$$|I|^{|O|}$$，和观测序列的长度呈指数级增长。
维特比算法是求解HMM预测问题的最佳算法，实际使用动态规划求概率最大路径，这时一条路径对应着一个状态序列。

**Viterbi**算法根据动态规划原理，有一个特性，此特性保证了Viterbi算法求解出来的是概率最大的路径。即：
> 如果最优路径在时刻$$t$$通过节点$$i_t$$，那么这一路径从节点$$i_t$$到终点$$i_t$$的部分路径，对于从$$i_t$$到$$i_T$$的所有可能的部分路径来说，必须是最优的，否则从$$i_t$$到$$i_T$$就有另一条更好的部分路径存在，这矛盾了。
>根据这一原理，我们从时刻$$t=1$$开始，递推地计算在时刻$$t$$状态为$$i$$的各部分路径的最大概率，直至得到时刻$$t=T$$状态为$$i$$的各条路径的最大概率，如此得到了时刻$$t=T$$最大概率即为最优路径的概率$$P^*$$，最优路径的终结点也得到了，从终结点从后往前逐步求节点$$i_{T-1},...,i_1$$，得到了最优路径$$I=(i_1,i_2,...,i_T)$$

第一次看的时候，感觉很绕，说实话还是自己不耐心看，觉得比较难；细下心看，找个示例，实现之后觉得，想法很朴素。
先来看暴力法破解状态序列时，有什么可以优化的吗？比如，前两个情况——$$(1,1,1)和(1,1,2)$$这两种状态序列，第一种情况分别求了，初始情况为状态1观测为红色的概率——转移为状态1且观测为红色的概率——转移为状态1且观测为红色的概率；第二种情况求了，初始情况为状态1观测为红色的概率——转移为状态1且观测为红色的概率——转移为状态2且观测为红色的概率。可以看出，前两次求概率部分重复了，如果第一种情况能保存前两次概率，第二种情况的复杂度就降低了。

维特比算法也用到了这种优化，首先从初始状态$$t=1$$出发，分别计算各状态在观测1时的概率并保存下来，本例为（0.1，0.16，0.28）然后计算$$t=2$$时刻时，$$t=1$$时每个状态会向$$t=2$$时的每个状态转移，能得到转移概率以及结合$$t=2$$时的观测得到$$t=2$$时刻，某状态在观测2时的概率，以$$t=2$$时状态1，此时观测为白色举例，$$t=1$$时的三种状态{1,2,3}都会对$$t=2$$时刻的状态1有转移概率，结合状态1对观测为白色的发射概率，本例可以得到概率（0.025，0.024，0.028）,这是$$t=1$$时刻状态1、2、3到$$t=2$$时刻状态1且观测为白色的概率，可以发现$$t=1$$状态3过来的概率最大，所以记下$$t=2$$时刻状态1概率为0.028，且将$$t=2$$时刻状态1的前一节点标记为状态3。

简而言之，本次从$$t=1$$到$$t=T$$，从前往后计算，将当前状态与观测的最大对应概率，以及当前状态的前一个状态保存下来，当$$t=T$$找到最优概率时，往$$t=1$$回溯，依次读取之前保存的前一个状态即可得到最优状态序列。

上述例子用**Viterbi算法**的Python实现。
```python

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-06-19 20:13:04
# @Author  : guanglinzhou (xdzgl812@163.com)
# @Link    : https://github.com/GuanglinZhou
# @Version : $$Id$$

import numpy as np


# 需要为每层状态建立两个列表，分别是概率列表和最大概率对应的前一个节点的列表
def Viterbi(start_probability, transition_probability, emission_probability, observations):
    bestWay = []
    probList = []
    nodeList = []
    for t in range(len(observations)):
        prob = []
        node = []
        for i in range(len(states)):
            if (t == 0):
                prob.append(start_probability[i] * emission_probability[i][obseIndex[observations[t]]])
                node.append(0)
            else:
                prob.append(np.max(
                    np.array(probList[t - 1]) * np.array(transition_probability)[:, i] * emission_probability[i][
                        obseIndex[observations[t]]]))
                node.append(np.argmax(np.array(probList[t - 1]) * np.array(transition_probability)[:, i]))
        probList.append(prob)
        nodeList.append(node)
    for t in reversed(range(len(observations))):
        if (t == len(observations) - 1):
            bestWay.append(np.argmax(probList[t]))
        else:
            bestWay.append(nodeList[t + 1][bestWay[t - 1]])
    bestWay = list(reversed(bestWay))
    statusList = [states[i] for i in bestWay]
    print(statusList)
    return statusList


if __name__ == '__main__':
    states = ('1', '2', '3')

    observations = ('红', '白', '红')
    obseIndex = {'红': 0, '白': 1}
    start_probability = [0.2, 0.4, 0.4]

    transition_probability = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    emission_probability = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
    Viterbi(start_probability, transition_probability, emission_probability, observations)


```


#### 参考文献

1、[语言模型](https://blog.csdn.net/lanxu_yy/article/details/29918015)

2、[标注问题与隐马尔科夫模型](https://blog.csdn.net/lanxu_yy/article/details/36245161)

3、[课件](http://www.cs.columbia.edu/~mcollins/cs4705-spring2018/slides/tagging.pdf)
