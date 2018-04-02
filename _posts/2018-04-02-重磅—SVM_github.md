---
layout:     post
title:      "重磅——SVM"
subtitle:   "SVM推导 SMO算法 核函数 非线性逻辑回归"
date:       2018-04-01 08:00:00
author:     "guanglinzhou"
header-img: "img/post-bg-alitrip.jpg"
comments: true
tags:
    - 算法
    - ML
---

<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>

### 重磅—SVM


----------
本篇博客主要内容为：
- SVM公式推导；
- SMO公式推导；
-  Python实现SMO和SVM，两种核（linear和RBF）；
-  逻辑斯蒂回归实现非线性分类；（有待更新....）

`由于SVM公式比较多，《统计学习方法》这本书做了详细的讲解，我结合原作者的论文和《统计学习方法》以及周志华老师的《机器学习》做个个人总结，便于之后的复习。对于过多的公式不做赘述，着重介绍自己在学习SVM时遇到的一些绊脚石，大家或多或少都遇到了，如果能帮助解决一两个小疑惑那是再好不过了，本人才疏学浅，如果有错误疏漏的地方，还请在评论指出。`

##### 项目地址：[Github—ML_Python，感兴趣的给个star！](https://github.com/GuanglinZhou/ML_Python/tree/master/SVM)

----------
**题记：**
>说实话，整理SVM我的内心是拒绝的，因为难啊！但你要问它重不重要？我的标题都加“重磅”了，表明它非常重要啊，怎么看出的？在图像分类方面，一直都是SVM的表现最好，直到Hinton团队提出的深度学习碾压了SVM误差率，才开启了深度学习的春天。再次，看《统计学习方法》这本书，薄薄的200页，11章内容，SVM一章就占了40页，无出其右，可见其内容繁多和重要性了。
>怎么办？硬着头皮上吧，难不成指望面试官不考？




----------

#### SVM公式推导
支持向量机和之前的逻辑回归同属于判别模型，即利用一个分离超平面去分隔样本。
与逻辑回归不同的是，SVM使用的是最大间隔超平面，能够保证很好的泛化能力。并且其具有强大的核技巧能够使得SVM应用于非线性分类中，事实上，逻辑斯蒂回归也可以使用类似的想法实现非线性分类，我们在博客的最后也会讲解和实现。


----------
`支持向量机实现了以下想法：`**它通过一些非线性映射方法将输入向量从输入空间$$X$$映射到高维的特征空间$$Z$$，在特征空间中，构造了一个线性决策平面，并且其有些特性能够保证支持向量机的高泛化能力。**

上述想法有两个问题需要解决：
- 如何找到一个分离超平面保证其具有良好的泛化能力，因为特征空间的维度非常大；
- 在高维特征空间如何计算的问题；



**针对第一个问题：**SVM中最优超平面定义为一个线性决策函数，其对两种类别的特征向量有最大间隔。
这种方法构造的超平面只需要考虑被称为**支持向量**的少数样本，即支持向量决定了间隔大小。

我们来用数学公式表述这种想法：
我们使用输入向量映射到特征空间后的特征向量为$$\vec{z}$$，则特征空间中的超平面可表示为（$$\omega是向量，b是标量$$）：

$$
\vec{\omega } \ \vec{z}+b=0
$$

由于我们首先推导线性可分支持向量机，此方法的输入空间不需要映射到特征空间（或者特征空间和输入空间相同），为了和书本保持一致，我们对超平面方程仍然使用：

$$
\vec{\omega } \ \vec{x}+b=0
$$

![](https://ws4.sinaimg.cn/large/006tKfTcgy1fpy8xu575aj30gj0bu40b.jpg)
如上图是一个线性可分的数据集，最大间隔的向量表示为间隔边界上两向量的差在超平面法向量的投影：

$$
marginMax=\frac{(\vec{x_2}-\vec{x_1})\cdot\vec{\omega}}{||\vec{\omega}||}
$$

结合超平面方程可得：

$$
marginMax=\frac{(1-b)-(-1-b)}{||\vec{\omega}||}=\frac{2}{||\vec{\omega}||}
$$

根据点到直线的距离可知,任何一个样本点到超平面的距离为：

$$
d=\frac{||\omega x_i+b||}{||\omega||}
$$

由$$y_i=+1或-1$$上式为

$$
d=\frac{y_i(\omega x_i+b)}{||\omega||}
$$

则我们得到了目标函数和约束条件：

$$\max \limits_{\omega ,b} \frac{2}{\parallel\omega\parallel}$$

$$s.t. \  \  \frac{y_i(\omega^Tx_i+b)}{||\omega||}\ge\frac{1}{||\omega||}$$

化简得：



$$\min \limits_{\omega ,b} \frac{1}{2}{\parallel\omega\parallel}^2$$

$$s.t. \ y_i(\omega^Tx_i+b)-1\ge0,\ \ \  i=1,2,...,N$$

至此我们得到了目标函数和约束条件的最终条件。（之前在这里推导比较困惑我的是`最大间隔`为什么是$$\frac{2}{\parallel\omega\parallel}$$，以及`某样本到超平面的距离`，所以我着重推导了一下，不是很难，其他基本和书上推导过程一致。）

目标函数得到了，是一个典型的凸二次规划问题。这是我们第一次遇到含有约束条件的优化问题，之前都是无约束的优化利用梯度法就可以求解了。
含有约束条件的凸优化问题，使用拉格朗日对偶来求解。
`对待拉格朗日对偶，判断好是在哪个变量下取得极值还是比较好理解的。`
拉格朗日对偶，是对于满足下式情况的约束最优化问题，构造原始问题的对偶问题，通过求解对偶问题，来得到原始问题的解。

$$
\min \limits_{x}f(x)$$

$$s.t. \ c_i(x)\le0,\ i=1,2,...,k
$$

$$\ h_j(x)=0,\ j=1,2,...,l
$$

构造拉格朗日函数

$$
L(x,\alpha,\beta)=f(x)+\sum_{i=1}^{k}\alpha_ic_i(x)+\sum_{j=1}^{l}\beta_jh_j(x)
$$

函数

$$\theta_P(x)=\max\limits_{\alpha,\beta}L(x,\alpha,\beta)$$

这里$$\theta_P(x)$$是拉格朗日函数$$L(x,\alpha,\beta)$$在变量$$\alpha,\beta$$上取得极大值($$f(x)$$相当于常数项)，经分析，当满足最初问题约束条件时，

$$\theta_P(x)=f(x)$$

最初的目标函数即$$f(x)$$在变量$$x$$上取极小值，则对应的是

$$\min \limits_{x}\theta_P(x)=\min \limits_{x}\max \limits_{\alpha,\beta}L(x,\alpha,\beta)$$

这就把原始问题表示为拉格朗日函数的极小极大问题了。
其对偶问题为拉格朗日问题的极大极小问题，同理推出：

$$\min \limits_{x}\theta_D(x)=\max \limits_{\alpha,\beta}\min \limits_{x}L(x,\alpha,\beta)$$

那么对偶问题的解和原始问题的解之间的关系是什么样呢？
使用$$d^*和p^*表示原始问题和对偶问题的解$$
则

$$
d^*=\max \limits_{\alpha,\beta}\min \limits_{x}L(x,\alpha,\beta)\le\min \limits_{x}\max \limits_{\alpha,\beta}L(x,\alpha,\beta)=p^*
$$

说明，对偶问题的解是原始问题解的下界，这称为**weak duality**—弱对偶性，可以通过求解对偶问题得到原始问题最优解的下界估计。
当问题满足`KKT条件`时，得到了强对偶性，此时对偶问题的解和原始问题的解相等。

KKT条件中有个重要的等式（**对偶互补条件**）即

$$
\alpha_i(y_i(\omega \cdot x_i+b)-1)=0
$$

此条件说明：
- $$\alpha_i=0$$则对应参数向量$$\omega$$第$$i$$维度为零，该样本对最终的分类决策函数没有影响；
- $$(y_i(\omega \cdot x_i+b)-1)=0$$，对应的样本为支持向量。

`则对最终分类函数有影响的仅是支持向量这些样本。`

援引SVM论文中的一段话：
>实验显示，如果训练向量能够被最优超平面无错误的分离，其在测试集向量上的错误率满足下面不等式，其上界为支持向量数量/训练集向量总数。
![](https://ws1.sinaimg.cn/large/006tKfTcly1fpy9d18comj30ag01xa9z.jpg)
>从上式可以看出，其和特征空间的维度没有关系，只和少量的支持向量以及训练向量数量有关，说明支持向量机的泛化能力很高（甚至在无限维度空间中）。论文中实验证明，其错误率低至0.03，在数十亿维度的特征空间中仍然泛化的很好。


通过拉格朗日函数构造原始问题的对偶问题，可以很容易得到线性可分支持向量机，线性支持向量机的对偶问题。
这里列举线性可分支持向量机的对偶问题（线性可分支持向量机使其特殊情况）：


$$
\min \limits_{\alpha} \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_i \cdot x_j)-\sum_{i=1}^N\alpha_i
$$

$$
s.t. \sum_{i=1}^N\alpha_iy_i=0
$$

$$0\le\alpha_i\le C,i=1,2,...,N$$
得到了对偶问题的最优解$$\vec{\alpha}=(\alpha_1,\alpha_2,...,\alpha_N)$$

`我们看到对偶问题最优解是个向量，其维度和样本量是一样的，在样本量非常大的时候，求解非常慢，这也是后续SMO引出的原因。`
当问题满足KKT条件时，得到原始问题的解：

$$
\omega=\sum_{i=1}^{N}\alpha_iy_ix_i
$$

分类决策函数表达为：

$$
f(x_j)=sign(\sum_{i=1}^{N}\alpha_iy_i(x_i\cdot x_j)+b
$$

`到此解决了作者提出的第一个问题——找到这个分离超平面且保证模型具有良好的泛化性能，对于第二个问题——在高维空间中如何计算的问题。`
正如我们之前所表述的，对于线性可分和线性不可分的情况，一般我们不需要将输入向量映射到特征空间中即可求出分离超平面了（或者成为输入空间和特征空间相同）。但是实质上，线性超平面是存在于特征空间中的，所以上式最准确的表达式是：

$$
f(z_j)=sign(\sum_{i=1}^{N}\alpha_iy_i(z_i\cdot z_j)+b
$$

注意看上式的分类决策函数，决策函数内部，是将输入向量（$$x_j$$）和支持向量（$$x_i$$）先映射到高维的特征空间$$Z$$中，求点积，而特征空间的维度一般是非常大，很难计算。
如果在输入空间存在一个函数
$$K(x_1,x_2)=\phi(x_1)\cdot\phi(x_2)$$，它恰好等于高维空间中的这个内积，那么支持向量机就不用计算复杂的非线性变换后的内积了，大大简化了计算。
对于给定的核$$K(x_1,x_2)$$，特征空间$$H$$和映射函数$$\phi$$的取法不唯一，可以取不同的特征空间，即使在同一特征空间也可以取不同的映射。
一般使用的多项式核和高斯核，我们在代码中实现了高斯核（RBF）。

`作者一开始提出的两个问题（超平面的选取以及高维特征空间的计算问题）都已经有了相应的解决方式，下面就需要方法来求解对偶问题的解了。`

对偶问题的解$$\vec{\alpha}$$，其维度和样本数一样，当样本容量很大时，求解非常困难。
目前广泛使用的序列最小最优化算法来求对偶问题的解。

----------


#### SMO公式推导

SMO的思想：训练支持向量机需要解决非常大的二次规划优化问题（对偶问题的解$$\vec{\alpha}$$维度和样本量大小一样）。 SMO将这个大的二次规划问题分解为一系列尽可能小的二次规划问题。SMO所需的内存量对于训练集大小上是线性的，这允许SMO处理非常大的训练集。

SMO算法通过两层循环，外层循环选择第一个变量，内层循环选择第二个变量。

外层：
- 首先`一次迭代`所有的训练集，如果某个样本违反KKT条件，则将其作为第一个变量。（按照这样，迭代一次所有的训练样本）；
- `多次迭代`非边界的训练样本（$$\alpha非0和C$$），如果某个样本违反KKT条件，同样将其作为第一个变量。（按照这样，迭代多次非边界的训练样本），直到所有非边界的训练样本都符合KKT条件。

外层在上述两个策略中循环直到所有的训练样本在检验范围$$\epsilon$$(一般选择$$10^{-3}$$)内都符合KKT条件或者到达迭代次数为止。

内层：内层循环寻找的变量$$\alpha_2$$希望使得$$|E_1-E_2|$$最大（编程时将每个样本对应的$$E_i$$保存在列表中，便于快速查找）。
内层循环会出现些特殊情况导致目标函数无法有足够的下降（论文：if the above heuristic does not make positive progress）
我在程序中使用的是下列条件判断：
![](https://ws4.sinaimg.cn/large/006tKfTcgy1fpy8zmnackj30ap01jaa2.jpg)


那么内层会使用下列启发式规则寻找第二个变量：
- 迭代非边界的样本直到有足够的下降；
- 迭代全体样本直到有足够的下降；

`上述两次迭代都随机选择开始点`,如果仍没有足够的下降，则重新选择$$\alpha_1$$
阈值$$b$$的使用$$b_1和b_2$$的中点更新。

$$
b_1=E_1+y_1(\alpha_1^{new}-\alpha_1^{old})K(x_1,x_1)+y_2(\alpha_2^{new}-\alpha_2^{old})K(x_1,x_2)+b
$$

$$
b_2=E_2+y_1(\alpha_1^{new}-\alpha_1^{old})K(x_1,x_2)+y_2(\alpha_2^{new}-\alpha_2^{old})K(x_2,x_2)+b
$$

$$
b=\frac{b_1+b_2}{2}
$$

SMO算法按照上面的过程选择第一个变量和第二个变量，那么选择了这两个变量如何处理呢？（每次选择两个变量进行更新，因为有个约束条件$$\sum_{i=1}^{N}\alpha_i y_i=0$$，所以至少更新两个变量）
两变量更新这里，对于最值判断容易困惑，其他根据公式推导即可。

如下：

更新前两变量为$$\alpha_1^{old}$$和$$\alpha_2^{old}$$，更新后的值为$$\alpha_1^{new}$$和$$\alpha_2^{new}$$。
则由约束条件可知：

$$\alpha_1^{old}+\alpha_2^{old}=\alpha_1^{new}+\alpha_2^{new}=k$$

且
$$0 \le \alpha_i \le C$$
结合原论文做了以下标注，
![](https://ws2.sinaimg.cn/large/006tKfTcgy1fpy8zx6ekfj30rn0cc0u9.jpg)
可看出更新$$\alpha_2$$：
- $$y_1\ne y_2$$：$$\alpha_2$$最小值是在0和-k取最大值，同理$$H=min(C,C+k)$$;
- $$y_1= y_2$$：$$\alpha_2$$最小值是在0和k-C取最大值，同理$$H=min(C,k)$$；


这里比较容易迷惑，其他按照公式推导即可。


----------
#### Python实现SMO和SVM，两种核（linear和RBF）
结合SMO论文中的伪代码，就可以开始愉快的编写SMO求解SVM的Python代码了。
代码如下：
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-29 17:06:57
# @Author  : guanglinzhou (xdzgl812@163.com)
# @Link    : https://github.com/GuanglinZhou
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
import math


# 加载数据集
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    dataMat = np.mat(dataMat)
    labelMat = np.mat(labelMat).transpose()
    return dataMat, labelMat


# 定义一个类作为数据结构，存储一些关键值
class classSMO:
    def __init__(self, dataMat, labelMat, C, tol):  # Initialize the structure with the parameters
        self.dataMat = dataMat
        self.labelMat = labelMat
        self.C = C
        self.tol = tol
        self.m = np.shape(dataMat)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 1)))  # first column is valid flag
        self.kernelType = 'linear'
        # self.kernelType = 'Gaussian'
        # 储存K(xi,xj)的值
        self.kernelFunctionValue = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.kernelFunctionValue[:, i] = kernel(i, self)


def kernel(i, objectSMO):
    if (objectSMO.kernelType == 'linear'):
        return np.dot(objectSMO.dataMat[:, :], objectSMO.dataMat[i, :].transpose())
    if (objectSMO.kernelType == 'Gaussian'):
        sigma = 1
        temp = -np.square(objectSMO.dataMat[:, :] - objectSMO.dataMat[i, :]) / (2 * math.pow(sigma, 2))
        ndarr = np.array(temp)
        ndarr = np.sum(ndarr, axis=1)
        temp = np.mat(ndarr).transpose()
        return np.exp(temp)


# 计算g(xi)的值
def computeGx(i, objectSMO):
    if (objectSMO.kernelType == 'linear'):
        g_xi1 = np.multiply(objectSMO.alphas, objectSMO.labelMat)
        g_xi2 = np.multiply(objectSMO.dataMat, objectSMO.dataMat[i])
        g_xi = np.sum(np.multiply(g_xi1, g_xi2)) + objectSMO.b
    elif (objectSMO.kernelType == 'Gaussian'):
        g_xi = np.multiply(objectSMO.alphas, objectSMO.labelMat)
        g_xi = np.multiply(g_xi, objectSMO.kernelFunctionValue[:, i])
        g_xi = np.sum(g_xi) + objectSMO.b
    else:
        g_xi = 0
    return g_xi


# 计算Ei的值
def computeEk(k, objectSMO):
    print('computeEi')
    g_xi = computeGx(k, objectSMO)
    return (g_xi - objectSMO.labelMat[k]).item()


# 检查误分类的个数
def checkClassifierErrorSample(objectSMO):
    errorNum = 0
    for i in range(objectSMO.dataMat.shape[0]):
        predi = np.sign(computeGx(i, objectSMO))
        yi = objectSMO.labelMat[i].item()
        if (predi != yi):
            errorNum += 1
    print('核函数为：{}, 样本总数为：{}，分类错误的样本个数为：{}'.format(objectSMO.kernelType, objectSMO.m, errorNum))


# 绘制原始数据
def plotData(Xmat, ymat):
    Xarray = np.array(Xmat)
    yarray = np.array(ymat)
    col = {+1: 'r', -1: 'b'}
    plt.figure()
    for i in range(Xarray.shape[0]):
        plt.plot(Xarray[i, 0], Xarray[i, 1], col[yarray[i][0]] + 'o')
    plt.show()


# 更新Ei
def updateEk(objectSMO, k):
    objectSMO.eCache[k] = computeEk(k, objectSMO)


# 更新alpha1和alpha2
def takeStep(i, j):
    if (i == j):
        return 0
    # y1 y2 E1 E2为标量形式
    y1 = objectSMO.labelMat[i].item()
    y2 = objectSMO.labelMat[j].item()
    E1 = computeEk(i, objectSMO)
    E2 = computeEk(j, objectSMO)
    # alpha1old alpha2old为标量形式
    alpha1old = objectSMO.alphas[i].item()
    alpha2old = objectSMO.alphas[j].item()
    s = y1 * y2
    if (labelMat[i] != labelMat[j]):
        L = max(0, alpha2old - alpha1old)
        H = min(C, C + alpha2old - alpha1old)
    else:
        L = max(0, alpha2old + alpha1old - C)
        H = min(C, alpha2old + alpha1old)
    if (L == H):
        return 0
    # Kij为标量形式
    K11 = objectSMO.kernelFunctionValue[1, 1]
    K22 = objectSMO.kernelFunctionValue[2, 2]
    K12 = objectSMO.kernelFunctionValue[1, 2]
    eta = K11 + K22 - 2 * K12
    if (eta > 0):
        alpha2New = alpha2old + y2 * (E1 - E2) / eta
        if (alpha2New < L):
            alpha2New = L
        elif (alpha2New > H):
            alpha2New = H
    else:
        print('eta<=0')
        return 0
    if (np.abs(alpha2New - alpha2old) < 0.00001):
        print('alpha2 step size too small')
        return 0
    alpha1New = alpha1old + s * (alpha2old - alpha2New)
    b1 = -E1 - y1 * K11 * (alpha1New - alpha1old) - y2 * K12 * (alpha2New - alpha2old) + objectSMO.b
    b2 = -E2 - y1 * K12 * (alpha1New - alpha1old) - y2 * K22 * (alpha2New - alpha2old) + objectSMO.b
    # update
    objectSMO.b = (b1 + b2) / 2
    updateEk(objectSMO, i)
    updateEk(objectSMO, j)
    objectSMO.alphas[i] = alpha1New
    objectSMO.alphas[j] = alpha2New
    return 1


# 选择第2个变量
def secondChoice(i1, objectSMO):
    maxAbsEDelta = 0
    i2OfMaxAbsEDelta = 0
    Ei1 = objectSMO.eCache[i1].item()
    for i2 in range(objectSMO.eCache.shape[0]):
        Ei2 = objectSMO.eCache[i2].item()
        absEDelta = np.abs(Ei1 - Ei2)
        if (absEDelta > maxAbsEDelta):
            maxAbsEDelta = absEDelta
            i2OfMaxAbsEDelta = i2
    return i2OfMaxAbsEDelta


# 内层循环
def examineExample(i1, objectSMO, C):
    y1 = objectSMO.labelMat[i1]
    alpha1Old = objectSMO.alphas[i1]
    E1 = computeEk(i1, objectSMO)
    r1 = E1 * y1
    if ((r1 < -tol and alpha1Old < C) or (r1 > tol and alpha1Old > 0)):
        set1 = set(np.where(objectSMO.alphas > 0)[0])
        set2 = set(np.where(objectSMO.alphas < C)[0])
        indexAlphaNot0andC = set1 & set2
        if (len(indexAlphaNot0andC) > 1):
            i2 = secondChoice(i1, objectSMO)
            if (takeStep(i1, i2)):
                return 1
        # loop over all no-bound alpha,starting at a random point
        indexAlphaNot0andCList = list(indexAlphaNot0andC)
        shuffle(indexAlphaNot0andCList)
        for i2 in indexAlphaNot0andCList:
            if (takeStep(i1, i2)):
                return 1
        # loop over all possible i1,starting at a random point
        indexAllDataSet = list(range(objectSMO.dataMat.shape[0]))
        shuffle(indexAllDataSet)
        for i2 in indexAllDataSet:
            if (takeStep(i1, i2)):
                return 1
    return 0


# 外循环
def mainRoutine(dataMat, C, maxIter):
    m, n = np.shape(dataMat)
    numChanged = 0
    examineAll = True
    iterNum = 0
    while ((iterNum < maxIter) and (numChanged > 0 or examineAll)):
        numChanged = 0
        if (examineAll):
            for i in range(m):
                numChanged += examineExample(i, objectSMO, C)
            iterNum += 1
        else:
            set1 = set(np.where(objectSMO.alphas > 0)[0])
            set2 = set(np.where(objectSMO.alphas < C)[0])
            indexAlphaNot0andC = set1 & set2
            for i in indexAlphaNot0andC:
                numChanged += examineExample(i, objectSMO, C)
            iterNum += 1
        if (examineAll == True):
            examineAll = False
        elif (numChanged == 0):
            examineAll = True


# 计算linear的权重
def computeTheta(objectSMO):
    theta = np.dot(np.multiply(objectSMO.alphas, objectSMO.labelMat).transpose(), objectSMO.dataMat)
    b = objectSMO.b
    thetaList = theta.tolist()
    thetaList[0].append(b)
    theta = np.mat(thetaList).transpose()
    return theta


# 绘制核为linear的超平面
def plotHyperplaneLinear(Xmat, ymat, theta, objectSMO):
    m, n = np.shape(Xmat)
    Xarray = np.array(Xmat)
    yarray = np.array(ymat)
    col = {+1: 'r', -1: 'b'}
    plt.figure()
    for i in range(Xarray.shape[0]):
        plt.plot(Xarray[i, 0], Xarray[i, 1], col[yarray[i][0]] + 'o')
    plt.ylim([-6, 6])
    plt.plot(Xarray[:, 0],
             (-(theta[0][0] * Xarray[:, 0] + np.multiply(theta[2][0], np.ones((m, 1)))) / theta[1][0]).transpose(),
             c='g')
    # 标出支持向量
    for i in range(objectSMO.m):
        if (round(objectSMO.alphas[i].item(), 3) < objectSMO.C and round(objectSMO.alphas[i].item(), 3) > 0):
            plt.plot(Xarray[i, 0], Xarray[i, 1], 'y' + 'o')
    plt.show()


if __name__ == '__main__':
    '''
    通过更改此处文件名，以及类别classSMO中成员kernelType，切换linear kernel和RBF kernel
    '''
    fileName = 'testSetLinear.txt'
    # fileName = 'testSetRBF.txt'
    dataMat, labelMat = loadDataSet(fileName)
    print(np.shape(dataMat))
    C = 200
    tol = 0.001
    maxIter = 200
    objectSMO = classSMO(dataMat, labelMat, C, tol)
    mainRoutine(dataMat, C, maxIter)
    plotData(objectSMO.dataMat, objectSMO.labelMat)
    theta = computeTheta(objectSMO)
    plotHyperplaneLinear(objectSMO.dataMat, objectSMO.labelMat, theta, objectSMO)
    checkClassifierErrorSample(objectSMO)


```



![](https://ws3.sinaimg.cn/large/006tKfTcgy1fpy8yw4qy4j30hs0eymxn.jpg)
![](https://ws1.sinaimg.cn/large/006tKfTcgy1fpy8z91wgwj30hs0eyaal.jpg)
![](https://ws3.sinaimg.cn/large/006tKfTcgy1fpy8zerwi8j309f00u0so.jpg)
![](https://ws4.sinaimg.cn/large/006tKfTcgy1fpy905hkbgj30hs0dcglu.jpg)
**高斯核函数将二维输入空间映射到高维特征空间，不知道在二维上如何绘制超平面的轮廓，所以这里就用checkClassifierErrorSample(objectSMO)函数打印输出误分类点的个数，根据结果可看到高斯核函数解决了非线性分类的问题。**
![](https://ws3.sinaimg.cn/large/006tKfTcgy1fpy8ycbk7ej30a400x0sp.jpg)

----------
#### 逻辑斯蒂回归实现非线性分类；（有待更新....）
`todo`，这两天导师干活逼迫的紧，这部分待更新......

----------


**后记**
>真的，推导并且实现一遍，发现其实SVM真的不难，也许是我们觉得它比较难，导致它变难了吧。


----------

参考资料：
- SVM，SMO论文
- 书《机器学习实战》《机器学习》
- [KKT条件与拉格朗日对偶性](https://matafight.github.io/2015/05/13/KKT%E6%9D%A1%E4%BB%B6%E4%B8%8E%E6%8B%89%E6%A0%BC%E6%9C%97%E6%97%A5%E5%AF%B9%E5%81%B6%E6%80%A7/)
- [核技巧](https://www.fanyeong.com/2017/11/13/the-kernel-trick/)
- [知乎—机器学习有很多关于核函数的说法，核函数的定义和作用是什么？](https://www.zhihu.com/question/24627666)
- [SMO算法](http://www.cnblogs.com/jerrylead/archive/2011/03/18/1988419.html)
- [SMO算法剖析](https://blog.csdn.net/luoshixian099/article/details/51227754)
