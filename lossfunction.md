**损失函数**

一般来说，我们在进行机器学习任务时，使用的每一个算法都有一个目标函数，算法便是对这个目标函数进行优化，特别是在分类或者回归任务中，便是使用损失函数（Loss Function）作为其目标函数，又称为代价函数(Cost Function)。

损失函数是用来评价模型的预测值 ![\tilde{y}=f\left( X \right)](https://www.zhihu.com/equation?tex=%5Ctilde%7By%7D%3Df%5Cleft%28+X+%5Cright%29) 与真实值 ![Y](https://www.zhihu.com/equation?tex=Y) 的不一致程度，它是一个非负实值函数。通常使用 ![L\left( Y,f\left( x \right) \right)](https://www.zhihu.com/equation?tex=L%5Cleft%28+Y%2Cf%5Cleft%28+x+%5Cright%29+%5Cright%29) 来表示，损失函数越小，模型的性能就越好。

设总有NN个样本的样本集为 ![\left( X,Y \right)=\left( x_{i},y_{i} \right)](https://www.zhihu.com/equation?tex=%5Cleft%28+X%2CY+%5Cright%29%3D%5Cleft%28+x_%7Bi%7D%2Cy_%7Bi%7D+%5Cright%29) ， ![y_{i},i\in\left[ 1,N \right]](https://www.zhihu.com/equation?tex=y_%7Bi%7D%2Ci%5Cin%5Cleft%5B+1%2CN+%5Cright%5D) 为样本ii的真实值， ![\tilde{y}_{i}=f(x_{i}),i\in\left[ 1,N \right]](https://www.zhihu.com/equation?tex=%5Ctilde%7By%7D_%7Bi%7D%3Df%28x_%7Bi%7D%29%2Ci%5Cin%5Cleft%5B+1%2CN+%5Cright%5D) 为样本ii的预测值， ![f](https://www.zhihu.com/equation?tex=f) 为分类或者回归函数。

那么总的损失函数为：

![](https://pic3.zhimg.com/v2-e5974566f9d7ab964defa60fb1d5bf52_b.jpg)

作者：人在旅途
链接：https://zhuanlan.zhihu.com/p/47202768
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

**(1) Zero-one Loss（0-1损失）**

它是一种较为简单的损失函数，如果预测值与目标值不相等，那么为1，否则为0，即：

![](https://pic1.zhimg.com/v2-0c2a7664eefd9a4c678dbd1b1020b6ec_b.jpg)

由式(1)可知，上述的定义太过严格，如果真实值为1，预测值为0.999，那么预测应该正确，但是上述定义显然是判定为预测错误，那么可以进行改进为Perceptron Loss。

**(2) Perceptron Loss（感知损失）**

表达式为：

![](https://pic1.zhimg.com/v2-0f4ba2fde127036b01a745a981b86790_b.jpg)

其中tt是一个超参数阈值，如在PLA([Perceptron Learning Algorithm,感知机算法](https://link.zhihu.com/?target=http%3A//kubicode.me/2015/08/06/Machine%2520Learning/Perceptron-Learning-Algorithm/))中取t=0.5。

**(3)Hinge Loss**

Hinge损失可以用来解决间隔最大化问题，如在SVM中解决几何间隔最大化问题，其定义如下：

![](https://pic4.zhimg.com/v2-d05c37ccd5c970f5912416d4a25c6b57_b.jpg)

**(4)交叉熵损失函数**

在使用似然函数最大化时，其形式是进行连乘，但是为了便于处理，一般会套上log，这样便可以将连乘转化为求和，由于log函数是单调递增函数，因此不会改变优化结果。因此log类型的损失函数也是一种常见的损失函数，如在LR(Logistic Regression, 逻辑回归)中使用交叉熵(Cross Entropy)作为其损失函数。即：

![](https://pic2.zhimg.com/v2-5c975b681fdd4543ce3a108bf90df22d_b.jpg)

**(5)平方误差损失（Square Loss）**

Square Loss即平方误差，常用于回归中。即：

![](https://pic3.zhimg.com/v2-a253e83fa4c667ab765d53dd885004fe_b.jpg)

对式(6)加和再求平均即MSE：

![](https://pic2.zhimg.com/v2-c1d0ef9712f8b6adcf0ec3f107c02ed9_b.jpg)

**(6) Absolute Loss**

Absolute Loss即绝对值误差，常用于回归中。即：

![](https://pic4.zhimg.com/v2-30890734a6a1ec10398ab63c14f9f9af_b.jpg)

**(7) Exponential Loss**

Exponential Loss为指数误差，常用于boosting算法中，如[AdaBoost](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/AdaBoost%2522%2520%255Ct%2520%2522_blank)。即：

![](https://pic2.zhimg.com/v2-5aa2f25c5085fa388a969d597b4ea881_b.jpg)

**(8)正则**

一般来说，对分类或者回归模型进行评估时，需要使得模型在训练数据上使得损失函数值最小，即使得经验风险函数最小化，但是如果只考虑经验风险(Empirical risk)，容易过拟合(详细参见防止过拟合的一些方法)，因此还需要考虑模型的泛化能力，一般常用的方法便是在目标函数中加上正则项，由损失项(Loss term)加上正则项(Regularization term)构成结构风险(Structural risk)，那么损失函数变为：

![](https://pic1.zhimg.com/v2-c49d84a4bdf40ba51cd396399db165f4_b.jpg)

其中，λλ是正则项超参数，常用的正则方法包括：L1正则与L2正则。

![](https://pic3.zhimg.com/v2-19691b4ff4b8d72ed4300d0ba452356a_b.jpg)
**各种损失函数图形如下：**

![](https://pic3.zhimg.com/v2-870f3d1991a13ab5f4f696698805a4a2_b.jpg)


作者：小飞鱼
链接：https://zhuanlan.zhihu.com/p/35709485
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

Cross Entropy Error Function（交叉熵损失函数）

*   例子
*   表达式
*   函数性质
*   学习过程
*   优缺点

* * *

这篇文章中，讨论的Cross Entropy损失函数常用于分类问题中，但是为什么它会在分类问题中这么有效呢？我们先从一个简单的分类例子来入手。

## 1\. 预测政治倾向例子

我们希望根据一个人的年龄、性别、年收入等相互独立的特征，来预测一个人的政治倾向，有三种可预测结果：民主党、共和党、其他党。假设我们当前有两个逻辑回归模型（参数不同），这两个模型都是通过sigmoid的方式得到对于每个预测结果的概率值：

**模型1**：

![](https://pic3.zhimg.com/v2-0c49d6159fc8a5676637668683d41762_b.jpg)

模型1预测结果

**模型1**对于样本1和样本2以非常微弱的优势判断正确，对于样本3的判断则彻底错误。

**模型2**：

![](https://pic3.zhimg.com/v2-6d31cf03185b408d5e93fa3e3c05096e_b.jpg)

模型2预测结果

**模型2**对于样本1和样本2判断非常准确，对于样本3判断错误，但是相对来说没有错得太离谱。

好了，有了模型之后，我们需要通过定义损失函数来判断模型在样本上的表现了，那么我们可以定义哪些损失函数呢？

## 1.1 Classification Error（分类错误率）

最为直接的损失函数定义为： ![classification\ error=\frac{count\ of\ error\ items}{count\ of \ all\ items}](https://www.zhihu.com/equation?tex=classification%5C+error%3D%5Cfrac%7Bcount%5C+of%5C+error%5C+items%7D%7Bcount%5C+of+%5C+all%5C+items%7D)

**模型1：** ![classification\ error=\frac{1}{3}](https://www.zhihu.com/equation?tex=classification%5C+error%3D%5Cfrac%7B1%7D%7B3%7D)

**模型2：** ![classification\ error=\frac{1}{3}](https://www.zhihu.com/equation?tex=classification%5C+error%3D%5Cfrac%7B1%7D%7B3%7D)

我们知道，**模型1**和**模型2**虽然都是预测错了1个，但是相对来说**模型2**表现得更好，损失函数值照理来说应该更小，但是，很遗憾的是， ![classification\ error](https://www.zhihu.com/equation?tex=classification%5C+error) 并不能判断出来，所以这种损失函数虽然好理解，但表现不太好。

## 1.2 Mean Squared Error (均方误差)

均方误差损失也是一种比较常见的损失函数，其定义为： ![MSE=\frac{1}{n}\sum_{i}^n(\hat{y_i}-y_i)^2](https://www.zhihu.com/equation?tex=MSE%3D%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%7D%5En%28%5Chat%7By_i%7D-y_i%29%5E2)

**模型1：** ![MSE=\frac{0.54+0.54+1.34}{3}=0.81](https://www.zhihu.com/equation?tex=MSE%3D%5Cfrac%7B0.54%2B0.54%2B1.34%7D%7B3%7D%3D0.81)

**模型2：** ![MSE=\frac{0.14+0.14+0.74}{3}=0.34](https://www.zhihu.com/equation?tex=MSE%3D%5Cfrac%7B0.14%2B0.14%2B0.74%7D%7B3%7D%3D0.34)

我们发现，MSE能够判断出来**模型2**优于**模型1**，那为什么不采样这种损失函数呢？主要原因是逻辑回归配合MSE损失函数时，采用梯度下降法进行学习时，会出现模型一开始训练时，学习速率非常慢的情况（[MSE损失函数](https://zhuanlan.zhihu.com/p/35707643)）。

有了上面的直观分析，我们可以清楚的看到，对于分类问题的损失函数来说，分类错误率和平方和损失都不是很好的损失函数，下面我们来看一下交叉熵损失函数的表现情况。

## 1.3 Cross Entropy Error Function（交叉熵损失函数）

## 1.3.1 表达式

## (1) 二分类

在二分的情况下，模型最后需要预测的结果只有两种情况，对于每个类别我们的预测得到的概率为 ![p](https://www.zhihu.com/equation?tex=p) 和 ![1-p](https://www.zhihu.com/equation?tex=1-p) 。此时表达式为：

![\begin{align}L = −[y\cdot log(p)+(1−y)\cdot log(1−p)]\end{align} \\](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7DL+%3D+%E2%88%92%5By%5Ccdot+log%28p%29%2B%281%E2%88%92y%29%5Ccdot+log%281%E2%88%92p%29%5D%5Cend%7Balign%7D+%5C%5C)

其中：
- y——表示样本的label，正类为1，负类为0
- p——表示样本预测为正的概率

## (2) 多分类

多分类的情况实际上就是对二分类的扩展：

![\begin{align}L = -\sum_{c=1}^My_{c}\log(p_{c})\end{align} \\](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7DL+%3D+-%5Csum_%7Bc%3D1%7D%5EMy_%7Bc%7D%5Clog%28p_%7Bc%7D%29%5Cend%7Balign%7D+%5C%5C)

其中：
- ![M](https://www.zhihu.com/equation?tex=M) ——类别的数量；
- ![y_c](https://www.zhihu.com/equation?tex=y_c) ——指示变量（0或1）,如果该类别和样本的类别相同就是1，否则是0；
- ![p_c](https://www.zhihu.com/equation?tex=p_c) ——对于观测样本属于类别 ![c](https://www.zhihu.com/equation?tex=c) 的预测概率。

现在我们利用这个表达式计算上面例子中的损失函数值：

**模型1**：
![\begin{align} CEE &= -[0\times log0.3 + 0\times log0.3 + 1\times log0.4] \\ &-[0\times log0.3 + 1\times log0.4 + 0\times log0.3]\\&-[1\times log0.1 + 0\times log0.2 + 0\times log0.7]\\ &= 0.397+ 0.397+ 1 \\ &= 1.8 \end{align} \\](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D+CEE+%26%3D+-%5B0%5Ctimes+log0.3+%2B+0%5Ctimes+log0.3+%2B+1%5Ctimes+log0.4%5D+%5C%5C+%26-%5B0%5Ctimes+log0.3+%2B+1%5Ctimes+log0.4+%2B+0%5Ctimes+log0.3%5D%5C%5C%26-%5B1%5Ctimes+log0.1+%2B+0%5Ctimes+log0.2+%2B+0%5Ctimes+log0.7%5D%5C%5C+%26%3D+0.397%2B+0.397%2B+1+%5C%5C+%26%3D+1.8+%5Cend%7Balign%7D+%5C%5C)

**模型2：**

![\begin{align} CEE &= -[0\times log0.1 + 0\times log0.2 + 1\times log0.7] \\ &-[0\times log0.1 + 1\times log0.7 + 0\times log0.2]\\&-[1\times log0.3 + 0\times log0.4 + 0\times log0.3] \\ &= 0.15+ 0.15+ 0.397 \\ &= 0.697 \end{align} \\](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D+CEE+%26%3D+-%5B0%5Ctimes+log0.1+%2B+0%5Ctimes+log0.2+%2B+1%5Ctimes+log0.7%5D+%5C%5C+%26-%5B0%5Ctimes+log0.1+%2B+1%5Ctimes+log0.7+%2B+0%5Ctimes+log0.2%5D%5C%5C%26-%5B1%5Ctimes+log0.3+%2B+0%5Ctimes+log0.4+%2B+0%5Ctimes+log0.3%5D+%5C%5C+%26%3D+0.15%2B+0.15%2B+0.397+%5C%5C+%26%3D+0.697+%5Cend%7Balign%7D+%5C%5C)

可以发现，交叉熵损失函数可以捕捉到**模型1**和**模型2**预测效果的差异。

## 2\. 函数性质

![](https://pic3.zhimg.com/v2-f049a57b5bb2fcaa7b70f5d182ab64a2_b.jpg)

可以看出，该函数是凸函数，求导时能够得到全局最优值。

## 3\. 学习过程

交叉熵损失函数经常用于分类问题中，特别是在神经网络做分类问题时，也经常使用交叉熵作为损失函数，此外，由于交叉熵涉及到计算每个类别的概率，所以交叉熵几乎每次都和**softmax函数**一起出现。

我们用神经网络最后一层输出的情况，来看一眼整个模型预测、获得损失和学习的流程：

1.  神经网络最后一层得到每个类别的得分**scores**；
2.  该得分经过**softmax函数**获得概率输出；
3.  模型预测的类别概率输出与真实类别的one hot形式进行交叉熵损失函数的计算。

学习任务分为二分类和多分类情况，我们分别讨论这两种情况的学习过程。

## 3.1 二分类情况

![](https://pic4.zhimg.com/v2-175cd936ac1c1ec85e67288e69a65763_b.jpg)

二分类任务学习过程

如上图所示，求导过程可分成三个子过程，即拆成三项偏导的乘积：

![\frac{\partial L}{\partial w_i}=\frac{\partial J}{\partial p}\cdot \frac{\partial p}{\partial s}\cdot \frac{\partial s}{\partial w_i} \\](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+L%7D%7B%5Cpartial+w_i%7D%3D%5Cfrac%7B%5Cpartial+J%7D%7B%5Cpartial+p%7D%5Ccdot+%5Cfrac%7B%5Cpartial+p%7D%7B%5Cpartial+s%7D%5Ccdot+%5Cfrac%7B%5Cpartial+s%7D%7B%5Cpartial+w_i%7D+%5C%5C)

## 3.1.1 计算第一项： ![\frac{\partial L}{\partial p}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+L%7D%7B%5Cpartial+p%7D)

- ![p](https://www.zhihu.com/equation?tex=p) 表示预测为True的概率；

- ![y](https://www.zhihu.com/equation?tex=y) 表示为True时等于1，否则等于0；

![\begin{aligned} \frac{\partial L}{\partial p} &=\frac{\partial -[y\cdot log(p) + (1-y)\cdot log(1-p)]}{\partial p}\\ &= -\frac{y}{p}-[(1-y)\cdot \frac{1}{1-p}\cdot (-1)] \\  &= -\frac{y}{p}+\frac{1-y}{1-p} \\ \end{aligned} \\](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+%5Cfrac%7B%5Cpartial+L%7D%7B%5Cpartial+p%7D+%26%3D%5Cfrac%7B%5Cpartial+-%5By%5Ccdot+log%28p%29+%2B+%281-y%29%5Ccdot+log%281-p%29%5D%7D%7B%5Cpartial+p%7D%5C%5C+%26%3D+-%5Cfrac%7By%7D%7Bp%7D-%5B%281-y%29%5Ccdot+%5Cfrac%7B1%7D%7B1-p%7D%5Ccdot+%28-1%29%5D+%5C%5C++%26%3D+-%5Cfrac%7By%7D%7Bp%7D%2B%5Cfrac%7B1-y%7D%7B1-p%7D+%5C%5C+%5Cend%7Baligned%7D+%5C%5C)

## 3.1.2 计算第二项： ![\frac{\partial p}{\partial s} ](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+p%7D%7B%5Cpartial+s%7D+)

这一项要计算的是softmax函数对于score的导数，我们先回顾一下softmax函数和分数求导的公式：

> ![p_c = \sigma(s_c) = \frac{1}{1+e^{s_c}}  \\](https://www.zhihu.com/equation?tex=p_c+%3D+%5Csigma%28s_c%29+%3D+%5Cfrac%7B1%7D%7B1%2Be%5E%7Bs_c%7D%7D++%5C%5C)
> ![f'(x) = \frac{g(x)}{h(x)}=\frac{g'(x)h(x)-g(x){h}'(x)}{h^2(x)} \\](https://www.zhihu.com/equation?tex=f%27%28x%29+%3D+%5Cfrac%7Bg%28x%29%7D%7Bh%28x%29%7D%3D%5Cfrac%7Bg%27%28x%29h%28x%29-g%28x%29%7Bh%7D%27%28x%29%7D%7Bh%5E2%28x%29%7D+%5C%5C)

![\begin{aligned}  \frac{\partial p}{\partial s} &= \frac{1'\cdot (1+e^{s})-1\cdot (1+e^{s})'}{(1+e^{s})^2} \\  &= \frac{0\cdot (1+e^{s})-1\cdot e^{s}}{(1+e^{s})^2} \\  &= \frac{-e^{s}}{(1+e^{s})^2} \\  &= \frac{1}{1+e^{s}}\cdot \frac{-e^{s}}{1+e^{s}} \\  &= \sigma(s)\cdot [1-\sigma(s)] \\ \end{aligned} \\](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D++%5Cfrac%7B%5Cpartial+p%7D%7B%5Cpartial+s%7D+%26%3D+%5Cfrac%7B1%27%5Ccdot+%281%2Be%5E%7Bs%7D%29-1%5Ccdot+%281%2Be%5E%7Bs%7D%29%27%7D%7B%281%2Be%5E%7Bs%7D%29%5E2%7D+%5C%5C++%26%3D+%5Cfrac%7B0%5Ccdot+%281%2Be%5E%7Bs%7D%29-1%5Ccdot+e%5E%7Bs%7D%7D%7B%281%2Be%5E%7Bs%7D%29%5E2%7D+%5C%5C++%26%3D+%5Cfrac%7B-e%5E%7Bs%7D%7D%7B%281%2Be%5E%7Bs%7D%29%5E2%7D+%5C%5C++%26%3D+%5Cfrac%7B1%7D%7B1%2Be%5E%7Bs%7D%7D%5Ccdot+%5Cfrac%7B-e%5E%7Bs%7D%7D%7B1%2Be%5E%7Bs%7D%7D+%5C%5C++%26%3D+%5Csigma%28s%29%5Ccdot+%5B1-%5Csigma%28s%29%5D+%5C%5C+%5Cend%7Baligned%7D+%5C%5C)

## 3.1.3 计算第三项： ![\frac{\partial s}{\partial w_i \\}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+s%7D%7B%5Cpartial+w_i+%5C%5C%7D)

一般来说，scores是输入的线性函数作用的结果，所以有：
![\frac{\partial s}{\partial w_i}=x_i \\](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+s%7D%7B%5Cpartial+w_i%7D%3Dx_i+%5C%5C)

## 3.1.4 计算结果 ![\frac{\partial L}{\partial w_i}](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+L%7D%7B%5Cpartial+w_i%7D)

![\begin{aligned}  \frac{\partial L}{\partial w_i} &= \frac{\partial L}{\partial p}\cdot \frac{\partial p}{\partial s}\cdot \frac{\partial s}{\partial w_i} \\  &= [-\frac{y}{p}+\frac{1-y}{1-p}] \cdot \sigma(s)\cdot [1-\sigma(s)]\cdot x_i \\  &= [-\frac{y}{\sigma(s)}+\frac{1-y}{1-\sigma(s)}] \cdot \sigma(s)\cdot [1-\sigma(s)]\cdot x_i \\  &= [-\frac{y}{\sigma(s)}\cdot \sigma(s)\cdot (1-\sigma(s))+\frac{1-y}{1-\sigma(s)}\cdot \sigma(s)\cdot (1-\sigma(s))]\cdot x_i \\  &= [-y+y\cdot \sigma(s)+\sigma(s)-y\cdot \sigma(s)]\cdot x_i \\  &= [\sigma(s)-y]\cdot x_i \\ \end{aligned} \\](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D++%5Cfrac%7B%5Cpartial+L%7D%7B%5Cpartial+w_i%7D+%26%3D+%5Cfrac%7B%5Cpartial+L%7D%7B%5Cpartial+p%7D%5Ccdot+%5Cfrac%7B%5Cpartial+p%7D%7B%5Cpartial+s%7D%5Ccdot+%5Cfrac%7B%5Cpartial+s%7D%7B%5Cpartial+w_i%7D+%5C%5C++%26%3D+%5B-%5Cfrac%7By%7D%7Bp%7D%2B%5Cfrac%7B1-y%7D%7B1-p%7D%5D+%5Ccdot+%5Csigma%28s%29%5Ccdot+%5B1-%5Csigma%28s%29%5D%5Ccdot+x_i+%5C%5C++%26%3D+%5B-%5Cfrac%7By%7D%7B%5Csigma%28s%29%7D%2B%5Cfrac%7B1-y%7D%7B1-%5Csigma%28s%29%7D%5D+%5Ccdot+%5Csigma%28s%29%5Ccdot+%5B1-%5Csigma%28s%29%5D%5Ccdot+x_i+%5C%5C++%26%3D+%5B-%5Cfrac%7By%7D%7B%5Csigma%28s%29%7D%5Ccdot+%5Csigma%28s%29%5Ccdot+%281-%5Csigma%28s%29%29%2B%5Cfrac%7B1-y%7D%7B1-%5Csigma%28s%29%7D%5Ccdot+%5Csigma%28s%29%5Ccdot+%281-%5Csigma%28s%29%29%5D%5Ccdot+x_i+%5C%5C++%26%3D+%5B-y%2By%5Ccdot+%5Csigma%28s%29%2B%5Csigma%28s%29-y%5Ccdot+%5Csigma%28s%29%5D%5Ccdot+x_i+%5C%5C++%26%3D+%5B%5Csigma%28s%29-y%5D%5Ccdot+x_i+%5C%5C+%5Cend%7Baligned%7D+%5C%5C)

可以看到，我们得到了一个非常漂亮的结果，所以，使用交叉熵损失函数，不仅可以很好的衡量模型的效果，又可以很容易的的进行求导计算。

## 3.2 多分类情况

## 4\. 优缺点

## 4.1 优点

在用梯度下降法做参数更新的时候，模型学习的速度取决于两个值：一、**学习率**；二、**偏导值**。其中，学习率是我们需要设置的超参数，所以我们重点关注偏导值。从上面的式子中，我们发现，偏导值的大小取决于 ![x_i](https://www.zhihu.com/equation?tex=x_i) 和 ![[\sigma(s)-y]](https://www.zhihu.com/equation?tex=%5B%5Csigma%28s%29-y%5D) ，我们重点关注后者，后者的大小值反映了我们模型的错误程度，该值越大，说明模型效果越差，但是该值越大同时也会使得偏导值越大，从而模型学习速度更快。所以，使用逻辑函数得到概率，并结合交叉熵当损失函数时，在模型效果差的时候学习速度比较快，在模型效果好的时候学习速度变慢。




# 何恺明大神的「Focal Loss」，如何更好地理解？


## 5\. 参考

[1]. [神经网络的分类模型 LOSS 函数为什么要用 CROSS ENTROPY](https://link.zhihu.com/?target=http%3A//jackon.me/posts/why-use-cross-entropy-error-for-loss-function/)

[2]. [Softmax as a Neural Networks Activation Function](https://link.zhihu.com/?target=http%3A//sefiks.com/2017/11/08/softmax-as-a-neural-networks-activation-function/)

[3]. [A Gentle Introduction to Cross-Entropy Loss Function](https://link.zhihu.com/?target=https%3A//sefiks.com/2017/12/17/a-gentle-introduction-to-cross-entropy-loss-function/)，