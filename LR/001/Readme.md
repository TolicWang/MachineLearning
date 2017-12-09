原文地址http://blog.csdn.net/The_lastest/article/details/78761577

# Logistic回归代价函数的数学推导及实现



---
logistic回归的代价函数形式如下：
$$J(\theta) = -\frac{1}{m}\left[\sum_{i=1}^{m}y^{(i)}\log h_\theta(x^{(i)}) + (1 - y^{(i)})\log (1 - h_\theta(x^{(i)}))\right]$$

可是这又是怎么来的呢？ 答：最大似然估计计算出来的

<font color = red size = 5>1.最大似然估计</font>

我们先来简单的回顾一下最大似然估计(Maximum likelihood estimation),详细[戳此处,见参数估计](http://blog.csdn.net/The_lastest/article/details/78759837)

所谓参数估计就是：对未知参数$\theta$进行估计时，在参数可能的取值范围内选取，使“样本获得此观测值$x_1,x_2...,x_n$"的概率最大的参数$\hat{\theta}$作为$\theta$的估计，这样选定的$\hat{\theta}$有利于$x_1,x_2...,x_n$"的出现。也就是说在已知数据集（结果）和模型（分布函数）的情况下，估计出最适合该模型的参数。

**举个例子：**

假设你有一枚硬币，随机抛10次；现在的结果是6次正面。我们都知道，抛一枚硬币，正面朝上和反面朝上的概率均是θ=0.5；但前提时，这是在大量的实验（抛硬币）情况下才有的结论。那在我们这个情况下，参数θ到底取何值时才能使得出现6次正面的肯能性最大呢？

我们知道，抛硬币是符合二项分布B(n,p)，也就是说我们现在已知样本结果以及函数分布，估计出使得该结果最大可能出现的参数$\hat{\theta}$。则有： 
$$\mathrm{L}=P(X=6)=\mathrm{C_{10}^6}\hat{\theta}^6(1-\hat{\theta})^4$$

而我们接下来要做的就是求当$\mathrm{L}$取最大值时，$\hat{\theta}$的值。我们很容易求得当$\hat{\theta}=0.6$时$\mathrm{L}$取得最大值0.25；而当$\hat{\theta}=0.5$时，$\mathrm{L}=0.21$

再假设你有一枚硬币，随机抛10次；现在的结果是7次正面。则此时使得该结果最大可能性出现参数$\hat{\theta}$又是多少呢？按照上面的方法我们很容易求得当$\hat{\theta}=0.7$时可能性最大。

**再举个例子：**

明显，在Logistic回归中，所有样本点也服从二项分布；设有$x_1,x_2,x_3$三个样本点，其类标为$1,1,0$；同时设样本点为1的概率为$P=h_{\theta}(x)$，那么当$P$等于多少时，其结果才最可能出现$1,1,0$呢？于是问题就变成最大化：
$$P*P(1-P)=h_{\theta}(x_1)*h_{\theta}(x_2)*(1-h_{\theta}(x_3))$$

而这就是最大似然估计（求解参数估计的一个方法）

<font color = red size = 5>2.最大似然估计的数学定义</font>

 - 设总体X是离散型，其概率分布为$P\{X=x\}=p(x;\theta),\theta$为未知参数，$X_1,X_2,...,X_n$为$X$的一个样本，则$X_1,X_2,...,X_n$ 取值为$x_1,...,x_n$的概率是：
 $$P\{X_1=x_1,...,X_n=x_n\}=\prod_{i=1}^nP\{X_i=x_i\}=\prod_{i=1}^np\{x_i;\theta\}$$

显然这个概率值是$\theta$的函数，将其记为
$$\mathrm{L}(\theta)=L(x_1,...,x_n;\theta)=\prod_{i=1}^np\{x_i;\theta\}$$

称$\mathrm{L}(\theta)$为样本$(x_1,...,x_n)$的**似然函数**.
若$\hat{\theta}$使得
$$\mathrm{L}(x_1,...,x_n;\hat{\theta})=\max \mathrm{L}(x_1,...,x_n;\theta)$$

则称$\hat{\theta}=\hat{\theta}(x_1,...,x_n)$为未知参数$\theta$的**最大似然估计值**.

**求解步骤**

a. 写出似然函数；

b. 方程两边同时取$\mathrm{ln}$的对数；

c. 令$\frac{\partial\mathrm{ln L}}{\partial \theta_i}=0$，求得参数

<font color = red size = 5>3.Logistic回归代价函的推导</font>

设:
\begin{align*}
&P(y=1|x;\theta)=h_{\theta}(x)\\[1ex]
&P(y=0|x;\theta)=1-h_{\theta}(x)\\[1ex]
&h(x)=g(z)=g(\theta^Tx)
\end{align*}

将两者合并到一起为:
$$p(y|x;\theta)=(h_{\theta}(x))^y(1-h_{\theta}(x))^{1-y}\tag{3.1}$$

因此我们可以得到似然函数：
\begin{align*}
L(\theta)&=\prod_{i=1}^mp(y^{(i)}|x^{(i)};\theta)\\[2ex]
&=\prod_{i=1}^m(h_{\theta}(x^{(i)}))^{y^{(i)}}(1-h_{\theta}(x))^{1-y^{(i)}}\tag{3.2}
\end{align*}

于是两边同时取自然对数得：
\begin{align*}
\mathcal{l}(\theta) &= \log{L}(\theta)\\[3ex]
&=\sum_{i=1}^m\left[y^{(i)}\log{h(x^{(i)})+(1-y^{(i)})\log{(1-h(x^{(i)}))}}\right]\tag{3.3}
\end{align*}

插播一个公式：
$$\log{a^bc^d}=\log{a^b}+\log{c^d}=b\log{a}+d\log{c}$$

且最大化$(3.3)$等价于最小化$-\frac{1}{m}l(\theta)$

下面我们梯度下降算法进行求解，所以就要对所有的参数求偏导，按照惯例，我们先考虑一个样本点($m=1$)的情况，然后在矢量化：
\begin{align*}
\frac{\partial \mathcal{l}}{\partial\theta_j}&=-\frac{\partial }{\partial\theta_j}y\log{(h_{\theta}(x))}+(1-y)\log{(1-h_{\theta}(x))}\\[2ex]
&=-\frac{\partial }{\partial\theta_j}y\log{(g(z))}+(1-y)\log{(1-g(z))}\\[2ex]
&=-\frac{y}{g(z)}\frac{\partial g(z)}{\partial \theta_j}-\frac{(1-y)}{(1-g(z))}\frac{\partial g(z)}{\partial \theta_j}\\[2ex]
&=-\left(\frac{y}{g(\theta^Tx)}-\frac{(1-y)}{(1-g(\theta^Tx))}\right)\frac{\partial g(\theta^Tx)}{\partial \theta^Tx}\frac{\partial \theta^Tx}{\partial \theta_j}\\[2ex]
&=-(y-g(z))x_j\tag{3.4}
\\[2ex]
\\
\\[2ex]
\frac{\partial \mathcal{l}}{\partial b}&=-\left(\frac{y}{g(\theta^Tx)}-\frac{(1-y)}{(1-g(\theta^Tx))}\right)\frac{\partial g(\theta^Tx)}{\partial \theta^Tx}\frac{\partial \theta^Tx}{\partial b}\\[2ex]
&=-(y-g(z))*1\tag{3.5}
\end{align*}
 
这儿用的是$g(z)=1/(1+exp(-z))$，同样还可以是$tanh(z)$,但是选取前者主要是因为其求导之后的结果为$g'(z)=g(z)(1-g(z))$便于化简。

于是根据梯度下降算法有：
\begin{align*}
&\theta_j = \theta_j-\frac{1}{m}\alpha(h(x)-y)\cdot x_j\tag{3.6}\\[2ex]
&b = b - \frac{1}{m}\alpha\sum(h(x)-y)\tag{3.7}
\end{align*}

注意，此时的$x,y$均为n个样本点的情况，与$(3.4)$中不同。$x_j$是指，每个样本点的第j维所组成的列向量。

$(3.6)$矢量化后为:
$$\theta=\theta-\frac{1}{m}\alpha\;x^T\cdot(h(x)-y) $$

"$\cdot$"表示点乘

<font color = red size = 5>4. python 实现</font>
```python
def gradDescent(X, y, W, b, alpha):
    maxIteration = 500
    for i in range(maxIteration):
        z = X*W+b
        error = sigmoid(z) - y
        W = W - (1.0/m)*alpha*X.T*error
        b = b - (1.0/m)*alpha*np.sum(error)
    return W,b
```

