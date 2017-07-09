# MechineLearning

<font size = 5 color = red> 1.为什么要引入神经网络（Neural Network）</font>

一句话总结就是当特征值n特别大时，比如当n为100时；仅仅是其2次项特征值$(x_1^2,x_1x_2,x_1x_3\dots x_1x_{100};x_2^2,x_2x_3\dots x_2x_{100};\dots)$就有大约5000个（从100累加到1）。而在实际问题中n的值往往有上百万，上亿。所以这样就非常容易导致过度拟合，以及计算量大的问题。因此，便引入了神经网络(neural network)。

<font size = 5 color = red> 2.神经网络模型（Neural Network Model）</font>

> Let's examine how we will represent a hypothesis function using neural networks. At a very simple level, neurons are basically computational units that take inputs (dendrites) as electrical inputs (called "spikes") that are channeled to outputs (axons). In our model, our dendrites are like the input features $x_1\dots x_n$, and the output is the result of our hypothesis function. In this model our $x_0$ input node is sometimes called the "**bias unit**." It is always equal to 1. In neural networks, we use the same logistic function as in classification, $\frac{1}{1+e^{-\theta^Tx}}$, yet we sometimes call it a sigmoid (logistic) **activation** function. In this situation, our "theta" parameters are sometimes called "**weights**".

如图就是一个只包含一个神经元的模型，黄色圆圈为神经元细胞(cell body)，
![这里写图片描述](http://img.blog.csdn.net/20170702175636007?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvVGhlX2xhc3Rlc3Q=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

而真正的神经网络是若干个这样不同的神经元组合而成的，如下图

![这里写图片描述](http://img.blog.csdn.net/20170702180601075?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvVGhlX2xhc3Rlc3Q=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

其中$x_0 = 1$，称为 bias unit，$a_0^{(2)}$称为mixture bias unit，也为1。通常我们不需要表示出来，知道其存在就好。另外，我们称Layer1为输入层(input layer)，Layer2为输出层(output layer)，中间的所有（这儿仅Layer2）层都称为隐藏层(hidden layer)。并且在这个例子中，我们称$a^2_0,a^2_1,a^2_2,a^2_3$为**活化单元**（activation unit）。

<font size = 5 color = red> 3.神经网络的数学定义（Mathematical definition）</font>

![这里写图片描述](http://img.blog.csdn.net/20170702182145679?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvVGhlX2xhc3Rlc3Q=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

$\Theta^{(j)}$是一个矩阵，表示第$j$层所对应的权重(weights)；也就是说每一层都有这样一个矩阵，通过该层的活化单元$a^{(j)}$（输入层也可视为活化单元）与该层对应的权重$\Theta^{(j)}$进行线性运算来得到下一层的活化单元$a^{(j+1)}$。

比如一个$4\times3$的权重矩阵：

$$\Theta^{(i)}=\begin{bmatrix} \Theta^{(i)}_{10}&\Theta^{(i)}_{11}&\Theta^{(i)}_{12}\\\Theta^{(i)}_{20}&\Theta^{(i)}_{21}&\Theta^{(i)}_{22}\\\Theta^{(i)}_{30}&\Theta^{(i)}_{31}&\Theta^{(i)}_{32}\\\Theta^{(i)}_{40}&\Theta^{(i)}_{41}&\Theta^{(i)}_{42}\\\end{bmatrix}$$


> The values for each of the "activation" nodes is obtained as follows:

\begin{align*} a_1^{(2)} = g(\Theta_{10}^{(1)}x_0 + \Theta_{11}^{(1)}x_1 + \Theta_{12}^{(1)}x_2 + \Theta_{13}^{(1)}x_3) \newline a_2^{(2)} = g(\Theta_{20}^{(1)}x_0 + \Theta_{21}^{(1)}x_1 + \Theta_{22}^{(1)}x_2 + \Theta_{23}^{(1)}x_3) \newline a_3^{(2)} = g(\Theta_{30}^{(1)}x_0 + \Theta_{31}^{(1)}x_1 + \Theta_{32}^{(1)}x_2 + \Theta_{33}^{(1)}x_3) \newline h_\Theta(x) = a_1^{(3)} = g(\Theta_{10}^{(2)}a_0^{(2)} + \Theta_{11}^{(2)}a_1^{(2)} + \Theta_{12}^{(2)}a_2^{(2)} + \Theta_{13}^{(2)}a_3^{(2)}) \newline \end{align*}


> This is saying that we compute our activation nodes by using a 3×4 matrix of parameters. We apply each row of the parameters to our inputs to obtain the value for one activation node. Our hypothesis output is the logistic function applied to the sum of the values of our activation nodes, which have been multiplied by yet another parameter matrix $\Theta^{(2)}$ containing the weights for our second layer of nodes.


**因此我们可以看出，其实每一个活化单元的值都是以上一层作为输入，以上一层的权重矩阵的对应一行为参数，然后进行关于函数$g(z)$的映射，而$g(z)$恰恰是之前所学的逻辑回归假设函数的表达形式。所以，神经网络可以认为是若干逻辑回归模型所组成的**（此观点为博主个人的主观猜想）。

> Each layer gets its own matrix of weights, $\Theta^{(j)}$. The dimensions of these matrices of weights is determined as follows:
> 
 If network has $s_j$ units in layer $j$ and $s_{j+1}$ units in layer $j+1$, then $\Theta^{(j)}$ will be of dimension $s_{j+1} \times (s_j + 1)$.
 
 
![这里写图片描述](http://img.blog.csdn.net/20170702190216051?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvVGhlX2xhc3Rlc3Q=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

<font size = 5 color = red> 4.矢量化（Vectorized implementation）</font>

先做出如下定义：

\begin{align*} a_1^{(2)} = g(\Theta_{10}^{(1)}x_0 + \Theta_{11}^{(1)}x_1 + \Theta_{12}^{(1)}x_2 + \Theta_{13}^{(1)}x_3) = \color{red}{g(z^{(2)}_1)}
 \newline a_2^{(2)} = g(\Theta_{20}^{(1)}x_0 + \Theta_{21}^{(1)}x_1 + \Theta_{22}^{(1)}x_2 + \Theta_{23}^{(1)}x_3) = \color{red}{g(z^{(2)}_2)}
 \newline a_3^{(2)} = g(\Theta_{30}^{(1)}x_0 + \Theta_{31}^{(1)}x_1 + \Theta_{32}^{(1)}x_2 + \Theta_{33}^{(1)}x_3) = \color{red}{g(z^{(2)}_3)}
 \newline h_\Theta(x) = a_1^{(3)} = g(\Theta_{10}^{(2)}a_0^{(2)} + \Theta_{11}^{(2)}a_1^{(2)} + \Theta_{12}^{(2)}a_2^{(2)} + \Theta_{13}^{(2)}a_3^{(2)}) = \color{red}{g(z^{(3)})}\newline \end{align*}

令：
$ a^{(1)} = x=\begin{bmatrix} x_0 \\ x_1\\ x_2 \\ x_3\end{bmatrix}, z^{(2)}=\begin{bmatrix} z^{(2)}_1 \\ z^{(2)}_1 \\ z^{(2)}_3\end{bmatrix}= \Theta^{(1)} a^{(1)}\implies a^{(2)} = g(z^{(2)}) $ 

setting $a^{(2)}_0 =1 \implies z^{(3)} = \Theta^{(2)}a^{(2)} \implies h_\Theta(x)=a^{(3)}=g(z^{(3)})$

**Sumarize:**

