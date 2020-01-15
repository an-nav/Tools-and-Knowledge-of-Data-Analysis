Markov Chain Monte Carlo

## 概述

MCMC方法由两个MC组成即**蒙特卡洛算法(Monte Carlo Simulation)**和**马尔科夫链(Markov Chain )**本篇将先讲述蒙特卡洛方法再讲述马尔科夫链，最后介绍他们结合之下的两个采样算法**M-H 采样(Metropolis-Hastings sampling)**和**吉布斯采样(Gibbs sampling)**，当然要注意的一点是MCMC是一种思想或者说一种方法论，M-H采样和Gibbs 采样是MCMC思想的一个体现。

## 蒙特卡洛算法 (Monte Carlo Simulation)

### 蒙特卡洛概述

**蒙特卡洛算法**的名称来自于赌场”蒙特卡洛“因为其本质上是对于随机过程的一种模拟，初诞生于美国的曼哈顿计划，与其相同的另一个名称来自赌场的算法是**拉斯维加斯算法**，两者虽然都是一种随机模拟但是有很大区别。

蒙特卡洛算法：采样越多，越接近最优解，即每次迭代都在靠近最优解(尽量找好的，不一定找到)

拉斯维加斯算法：采样越多，越可能找到最优解，即直接想找到最优解(尽量找最好的，但不一定能找到)



方法诞生之初是用来求解复杂的积分问题如下图：  

<img src="../Resource/MCMC_1.PNG" alt="积分图" style="zoom:50%;" />

我们要求积分：
$$
\theta=\int_a^bf(x)dx\tag{1}
$$
如果$f(x)$很难表示或者说这个函数十分复杂那么我们很难直接去求出这个积分。这时候一个简单的近似解法就是在函数上选取N个点$f(x_i)$取平均然后乘以$b-a$即:
$$
\theta=\frac{b-a}{N}\sum_{i=1}^Nf(x_i)\tag{2}
$$
但是这种方法是假设所有的$x$是在$(a,b)$区间均匀分布的，但是我们可以看到对于积分而已函数陡峭部分对于积分的贡献要小于函数平缓部分，也就是说$x$并不应该被认为是均匀分布的，而是服从一定的概率分布$p(x)$，让其在对于积分贡献大的地方出现的更频繁。

由此我们的问题可以这样转化:
$$
\begin{align}
\theta&=\int_a^bf(x)dx\\
&=\int_a^b\frac{f(x)}{p(x)}p(x)dx\\
&=E_{p(x)}[\frac{f(x)}{p(x)}]\\
&\approx\frac{1}{N}\sum_{i=1}^N\frac{f(x_i)}{p(x_i)}\tag{3}
\end{align}
$$
式3就是蒙特卡洛方法的一般形式了，如果我们认为$p(x_i)$是在$(a,b)$上的均分布那么式子3就为：
$$
\frac{b-a}{N}\sum_{i=1}^Nf(x_i)
$$
可以看到我们最开始求解的式2的情况即是我们认为$p(x)$为均分布情况下的蒙特卡洛方法得到的结果，由此我们将问题转化到了求解分布$p(x)$上面来。

### 采样概率分布与拒绝采样

对于我们常见的分布来说，计算机能够十分方便的进行采样如0,1之间的均分布$uniform(0,1)$可以通过线性同余发生器很方便的采样，再到稍微复杂一些的正态分布，也可以通过一些变换得到。例如要采样满足二位正态分布的样本$(z_1,z_2)$我们可以先采样满足$uniform(0,1)$的样本$(x_1,x_2)$然后通过Box-Muller变换得到：
$$
\begin{align}
z_1=\sqrt{-2lnx_1}cos(2\pi x_2)\\
z_2=\sqrt{-2lnx_2}cos(2\pi x_1)\\
\end{align}
$$
但是极大多数情况下我们面对的$p(x)$都非常非常的复杂难以直接通过变换得到，因此我们想到通过一个我们已知的较为简单的分布来采样我们的目标分布，这种方法即**拒绝采样(rejection sampling)**。

拒绝采样的思想见下图:

<img src="../Resource/MCMC_2.png" alt="rejection sampling" style="zoom: 80%;" />





我们首先构造一个常见分布$q(z)$称其为proposal distribution，然后让$q(z)$乘以一个常数$k$，这样就能保证$kq(z)$始终能在$p(z)$的上方。

其采用步骤如下：

- 从 proposal distribution $q(z)$中生成一个样本$z_0$
- 从$uniform(0,1)$中生成一个$u$
- if $u<\frac{p(z_0)}{kq(z_0)}$
  - 接受样本$z_0$
- else 
  - 拒绝样本$z_0$

重复$M$次后我们即可得到一系列目标分布$p(x)$的样本。但是这种方法面临了一个困境，有其判断条件可以知道，如果$kq(z)$和$p(z)$相差的很远那么采样的效率将会非常的低下可能几乎无法完成，而在高维的情况下找到一个很接近目标分布$p(x)$而且简单的proposal distribution$q(z)$是非常困难或者是不可能的。因此求解$p(x)$依然是一个大问题，所幸下一节的马尔科夫链可以提供求解$p(x)$的思路。

## 马尔科夫链(Markov Chain )

### 马尔科夫链概述

离散的具有马尔科夫性质的随机过程称为马尔科夫链。马尔科夫性质即当前状态依赖且仅依赖于其前一个状态，从概率图的角度来说，假设我们的状态是序列$X_1,X_2,...,X_t,X_{t+1}$那么我们在$X_{t+1}$状态的条件概率只依赖于$X_t$即：
$$
P(X_{t+1}|X_1,X_2,...,X_t)=P(X_{t+1}|X_t)
$$
可见其本身非常简单，而正是因为这样其能够大大简化模型的复杂度。

### 马尔科夫链状态转移矩阵

既然马尔科夫链的当前状态只依赖于其前一个状态，那么我们就可以写出模型中每个状态之间的状态转移矩阵P。

![马尔科夫链图](../Resource/MCMC_3.PNG)

在上图股市模型中我们定义状态1为牛市、状态2为熊市、状态3为横盘，而状态转移矩阵P中的某一元素$P_{ij}$表示从i状态转换到 j 状态的概率，如从牛市（状态1）转换到熊市（状态2）的概率是0.075，那么这个股市模型的状态转移矩阵P为：
$$
P=\left(
 \begin{matrix}
   0.9&0.075&0.025\\
   0.15&0.8&0.05\\
   0.25&0.252&0.5
  \end{matrix} 
\right)
$$
得到了马尔科夫链的状态转移矩阵，接下来我们来看看它的一些性质。这里引用刘建平Pinard博客中的一个例子。我们初始给定股市三个状态一个概率分布比如:$[0.4,0.4,0.2]$即0.4的概率是牛市，0.4的概率是熊市，0.2的概率横盘，我们设定这个时刻为状态$t_0$然后我们用状态转移矩阵来计算$t_1,t_2,t_3,...t_n$时刻的概率分布。

```Python
import numpy as np
matrix = np.matrix([[0.9,0.075,0.025],[0.15,0.8,0.05],[0.25,0.25,0.5]], dtype=float)
vector1 = np.matrix([[0.4,0.4,0.2]], dtype=float)
for i in range(100):
    vector1 = vector1*matrix
    print ("Current round:" , i+1)
    print (vector1)
```

输出结果如下:

```
Current round: 1
[[0.47 0.4  0.13]]
Current round: 2
[[0.5155  0.38775 0.09675]]
Current round: 3
[[0.5463  0.37305 0.08065]]
Current round: 4
[[0.56779  0.359575 0.072635]]
Current round: 5
[[0.583106 0.348403 0.068491]]
.......
Current round: 58
[[0.62499999 0.3125     0.0625    ]]
Current round: 59
[[0.625  0.3125 0.0625]]
Current round: 60
[[0.625  0.3125 0.0625]]
Current round: 61
[[0.625  0.3125 0.0625]]
Current round: 62
[[0.625  0.3125 0.0625]]
Current round: 63
[[0.625  0.3125 0.0625]]
........
```

从上面的输出结果可以看到在第59轮之后我们的状态$t$就收敛到一个固定的概率分布了，我们再换一个初始分布来看看比如：$t_0=[0.3,0.4,0.3]$

在带入上面的程序得到结果如下:

```
Current round: 1
[[0.405  0.4175 0.1775]]
Current round: 2
[[0.4715  0.40875 0.11975]]
Current round: 3
[[0.5156 0.3923 0.0921]]
Current round: 4
[[0.54591  0.375535 0.078555]]
Current round: 5
[[0.567288 0.36101  0.071702]]
......
Current round: 59
[[0.62499999 0.3125     0.0625    ]]
Current round: 60
[[0.625  0.3125 0.0625]]
Current round: 61
[[0.625  0.3125 0.0625]]
Current round: 62
[[0.625  0.3125 0.0625]]
Current round: 63
[[0.625  0.3125 0.0625]]
......
```

我们的状态$t$依然收敛到了同一个概率分布。从上面的例子可以发现我们选取了不同的初始状态却收敛到了同一个状态，那这不正说明在马尔科夫链中我们可以任意选取初始状态最终都会收敛到一个稳定的状态吗。这样在我们采样的过程中我们就可以从任意概率分布开始经过一系列的状态转移最终采样我们既定的目标分布中的样本。

除了任意初始状态会收敛到一个稳定状态，状态转移矩阵本身也具有性质，对于一个确定的状态转移矩阵$P$，它的$n$次幂$P^n$在$n$大于一定数之后也是确定的，依然用上面的状态转移矩阵为例：

```python
matrix = np.matrix([[0.9,0.075,0.025],[0.15,0.8,0.05],[0.25,0.25,0.5]], dtype=float)
for i in range(10):
    matrix = matrix*matrix
    print "Current round:" , i+1
    print matrix
```

结果如下：

```
Current round: 1
[[0.8275  0.13375 0.03875]
 [0.2675  0.66375 0.06875]
 [0.3875  0.34375 0.26875]]
Current round: 2
[[0.73555  0.212775 0.051675]
 [0.42555  0.499975 0.074475]
 [0.51675  0.372375 0.110875]]
Current round: 3
[[0.65828326 0.28213131 0.05958543]
 [0.56426262 0.36825403 0.06748335]
 [0.5958543  0.33741675 0.06672895]]
Current round: 4
[[0.62803724 0.30972343 0.06223933]
 [0.61944687 0.3175772  0.06297594]
 [0.6223933  0.3148797  0.062727  ]]
Current round: 5
[[0.62502532 0.31247685 0.06249783]
 [0.6249537  0.31254233 0.06250397]
 [0.62497828 0.31251986 0.06250186]]
Current round: 6
[[0.625  0.3125 0.0625]
 [0.625  0.3125 0.0625]
 [0.625  0.3125 0.0625]]
Current round: 7
[[0.625  0.3125 0.0625]
 [0.625  0.3125 0.0625]
 [0.625  0.3125 0.0625]]
Current round: 8
[[0.625  0.3125 0.0625]
 [0.625  0.3125 0.0625]
 [0.625  0.3125 0.0625]]
```

可以看到在第6轮之后整个状态转移矩阵也收敛到了稳定的状态且每行都相同这和我们之前得到的平稳分布是一直的。上面提到的两个马尔科夫链的性质不光在**离散状态**时成立，**连续状态**也成立。

OK，我们用数学语言来描述一下马尔科夫链的收敛性质:

如果一个非周期的马尔科夫链有状态转移矩阵PP, 并且它的任何两个状态是连通的，那么$\lim\limits_{i\rightarrow\infty}P_{ij}$与$i$无关，我们有：

1）
$$
\lim\limits_{n\rightarrow\infty}P^n_{ij}=\pi(j)
$$
2）
$$
\lim\limits_{n\rightarrow}P^n=\left(
\begin{matrix}
\pi(1)&\pi(2)&\pi(3)&...&\pi(j)&...\\
\pi(1)&\pi(2)&\pi(3)&...&\pi(j)&...\\
...&...&...&...&...&...\\
...&...&...&...&...&...\\
\pi(1)&\pi(2)&\pi(3)&...&\pi(j)&...\\
...&...&...&...&...&...\\
\end{matrix}
\right)
$$
3)
$$
\pi(j)=\sum_{i=0}^\infty \pi(i)P_{ij}
$$
4) π是方程πP=ππP=π的唯一非负解，其中：
$$
\begin{align}
&\pi=[\pi(1),\pi(2)&,\pi(3),...,\pi(j),...]\\
&\sum_{i=0}^\infty \pi(i)=1
\end{align}
$$
上面的性质中需要解释的有：

1. 非周期的马尔科夫链：这个主要是指马尔科夫链的状态转化不是循环的，如果是循环的则永远不会收敛。幸运的是我们遇到的马尔科夫链一般都是非周期性的。用数学方式表述则是：对于任意某一状态$i$，$d$为集合$\{n∣n≥1,P^n_{ij}>0\}$ 的最大公约数，如果 $d=1$ ，则该状态为非周期的
2. 任何两个状态是连通的：这个指的是从任意一个状态可以通过有限步到达其他的任意一个状态，不会出现条件概率一直为0导致不可达的情况。
3. 马尔科夫链的状态数可以是有限的，也可以是无限的。因此可以用于连续概率分布和离散概率分布。、
4. $\pi$通常称为马尔科夫链的平稳分布。

有了马尔科夫链我们只需找到合适的状态转移矩阵$P$就可以通过数次迭代来得到平稳的目标分布了，但是问题就是这个状态转移矩阵$P$不好找，所幸M-H 采样和Gibbs采样解决了这一问题。

## M-H采样 (Metropolis-Hastings sampling)

### 细致平稳条件 (detailed balance)

解决寻找平稳分布$\pi$之前，我们还需先看看马尔科夫链的的细致平稳条件其定义如下：

如果非周期马尔科夫链的状态转移矩阵$P$和概率分布$π(x)$对于所有的$i,j$满足：
$$
\pi(i)P_{ij}=\pi (j)P_{ji}
$$
则称概率分布$\pi(x)$为状态转移矩阵$P$的平稳分布。由此可以得到细致平稳条件是平稳分布的充分条件，证明如下:

由细致平稳条件可得:
$$
\begin{align}
\sum_i\pi(i)P_{ij}=\sum_i\pi(j)P_{ji}=\pi(j)\sum_iP_{ji}=\pi(j)

\end{align}
$$
用矩阵表示即:
$$
\pi P=\pi
$$
也就是说满足马尔科夫链的收敛性质，但是不幸的是我们仅仅通过细致平稳条件很难找到一个合适的状态转移矩阵$P$。比如我们的目标分布是$\pi(x)$我们随机一个状态转移矩阵$Q$，大部分情况下我们无法满足细致平稳条件即:
$$
\pi(i)Q_{ij}\neq\pi (j)Q_{ji}
$$
那如何让这个等式成立来使我们得到平稳分布呢？MCMC采样就解决了这一问题。

### MCMC采样

一般情况下，对于我们的目标分布$\pi(x)$我们找不到一个$Q$来让其满足细致平稳条件即:
$$
\pi(i)Q_{ij}\neq\pi (j)Q_{ji}
$$
但是我们可以对上式进行一个改造引入一个$a(i,j)$来使等号成立：
$$
\pi(i)Q_{ij}a(i,j)=\pi(j)Q_{ji}a(j,i)
$$
为此我们只需要利用对称就可以使得等号成立，即$a(i,j)$满足：
$$
\begin{align}
a(i,j)=\pi(j)Q_{ij}\\
a(j,i)=\pi(i)Q_{ji}
\end{align}
$$
这样我们就得到了目标分布$\pi(x)$的马尔科夫链状态转移矩阵P：
$$
P_{ij}=Q_{ij}a(i,j)
$$
这样我们的状态转移矩阵就可以通过任意一个$Q$来得到了，$a(i,j)$一般称之为接受率是一个$[0,1]$之间的概率值即目标状态转移矩阵$P$可以通过常见的状态转移矩阵$Q$通过一个的接受概率得到，这一点很像我们之前讲过的拒接采样，思路上来说两者是一致的。

接下来总结一下MCMC采样的过程：

1. 任意选定一个马尔科夫链状态转移矩阵$Q$，平稳分布$\pi(x)$，设定状态转移阈值$n_1$，设定需要采样的个数$n_2$
2. 从简单概率分布中采样得到一个初始$x_0$

3. for t=0 to n1+n2−1:

　　　a. 从条件概率分布$Q(x|x_t)$中采样得到样本$x_*$

　        b. 从均匀分布采样$u∼uniform[0,1]$

　        c. 如果$u<\alpha(x_t,x_∗)=π(x_∗)Q(x_∗,x_t)$, 则接受转移$x_t→x_∗$，即$x_{t+1}=x_∗$

　　　d. 否则不接受转移，即$x_{t+1}=x_t$

样本集$(x_{n1},x_{n1+1},...,x_{n1+n2−1})$即为我们需要的平稳分布对应的样本集。

接下来先说两个名词解释方便理解下面的叙述

- 从我们的迭代开始到收敛到平稳分布的这个过程称为burn-in
- burn-in所花费的时间称为mixing-time

上述MCMC采样的问题在于在c步这的$\alpha(x_t,x_∗)$可能非常小例如0.1那么我们的采样效率就非常的低也就是说mixing-time会非常的长，这样是非常浪费计算资源的，再一个就是$\alpha(x_t,x_∗)$本身在高维度下可能非常的难以计算由此终于轮到本节的主角M-H采样登场了。

### M-H 采样

首先回到MCMC采样中的细致平稳条件，我们假设$a(i,j)=0.1; a(j,i)=0.2$:
$$
π(i)Q_{ij}×0.1=π(j)Q_{ji}×0.2
$$
而采样效率低就是因为接受率过小，那么我们可以将上式两边同时扩大五倍，将$a(j,i)$放大到1：
$$
π(i)Q_{ij}×0.5=π(j)Q_{ji}×1
$$
由此我们的接受率就可以进行如下的改造：
$$
a(i,j)=min\{\frac{\pi (j)Q_{ji}}{\pi(i)Q_{ij}},1\}
$$


这样我们的接受率就被放大，采样效率也得以提升。

通过改造M-H采样的算法如下：

1. 任意选定一个马尔科夫链状态转移矩阵$Q$，平稳分布$\pi(x)$，设定状态转移阈值$n_1$，设定需要采样的个数$n_2$
2. 从简单概率分布中采样得到一个初始$x_0$

3. for t=0 to n1+n2−1:

　　　a. 从条件概率分布$Q(x|x_t)$中采样得到样本$x_*$

　        b. 从均匀分布采样$u∼uniform[0,1]$

　        c. 如果$u<min\{\frac{\pi (j)Q_{ji}}{\pi(i)Q_{ij}},1\}$, 则接受转移$x_t→x_∗$，即$x_{t+1}=x_∗$

　　　d. 否则不接受转移，即$x_{t+1}=x_t$

样本集$(x_{n1},x_{n1+1},...,x_{n1+n2−1})$即为我们需要的平稳分布对应的样本集。

如果我们选择的状态转移矩阵是对称的那么接受率可以进一步化简为：
$$
a(i,j)=min\{\frac{\pi (j)}{\pi(i)},1\}
$$

### M-H采样例子

我们用M-H采样一个beta分布为例

```python
import random
import math
import numpy as np
from scipy.stats import norm
import scipy.special as ss
import matplotlib.pyplot as plt

def MH_sampling(n, p):
    curr = random.uniform(0, 1)
    state = []
    for i in range(0, n):
        state.append(curr)
        next_state = norm.rvs(loc=curr)
        alpha = min((p(next_state)/p(curr)),1)
        u = random.uniform(0, 1)
        if u < alpha:
            curr = next_state
    return state

# Beta分布概率密度函数
def beta(x):
    a=0.5
    b=0.6
    return (1.0 / ss.beta(a,b)) * x**(a-1) * (1-x)**(b-1)

Ly = []
Lx = []
i_list = np.mgrid[0:1:100j]
for i in i_list:
    Lx.append(i)
    Ly.append(beta(i))
    
plt.plot(Lx, Ly, label="Real Distribution")
plt.hist(MH_sampling(100000,beta),density=1,bins=25, histtype='step',label="Simulated_MCMC")
plt.legend()
plt.show()
```

结果如图：

![M-H采样beta分布](../Resource/MCMC_4.PNG)

从结果图看M-H 采样能很好的工作并采样分布了，但是在大数据时代，M-H采样面临着两大难题：

1. 我们的数据特征非常的多，M-H采样由于接受率计算式$\frac{π(j)Q(j,i)}{π(i)Q(i,j)}$的存在，在高维时需要的计算时间非常的可观，算法效率很低。同时$a(i,j);a(j,i)$一般小于1，有时候辛苦计算出来却被拒绝了。能不能做到不拒绝转移呢？
2. 由于特征维度大，很多时候我们甚至很难求出目标的各特征维度联合分布，但是可以方便求出各个特征之间的条件概率分布。这时候我们能不能只有各维度之间条件概率分布的情况下方便的采样呢？

Gibbs采样解决了上面两个问题，下一节我们就来讨论Gibbs采样。

## Gibbs 采样 (Gibbs sapmling）

上面说到在高维的情况下，M-H采样存在两个问题，1）接受率导致算法效率低 2）高维联合概率分布十分难以计算。而Gibbs采样能很好的解决这两个问题。我们依然从细致平稳条件入手。

### 二维Gibbs 采样

先看看二维的情况，假设有一个概率分布$p(x,y)$，考察$x$坐标相同的两个点$A(x_1 ,y_1),B(x_1,y_2)$我们有如下发现：
$$
\begin{align}
P(x_1,y_1)P(y_2|x_1)=P(y_1|x_1)P(x_1)P(y_2|x_1)\\
P(x_1,y_2)P(y_1|x_1)=P(y_2|x_1)P(x_1)P(y_1|x_1)
\end{align}
$$
于是我们可以得到：
$$
P(x_1,y_1)P(y_2|x_1)=P(x_1,y_2)P(y_1|x_1)
$$
即
$$
P(A)P(y_2|x_1)=P(B)P(y_1|x_1)
$$
基于以上等式，我们发现，在$x=x_1$这条平行于![y](https://math.drivingc.com/?latex=y)轴的直线上，如果使用条件分布$P(y|x_1)$做为任何两个点之间的转移概率，那么任何两个点之间的转移满足细致平稳条件。同样的，如果我们在$y=y_1$这条直线上任意取两个点$A(x_1,y_1),C(x_2,y_1)$也有如下等式
$$
P(A)P(x_2|y_1)=P(C)P(x_1|y_1)
$$


<img src="../Resource/MCMC_5.PNG" alt="Gibbs sampling" style="zoom:80%;" />

由此我们可以构造平面上任意两点的状态转移矩阵Q
$$
\begin{align}
Q(A\rightarrow B)&=P(y_b|x_1)\quad &if\quad x_A=x_B=x_1\\
Q(A\rightarrow C)&=P(x_c|y_1)\quad &if\quad y_A=y_C=y_1\\
Q(A\rightarrow D)&=0    &else
\end{align}
$$
有了如上概率转移矩阵Q，我们很容易就可以验证对于平面上任意两点$X,Y$都满足细致平稳条件
$$
P(X)P(X\rightarrow Y)=P(Y)P(Y\rightarrow X)
$$
利用上一节找到的状态转移矩阵，我们就得到了二维Gibbs采样，这个采样需要两个维度之间的条件概率。具体过程如下：

1. 输入平稳分布$π(x_1,x_2)$，设定状态转移次数阈值$n_1$，需要的样本个数$n_2$

2. 随机初始化初始状态值$x^{(0)}_1$和$x^{(0)}_2$

3. $for\quad t=0\quad to \quad n_1+n_2−1$: 

   ​	1. 从条件概率分布$P(x_2|x^{(t)}_1)$中采样得到样本$x^{(t+1)}_2$

　　　2. 从条件概率分布$P(x_1|x^{(t+1)}_2)$中采样得到样本$x^{(t+1)}_1$

　　　　样本集${(x^{(n1)}_1,x^{(n1)}_2),(x^{(n1+1)}_1,x^{(n1+1)}_2),...,(x^{(n1+n2−1)}_1,x^{(n1+n2−1)}_2)}$即为我们需要的平稳分布对应的样本集。

　　　　整个采样过程中，我们通过轮换坐标轴，采样的过程为：
$$
(x^{(1)}_1,x^{(1)}_2)→(x^{(1)}_1,x^{(2)}_2)→(x^{(2)}_1,x^{(2)}_2)→...→(x^{(n1+n2−1)}_1,x^{(n1+n2−1)}_2)
$$
用下图可以很直观的看出，采样是在两个坐标轴上不停的轮换的。当然，坐标轴轮换不是必须的，我们也可以每次随机选择一个坐标轴进行采样。不过常用的Gibbs采样的实现都是基于坐标轴轮换的。

<img src="../Resource/MCMC_6.PNG" alt="2D Gibbs sampling"  />

### Gibbs采样的特点

从二维Gibbs采样可以发现Gibbs本质上来说是一种特殊的M-H采样，主要有两处特殊:

1. 马尔科夫链中的状态转移矩阵$Q$就是我们要采样的目标分布$P$的条件概率分布
2. 其接受率恒为1  

第一点很好理解，在二维Gibbs采样中我们可以看到每次转移都是沿坐标轴或者说平行于坐标轴的直线进行转移因此我们的采样过程就是在目标分布$P(x_1,x_2,...,x_n)$中固定其他维度只转移某一维$i$ 即上图所示的坐标轴轮换过程，所以转移矩阵$Q$是我们目标分布$P$的条件概率分布。

而对于第二点我们可以从M-H采样对于接受率的定义出发则有:
$$
\begin{align}
\frac{P(x^*)Q(x^*\rightarrow x)}{P(x)Q(x\rightarrow x^*)}=\frac{P(x^*_i)P(x_i|x_{-i}^*)}{P(x_i)P(x_i^*|x_{-i})}&=\frac{P(x_i^*|x^*_{-i})P(x^*_{-i})P(x_i|x_{-i}^*)}{P(x_i|x_{-i})P(x_{-i})P(x^*_i|x_{-i})}\tag{4}\\
&=\frac{P(x_i^*|x^*_{-i})P(x^*_{-i})P(x_i|x_{-i}^*)}{P(x_i|x^*_{-i})P(x^*_{-i})P(x^*_i|x^*_{-i})}\tag{5}\\
&=1
\end{align}
$$
其中：

- $x_i$表示$t$时刻对于概率分布$P$中第$i$的采样
- $x^*_i$表示$t+1$时刻对于概率分布$P$中第$i$的采样
- $x_{-i}$表示$t$时刻概率分布P中除$i$以外的其他维度$\rightarrow x_{-i}=x_1,x_2,...,x_{i-1},x_{i+1},...,x_n$
- $x^*_{-i}$表示$t+1$时刻概率分布P中除$i$以外的其他维度

解释一下从式4到式5

从上面的符号说明可以看出$x_{-i}$和$x_{-i}^*$都是概率分布P中除$i$以外的其他维度而这些维度再采样$x_i$的时候是不变的即:
$$
x_{-i}=x^*_{-i}
$$
因此可以得到Gibbs采样的接受率$\alpha=1$。

### 多维Gibbs采样

根据上述Gibbs采样的特点我们很容易就能从二维拓展到维多，对于$n$为概率分布$\pi(x_1,x_2,...,x_n)$我们可以通过在$n$个坐标轴上轮换采样具体如下：

1. 输入平稳分布$π(x_1,x_2,...,x_n)$，设定状态转移次数阈值$n_1$，需要的样本个数$n_2$

2. 随机初始化初始状态值$x^{(0)}_1,x^{(0)}_2,...,x_n^{(0)}$

3. $for\quad t=0\quad to \quad n_1+n_2−1$: 
   1. 从条件概率分布$P(x_1|x^{(t)}_2,x^{(t)}_3,...,x^{(t)}_n)$中采样得到样本$x^{(t+1)}_1$
   2. 从条件概率分布$P(x_2|x^{(t+1)}_1,x^{(t)}_3,...,x^{(t)}_n)$中采样得到样本 $x^{(t+1)}_2$
   3. 从条件概率分布$P(x_3|x^{(t+1)}_1,x^{(t+1)}_2,...,x^{(t)}_n)$中采样得到样本 $x^{(t+1)}_3$
   4. ...
   5. 从条件概率分布$P(x_n|x^{(t+1)}_1,x^{(t+1)}_2,...,x^{(t+1)}_{n-1})$中采样得到样本 $x^{(t+1)}_n$

样本集${(x^{(n1)}_1,x^{(n1)}_2,...,x_n^{(n1)}),(x^{(n1+1)}_1,x^{(n1+1)}_2,...,x_n^{(n1+1)}),...,(x^{(n1+n2−1)}_1,x^{(n1+n2−1)}_2,...,x_n^{(n1+n2-1)}})$即为我们需要的平稳分布对应的样本集。

OK，至此Gibbs sampling介绍的已经差不多了，引用*刘建平Pinard*的一个二维Gibbs采样的例子结束本文吧。

假设我们要采样二维正态分布$Norm(\mu,\Sigma)$:
$$
\begin{align}
\mu&=(\mu_1,\mu_2)=(5,-1)\\
\Sigma&=\left(
\begin{matrix}
\sigma_1^2&\rho\sigma_1\sigma_2\\
\rho\sigma_1\sigma_2&\sigma_2^2
\end{matrix}
\right)=\left(
\begin{matrix}
1&1\\
1&4
\end{matrix}
\right)
\end{align}
$$
采样过程中的条件概率为:
$$
P(x_1|x_2)=Norm(\mu_1+\rho\sigma_1/\sigma_2(x_2-\mu_2),(1-\rho^2)\sigma_1^2)\\
P(x_2|x_1)=Norm(\mu_2+\rho\sigma_2/\sigma_1(x_1-\mu_1),(1-\rho^2)\sigma_2^2)
$$

```python
#Gibbs sampling
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
import random
import math
from matplotlib import pyplot as plt
plt.style.use('ggplot')
samplesource = multivariate_normal(mean = [5, -1], cov = [[1, 1], [1, 4]])

def p_ygivenx(x, m1, m2, s1, s2):
    return (random.normalvariate(m2 + rho * s2 / s1 * (x - m1), math.sqrt((1 - rho ** 2) * (s2**2))))

def p_xgiveny(y, m1, m2, s1, s2):
    return (random.normalvariate(m1 + rho * s1 / s2 * (y - m2), math.sqrt((1 - rho ** 2) * (s1**2))))

N = 5000*20
x_res = []
y_res = []
z_res = []
m1 = 5
m2 = -1
s1 = 1
s2 = 2

rho = 0.5
y = m2

for i in range(N):
    x = p_xgiveny(y, m1, m2, s1, s2)
    y = p_ygivenx(x, m1, m2, s1, s2)
    z = samplesource.pdf([x,y])
    x_res.append(x)
    y_res.append(y)
    z_res.append(z)

num_bins = 50
plt.hist(x_res, num_bins, density=1, facecolor='green', alpha=0.5)
plt.hist(y_res, num_bins, density=1, facecolor='red', alpha=0.5)
plt.title('Histogram')
plt.show()
```

两个正态分布各自采样结果如图：

![gibbs sampling 1d norm distribution](../Resource/MCMC_7.PNG)

采样到的二维正态分布样本集如下:

```python
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
ax.scatter(x_res, y_res, z_res,marker='o', color='b')
plt.show()
```

![gibbs sampling 2d norm distribution](../Resource/MCMC_8.PNG)

## 有待思考

1. 马尔科夫链为何收敛到平稳分布的证明
2. 为何可以直接认为马尔科夫链收敛到的平稳分布就是我们的目标分布？

## 参考

1. [博客园-MCMC系列-刘建平Pinard](https://www.cnblogs.com/pinard/p/6625739.html)
2. [DrivingC-MCMC和Gibbs Sampling](https://drivingc.com/p/5c25bc6f4b0f2b1e262c20ea)
3. [CSDN-Box-Muller变换原理详解-帅帅GO](https://blog.csdn.net/weixin_41793877/article/details/84700875)
4. [BiliBili-机器学习-白板推导系列(十三)-MCMC（Markov Chain Monte Carlo）-shuhuai008](https://www.bilibili.com/video/av32430563?p=1)
5. [wikipedia-Markov chain](https://en.wikipedia.org/wiki/Markov_chain)