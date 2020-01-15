# 狄利克雷分布(Dirichlet distribution)

## 简单概念

狄利克雷分布是Beta分布向高维的扩展，当维度趋于无穷时称为狄利克雷过程。

## 概率密度函数

$$
\begin{align}
P(x_i|\alpha_i)&=\frac{1}{B(\alpha_i)}\prod_{i=1}^{K}x_i^{\alpha_i-1}\\
&=\frac{\Gamma(\sum_{i=1}^k\alpha_i)}{\prod_{i=1}^k\Gamma(\alpha_i)}\prod_{i=1}^{K}x_i^{\alpha_i-1}\tag{1}
\end{align}
$$

## 期望

$$
E[x_i]=\frac{x_i}{\sum_{k=1}^Kx_k}
$$

## 方差

$$
\begin{align}
&Var[x_i]=\frac{\overline{a_i}(1-\overline{a_i})}{\alpha_0+1}\\
&where\quad \overline{\alpha_i}=\frac{x_i}{\sum_{k=1}^Kx_k}\\
&\quad\quad\quad\quad \alpha_0=\sum_{k=1}^Kx_k
\end{align}
$$

## 共轭先验分布

狄利克雷分布与多项式分布是共轭先验分布关系。

证明：

设有一参数为$\theta_i$的多项式分布:
$$
P(x_i|\theta_i)=\frac{\Gamma(\sum_{i=1}^{k}x_i+1)}{\prod_{i=1}^{k}\Gamma(x_i+1)}\prod_{i=1}^{k}\theta_i^{x_i}
$$
而参数$\theta_i$服从狄利克雷分布即$\theta_i\sim Dir(\theta_i|\alpha_i)$则狄利克雷分布为:
$$
P(\theta_i|\alpha_i)=\frac{1}{B(\alpha_i)}\prod_{i=1}^k\theta_{i}^{\alpha_i-1}
$$
我们要求的是后验$P(\theta_i|x_i,\alpha_i)$根据贝叶斯公式可得:
$$
\begin{align}
P(\theta_i|x_i,\alpha_i)&=P(\theta_i|\alpha_i)\frac{P(x_i|\theta_i)}{\int_0^1P(x_i,\theta_i|\alpha_i)d\theta}\\
&=P(\theta_i|\alpha_i)\frac{P(x_i|\theta_i)}{P(x_i|\alpha_i)}\tag{2}\\
P(\theta_i,x_i|\alpha_i)&=P(\theta_i|\alpha_i)P(x_i|\theta_i)\tag{3}\\
&=\frac{\Gamma(\sum_{i=1}^{k}x_i+1)}{\prod_{i=1}^{k}\Gamma(x_i+1)}\prod_{i=1}^{k}\theta_i^{x_i}\frac{1}{B(\alpha_i)}\prod_{i=1}^k\theta_{i}^{\alpha_i-1}\\
&=\frac{B(\alpha_i+x_i)}{B(\alpha_i)}\frac{\Gamma(\sum_{i=1}^{k}x_i+1)}{\prod_{i=1}^{k}\Gamma(x_i+1)}\frac{1}{B(\alpha_i+x_i)}\prod_{i=1}^k\theta_i^{\alpha_i+x_i-1}\\
&=h(x)Dir(\theta|\alpha_i+x_i)\tag{4}\\
h(x)&=\frac{B(\alpha_i+x_i)}{B(\alpha_i)}\frac{\Gamma(\sum_{i=1}^{k}x_i+1)}{\prod_{i=1}^{k}\Gamma(x_i+1)}
\end{align}
$$
而联合概率$P(\theta_i,x_i|\alpha_i)=P(\theta_i|x_i,\alpha_i)P(x_i|\alpha_i)$其中联合概率已经求得我们只要得到$P(x_i|\alpha_i)$即可得到后验概率
$$
\begin{align}
P(x_i|\alpha_i)&=\int_0^1P(x_i,\theta_i|\alpha_i)d\theta_i\\
&=\frac{B(\alpha_i+x_i)}{B(\alpha_i)}\frac{\Gamma(\sum_{i=1}^{k}x_i+1)}{\prod_{i=1}^{k}\Gamma(x_i+1)}\int_0^1Dir(\theta|\alpha_i+x_i)d\theta_i\\
&=\frac{B(\alpha_i+x_i)}{B(\alpha_i)}\frac{\Gamma(\sum_{i=1}^{k}x_i+1)}{\prod_{i=1}^{k}\Gamma(x_i+1)}\\
&=h(x)\tag{5}
\end{align}
$$


因此后验概率为:
$$
P(\theta_i|x_i,\alpha_i)=Dir(\alpha_i+x_i)\tag{6}
$$
所以狄利克雷分布是多项式分布的共轭先验分布。

## 参考

1. [wikipedia-Dirichlet distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution)



