# 多项式分布(Multinomial distribution)

## 简单概念

多项式分布是二项分布的一个推广，将二项分布公式中的状态推广至多种即为多项式分布。举个形象的例子来说二项分布是抛硬币，多项式分布就类似于掷骰子。

## 多项式分布的概率质量函数

$$
f(x_1,x_2,...x_k;n,p_1,p_2,...p_k)=\begin{cases}
\frac{n!}{x_1!x_2!...x_n!}p_1^{x_1}p_2^{x_2}...p_k^{x_k},\quad &when\sum_{i=1}^kx_i=n\\
0\quad&otherwise
\end{cases}
$$

可用gamma函数改写成如下形式
$$
f(x_1,x_2,...x_k;n,p_1,p_2,...p_k)=\frac{\Gamma(\sum_{i=1}^{k}x_i+1)}{\prod_{i=1}^{k}\Gamma(x_i+1)}\prod_{i=1}^{k}p_i^{x_i}
$$


## 期望和方差

### 期望

$$
E[x_i]=np_i
$$

### 方差

$$
Var(x_i)=np_i(1-p_i)
$$



## 参考

1. [wikipedia-Multinomial distribution](https://en.wikipedia.org/wiki/Multinomial_distribution)