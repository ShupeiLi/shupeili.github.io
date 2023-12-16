---
layout: post
title: "广度优先遍历树的时间复杂度分析"
categories: algorithms
tags: algorithms
math: true

---

## 问题背景

<img src="/assets/img/algorithms/post-2023-08-21.png" alt="Question" />

## 推导
答案为 $$\Theta(N)$$。

时间复杂度为：

$$
\begin{align*}
&\sum_{k = 0}^{\log N - 1} 2^k \left(\log N - k \right)\\
=& \sum_{k = 0}^{\log N - 1} 2^k\cdot \log N - \sum_{k = 0}^{\log N - 1} k\cdot 2^k\\
=& \log N\cdot \frac{1 - 2^{\log N}}{1 - 2} - \sum_{k = 0}^{\log N - 1} k\cdot 2^k\\
=& \left(N - 1\right)\log N - \sum_{k = 0}^{\log N - 1} k\cdot 2^k
\end{align*}
$$

回忆关于求 $$\sum_{k = 1}^n k\cdot x^k$$ 的部分和：

$$
\begin{align*}
\sum_{k = 1}^n k\cdot x^k &= x\cdot \frac{d}{dx}\left[\frac{x\left(x^n - 1\right)}{x - 1} \right]\\
&= x\cdot \frac{d}{dx}\left(\frac{x^{n + 1}}{x - 1} - \frac{x}{x - 1} \right)\\
&= \frac{x^{n + 1}\left(nx - n - 1 \right) + x}{\left(x - 1 \right)^2}
\end{align*}
$$

代入得：

$$
\begin{align*}
\sum_{k = 0}^{\log N - 1} k\cdot 2^k &= \sum_{k = 1}^{\log N - 1} k\cdot 2^k\\
&= 2^{\log N}\left(2\left(\log N - 1 \right) - \left(\log N - 1 \right) - 1 \right) + 2\\
&= N\log N - 2N + 2
\end{align*}
$$

时间复杂度可化简为：

$$
\begin{align*}
&\sum_{k = 0}^{\log N - 1} 2^k \left(\log N - k \right)\\
=& N\log N - \log N - \left(N\log N - 2N + 2 \right)\\
=& 2N - \log N - 2\\
\sim & \Theta(N)
\end{align*}
$$

课程视频里提供了另一种思路。观察到，树的第一层被考虑了 $$1$$ 次，前两层被考虑了 $$1 + 2 = 3$$ 次，前三层被考虑了 $$1 + 2 + 4 = 7$$ 次，前四层被考虑了 $$1 + 2 + 4 + 8 = 15$$ 次。因此规律是前 $$k$$ 层被考虑了 $$2^k - 1$$ 次。由此得到：

$$
\begin{align*}
& 2^1 + 2^2 + \cdots + 2^{\log N} - H\\
=& 2\cdot (N - 1) - H\\
\sim & \Theta\left(N\right)
\end{align*}
$$

## References
1. Slides of CS 61B Data Structures, 2018 Spring, UCB.

