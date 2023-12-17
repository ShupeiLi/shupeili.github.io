---
layout: post
title: "Review: Evolutionary Algorithms"
categories: machine_learning
tags: machine_learning
math: true
pseudocode: true

---

## Basics
### Searching Big Search Spaces
- Search space: $$\{0, 1\}^n$$.
- $2^n$ possible candidates.
- Assume $$\mathbf{a}^*\in \{0, 1\}^n$$ is the goal vector.

> - Size (formally called the cardinality) of the binary search space: $$\vert \{0, 1\}^n \vert = 2^n$$ . It is growing exponentially.
> - The following objective function $f(\mathbf{x}) = \sum_{i = 1}^n x_i$ is called OneMax or CountingOnes, which is of fundamental importance in the theory of evolutionary algorithms.
{: .prompt-info }

#### Monte-Carlo Search

<pre id="mc-search" class="pseudocode">
\begin{algorithm}
\caption{Monte-Carlo Search}
\begin{algorithmic}
\STATE $k\leftarrow 1$
\STATE Randomly initialize a candidate solution $\mathbf{a}_k\in\{0, 1\}^n$.
\WHILE{$\mathbf{a_k} \neq \mathbf{a}^*$}
    \STATE $k \leftarrow k + 1$
    \STATE Random generate a new candidate $\mathbf{a}_k\in\{0, 1\}^n$
\ENDWHILE
\end{algorithmic}
\end{algorithm}
</pre>

**Why Monte-Carlo search is inefficient?**\
Intuition: identical strings might be tested repeatedly.

**Proof 1**\
$p_k$: the probability of generating $\mathbf{a}^*$ in the first $k$ iterations.

$$
\begin{align*}
p_1 &= P\{\mathbf{a}_1 = \mathbf{a}^*\} = \frac{1}{2^n} \\
p_k &= 1 - \left(1 - \frac{1}{2^n}\right)^k \\
k &= \frac{\ln (1 - p_k)}{\ln (1 - 2^{-n})} \\
\end{align*}
$$

Use $\ln (1 + x) \approx x$ for $x \approx 0$,

$$
\begin{align*}
k &\approx -2^n\ln (1 - p_k)
\end{align*}
$$

Monte-Carlo search performs worse than the complete enumeration:

$$
\begin{align*}
-2^n \ln (1 - p_k) &\geq 2^n\\
p_k > 1 - \frac{1}{e} &\approx 0.63
\end{align*}
$$

**Proof 2**

From the perspective of the geometric distribution:

$$
\begin{align*}
p &= P\{\mathbf{a}_1 = \mathbf{a}^*\} = \frac{1}{2^n} \\
p_k &= (1 - p)^{k - 1}p\\
E[k] &= \sum_{k = 1}^{\infty} k\cdot p_k\\
     &= \sum_{k = 1}^{\infty} k\cdot (1 - p)^{k - 1}p\\
     &= \frac{1}{p} \\
     &= 2^n
\end{align*}
$$

The expected running time is exponential on average.

$$
\begin{align*}
Pr(k > 2^n) &= 1 - \sum_{k = 1}^{2^n} p_k = \left(1 - 2^{-n}\right)^{2^n}
\end{align*}
$$

Let $c = 2^n$. Use the limit $\lim_{x\rightarrow \infty}\ln \left(1 - \frac{1}{x} \right) = -1$,

$$
Pr(k > 2^n) = \exp\left[\ln\left(1 - \frac{1}{c}\right)^c \right] \rightarrow e^{-1} \approx 0.3679
$$

#### Evolutionary Search

> Evolutionary algorithms is not Monte-Carlo search.
{: .prompt-danger }

<pre id="evolutionary search" class="pseudocode">
\begin{algorithm}
\caption{Evolutionary Search}
\begin{algorithmic}
\STATE $k \leftarrow 1$
\STATE Randomly initialize a candidate solution $\mathbf{a}_k \in\{0,1\}^n$.
\WHILE{$\mathbf{a}_k \neq \mathbf{a}^*$}
    \STATE Create a copy $\mathbf{a}_k^{\prime}$ of $\mathbf{a}_k$.
    \STATE Flip each bit in $\mathbf{a}_k^{\prime}$ with probability $p \in(0,1)$.
    \IF{$d\left(\mathbf{a}_k^{\prime}, \mathbf{a}^*\right) < d\left(\mathbf{a}_k, \mathbf{a}^*\right)$}
    \COMMENT{$d(a, b)$ computes the Hamming Distance.}
        \STATE $\mathbf{a}_{k+1} \leftarrow \mathbf{a}_k^{\prime}$
    \ELSE
    \STATE $\mathbf{a}_{k+1} \leftarrow \mathbf{a}_k$
    \ENDIF
    \STATE $k \leftarrow k+1$
\ENDWHILE
\end{algorithmic}
\end{algorithm}
</pre>

**Algorithm analysis**\
Assume exactly $m$ bits are still wrong.

$$
\begin{align*}
P\left\{\mathbf{a}_k^{\prime} \text { improves } \mathbf{a}_k \text { by } 1 \text { bit }\right\} &\geq m p(1-p)^{m-1}(1-p)^{n-m}=m p(1-p)^{n-1}\\
E_{1-\text { bit improvement }} &\leq \frac{1}{m p(1-p)^{n-1}}\\
E_{\text {iter }}(n) &= \sum_{m=1}^n E_{1-\text { bit improvement }} \leq \frac{1}{p(1-p)^{n-1}} \sum_{m=1}^n \frac{1}{m}\\
\end{align*}
$$

Assume we need $n\ 1$-bit improvements. Use $\lim_{n \rightarrow \infty}\left(\sum_{k=1}^n \frac{1}{k}-\ln n\right)=\gamma=0.522$:

$$
\begin{align*}
E_{\text {iter }}(n) &\leq \frac{\ln n}{p(1-p)^{n-1}} + \gamma\\
\end{align*}
$$

Assuming $p=\frac{q}{n}$ with an integer $q$. Use $\lim_{n \rightarrow \infty}\left(1+\frac{x}{n}\right)^n=e^x$:

$$
E_{\text {iter }}(n)<\frac{e^q}{q} n \ln n \Rightarrow E_{\text {iter }}(n) \in O(n \ln n)
$$

**Remarks**
- For this simple example: evolution-like algorithms are logarithmic, not exponential, concerning their running time.
- The analysis of Algorithm 2 is oversimplified:
    - Only one-bit mutations.
    - Only improving mutations.
    - Only an upper bound on $E(n)$.
    - We can assume to start with $n/2$ correct bits.
- Algorithm 2 is a so-called $(1+1)$-algorithm.

### Optimization
**Definition**

|Approach|Input|Model|Output|
|:---:|:---:|:---:|:---:|
|Modeling|$\checkmark$|$?$|$\checkmark$|
|Simulation|$\checkmark$|$\checkmark$|$?$|
|Optimization|$?$|$\checkmark$|$\checkmark$|

Given the objective function $f: M\rightarrow \mathbb{R}$, the optimization goal is $\min f(\mathbf{x})$.
- $f$: objective function.
    - High-dimensional.
    - Non-linear, multimodal.
    - Discontinuous, noisy, dynamic. 
- $M\subseteq M_1\times M_2\times \cdots\times M_n$ is heterogeneous.
- Restrictions possible over $M, f(x)$.
- Good global / local, robust optimum desired.

Remarks
- Evolutionary Algorithms (EAs) are mostly used for optimization.
- EAs are **global random search** algorithms.
- Global: find the global optimum (in the long run).

**Terminology**
- Classification of optimization algorithms.
    - Direct optimization algorithms: evolutionary algorithms.
    - First-order optimization algorithms: gradient methods.
    - Second-order optimization algorithms: quasi-Newton methods.
- Black-box optimization: the analytical form of the optimization problem is unknown, e.g. simulation models, real-life experiments.
- Hamming distance: $\delta_H (\mathbf{x}, \mathbf{y}) = \sum_{i} I(x_i\neq y_i) = \sum_{i} \vert x_i - y_i\vert $.

**Theoretical statements for EA**
- Global convergence with probability 1.
    
    $$
    Pr\left(\lim_{t\rightarrow \infty} \mathbf{x}_t\in X^* \right) = 1
    $$

    where $$\{\mathbf{x}_1, \mathbf{x}_2, \dots\}$$ is the sequence of all points generated by the algorithm and $X^*$ is the set of global optimizer.
    - General statement holds for all functions with a measurable size of the set of global maximizers.
    - Useless for practical situations:
        - Time plays a major role in practice.
        - Not all objective functions are relevant in practice.
- Convergence velocity:

    $$
    \varphi = E\left[f_{\text{best}}^{t+1} - f_{\text{best}}^{t} \right]
    $$

    - Typically, convex objective functions.
    - Very extensive results available for evolution strategies and genetic algorithms.
    - Not all objective functions are relevant in practice.
- An infinite number of pathological cases.
    - No Free Lunch (NFL) theorem: all optimization algorithms perform equally well iff performance is averaged over all possible optimization problems.
    - Fortunately,we are not interested in "all possible optimization problems".

### Introduction to EAs
EAs taxonomy: genetic algortims (GA), evolutionary strategies (ES), evolutionary programming (EP), and genetic programming (GP).

**GA vs ES**

|Genetic Algorithms|Evoluntionary Strategies|
|:---:|:---:|
|Discrete representations|Mixed-integer capabilities|
|Emphasis on crossover|Emphasis on mutation|
|No self-adaptation|Self-adaptation|
|Larger population sizes|Small population sizes|
|Probabilistic selection|Deterministic selection|
|Developed in US|Developed in Europe|
|Theory focused on schema processing|Theory focused on convergence speed|

**Overview of EAs**
- Main components.
    - Representation of individuals: coding.
    - Evaluation method for individuals: fitness.
    - Variation operators: mutation and crossover.
    - Selection mechanism: parent (mating) selection mechanism and survivor (environmental) selection mechanism.
- Advantages.
    - Widely applicable, also in cases where no good solution techniques are available.
        - Multimodalities, discontinuities, constraints. 
        - Noisy objective functions. 
        - Multiple criteria decision making problems. 
        - Implicitly defined problems (simulation models).
    - No presumptions with respect to search space.
    - Low development costs, i.e., costs to adapt to new problems.
    - The solutions of EAs have straightforward interpretations.
    - Can run interactively, always deliver solutions.
    - Self-adaptation of strategy parameters.
- Disadvantages.
    - No guarantee for finding optimal solutions within a finite amount of time. This is true for all global optimization methods.
    - No complete theoretical basis (yet), but much progress is being made.
    - Parameter tuning is sometimes based on trial and error. Solution: Self-adaptation of strategy parameters
- Two views.
    1. Global random search methods.
        - Probabilistic search with high "creativity".
        - Diversified search.
        - Applying local search operators.
    2. Nature-based search techniques.
        - Stochastic influence.
        - Population based.
        - Adaptive behavior.
        - Recognizing / amplifying strong gene patterns.

### Asymptotic Notations
Assume $T (n)$ and $g(n)$ are both defined on $\mathbb{N}>0$ and take value in $$\mathbb{R} >0 \cup \{\infty\}^1$$.
- **Big-O notation** ($T (n)$ is bounded above) describes the worst case of the running time of an algorithm. We say the running time of an algorithm is $T (n) \in O(g(n))$ if there $\exists n_0 > 0$ and $C > 0$ such that $\forall n > n_0 , T (n) \leq Cg(n)$. Equivalently, $T(n)\in O(g(n)) \Leftrightarrow \limsup_{n\rightarrow \infty} \frac{T(n)}{g(n)} < \infty$.
- **Big-Omega notation** ($T (n)$ is bounded below, Knuth’s definition) describes the best case of running time of an algorithm. We say $T (n) \in \Omega (g(n))$ if there exists if there $\exists n_0 \geq 0$ and $c > 0$ such that $\forall n > n_0$, $T (n) \geq cg(n)$. Equivalently, $T (n) \in \Omega (g(n)) \Leftrightarrow \liminf_{n\rightarrow\infty} \frac{T(n)}{g(n)}> 0$.
- **Big-Theta notation** ($T (n)$ is bounded above and below). We say $T (n) \in \Theta (g(n))$ if there $\exists n_0 > 0, c_1 > 0$, and $c_2 > 0$ such that $\forall n > n_0 , c_1 g(n) \leq T (n) \leq c_2 g(n)$. Equivalently, $T (n) \in \Theta(g(n))$ iif $T (n) \in O(g(n))$ and $T (n) \in \Omega (g(n))$.
- **Small-O notation** ($T (n)$ is dominated asymptotically). We say $T (n) \in o(g(n))$ if $\forall C > 0$ there $\exists n_0 > 0$ such that $\forall n > n_0$, $T (n) < Cg(n)$, meaning $T (n)$ grows way slower than $g(n)$. Equivalently, $T (n) \in o(g(n)) \Leftrightarrow \lim_{n\rightarrow \infty} \frac{T(n)}{g(n)} = 0$.
- **Small-Omega notation** ($T (n)$ is dominating asymptotically). We say $T (n) \in \omega (g(n))$ if $\forall C > 0$ there $\exists n_0 > 0$ such that $\forall n > n_0 , T (n) > Cg(n)$, meaning $\lim T (n)$ grows way faster than $g(n)$. Equivalently, $T (n) \in \omega (g(n)) \Leftrightarrow \lim_{n\rightarrow \infty} \frac{T(n)}{g(n)} = \infty$.
- **The same order** (asymptotically equal). We say $T(n) \sim g(n)$ if $\forall \varepsilon > 0$ there $\exists n_0 > 0$ such that $\forall n > n_0$, $\vert T(n)/g(n) − 1\vert < \varepsilon$. Equivalently, $T (n)\sim \lim_{n\rightarrow\infty} \frac{T(n)}{g(n)} = 1$.

## Genetic Algorithms

## Evolutionary Strategies

## References
1. Slides of Evolutionary Algorithms course, 2023 Fall, Leiden University.