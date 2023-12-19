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
### Overview
<pre id="ga" class="pseudocode">
\begin{algorithm}
\caption{Genetic Algorithm}
\begin{algorithmic}
\STATE $t \leftarrow$ 0
\STATE \texttt{Initialization}$(P(t))$
\STATE \texttt{Evaluation}$(P(t))$
\WHILE{Termination criterion not met}
    \STATE $P(t)_{\text{temp}} \leftarrow$ \texttt{MatingSelection}$(P(t))$
    \STATE $P(t)_{\text{temp}} \leftarrow$ \texttt{Crossover}$(P(t)_{\text{temp}}, p_c)$
    \STATE $P(t)_{\text{temp}} \leftarrow$ \texttt{Mutation}$(P(t)_{\text{temp}}, p_m)$
    \STATE \texttt{Evaluation}$(P(t))$
    \STATE $P(t + 1)\leftarrow P(t)_{\text{temp}}$
    \STATE $t \leftarrow t + 1$
\ENDWHILE
\end{algorithmic}
\end{algorithm}
</pre>

**Main features**
- Often binary search space $$\{0, 1\}^l$$.
- Emphasis on crossover (recombination).
- No environmental selection. Probabilistic mating selection (proportional etc.).
- Constant strategy parameters.

### Representation
#### Genotype / Phenotype Space
![Genotype space vs Phenotype space.](/assets/img/courses/ga-review-1.png){: .normal w="700"}

Remarks
- Canonical GAs: individuals $$\mathbf{a}\in\{0, 1\}^l$$.
- Genotype space: $$\{0, 1\}^l$$.
- Problem-specific representation (phenotype space) mapped to $$\{0, 1\}^l$$ decoding and encoding functions.
- Variation operators act on genotypes. Selection acts on phenotype evaluations.
- Phenotypes are evaluated. Genotypes are decoded into phenotypes.

#### Mapping Real Numbers
Solve a problem defined over real values:
1. Subdivide $\mathbf{a}$ into $n$ segments of equal length.
2. Decode each segment into integer value.
3. Map each integer linearly into a given real-valued interval.

Key points
- Phenotype space: $M = \prod_{i = 1}^n[u_i, v_i]\subseteq \mathbb{R}^n$ with interval bounds $u_i < v_i$
- Genotype space: $$\{0, 1\}^l$$ with $l = n\cdot l_x$.
- Decoding function: $$h^{\prime}(\mathbf{a})=\left[h_1\left(a_1, \ldots, a_{l_x}\right), h_2\left(a_{1+l_x}, \ldots, a_{2 l_x}\right), \ldots, h_n\left(a_{1+(n-1) l_x}, \ldots, a_l\right)\right]$$ 
    with (like little-endian)

    $$
    \begin{align*}
    a_1^{\prime}, a_2^{\prime}, \ldots, a_{l_x}^{\prime} & =a_{1+(i-1) l_x}, a_{2+(i-1) l_x}, \ldots, a_{l+(i-1) l_x} \\
    h_i\left(a_1^{\prime}, \ldots, a_{l_x}^{\prime}\right) & =u_i+\frac{v_i-u_i}{2^{l_x}-1}\left(\sum_{k=0}^{l_x - 1} a_{l_x - k}\cdot 2^{k}\right)
    \end{align*}
    $$

    Example:

    Assume $u_i = -50, v_i = 50, l_x = 9$ bits used to represent $$\{0, 1, \dots, 511\}$$.
    
    ![Mapping real numbers](/assets/img/courses/ga-review-2.png){: .normal w="700"}

    - $100110100 \rightarrow 2^0 + 2^3 + 2^4 + 2^6 = 89 \rightarrow -50 + \frac{100}{511}\times 89 \approx -32.5831$.
    - $111011110 \rightarrow 2^0 + 2^1 + 2^2 + 2^4 + 2^5 + 2^6 + 2^7 = 247 \rightarrow -50 + \frac{100}{511}\times  \approx -1.6634$.

#### Binary Code
- Convert a decimal number to binary is to divide the number by 2 and round down to the nearest integer.
- Convert a binary number to decimal is to multiply the number by 2 and add the result, i.e., $\sum_{i=0}^{l - 1} d_i \cdot 2^i$. For example, $$1010_2\rightarrow 2^3 + 2^1 = 10_{10}$$.

#### Gray Code
- Gray code is a binary code in which two successive codewords differ in only one bit.
- Assume the binary code of a string is $(b_{n - 1}\dots b_0)$ and its gray code is $g_n\dots g_1$. We have the following relationship:

    $$
    \begin{align*}
    g_i &= b_i \oplus b_{i - 1}\\
    b_{i - 1} &= b_i \oplus g_i
    \end{align*}
    $$

    Example: bitstring 1011.
    - Binary decoding: $2^0 + 2^1 + 2^3 = 11$.
    - Gray decoding:

        $$
        \begin{align*}
        & 1\quad  0\quad  1\quad  1\\
        & 0\quad  1\quad  0\quad  1\\
        & 0\quad  0\quad  1\quad  0\\
        \oplus\quad & 0\quad  0\quad  0\quad  1\\
        \hline
        &1\quad  1\quad  0\quad  1
        \end{align*}
        $$

        $2^3 + 2^2 + 2^0 = 13$
    - If the binary code is 1011, the corresponding gray code is 1110 ($14_{10}$).

### Selection
#### Proportional Selection
Implementation: roulette wheel technique.
- Assign to each individual a part of the roulette wheel (size proportional to its fitness).
- Spin the wheel $\mu$ times to select $\mu$ individuals.

**Naive version**

$$
p_i = \frac{f_i}{\sum_{j = 1}^\mu f_j}
$$

Disadvantages
- Functions $f$ and $f + c$ with constant $c$ handled differently
- If all function values in a population are similar $\Rightarrow$ random selection.
- Require positive-values, maximization.
- Need the population size $\mu$.

**Scaling version**\
Let $c = f_{\min}$,

$$
\begin{align*}
f' &= f - c\\
p'_i &= \frac{f_i - c}{\sum_{j = 1}^\mu f_j - c\cdot \mu}
\end{align*}
$$

- Scaling increases selection probabilities of above-average individuals.
- Decrease selection probabilities of below-average individuals.
- Improved version: shift the values by slightly more than the smallest value to have all probabilities above 0, e.g., $f' = f - c + 0.1$.

#### Rank Selection
In rank selection, the selection probability does not depend directly on the fitness, but on the fitness rank of an individual within the population. 

- This puts large fitness differences into perspective.
- The exact fitness values themselves do not have to be available, but only a sorting of the individuals according to quality.

Example: a rank-based function,

$$
\phi(\mathbf{a}) = \frac{2r_i}{\mu(\mu + 1)}
$$

where $r_i$ is the rank of the $i$-th individual's fitness value $f_i$ in the population.

#### Tournament Selection
Assume the tournament size is $k$.

**Version 1**
1. Select $k$ individuals from the population and perform a tournament amongst them.
2. Select the best individual from the $k$ individuals.
3. Repeat process 1 and 2 until you have the desired amount of population.

**Version 2**
1. Select $k$ individuals from the population at random.
2. Choose the $i$-th best individual with probability $p\cdot(1-p)^i$.
3. Repeat process 1 and 2 until you have the desired amount of population.

### Crossover
#### One-point Crossover
1. Choose a random point on the two parents.
2. Split parents at this crossover point.
3. Create children by exchanging tails

#### $n$-point Crossover
1. Choose $n$ random crossover points.
2. Split along those points.
3. Glue parts, alternating between parents.

#### Uniform Crossover
1. For each $$i\in\{1, \dots, l\}$$, flip a coin.
2. If "head", copy bit from Parent 1 to Offspring 1, Parent 2 to Offspring 2.
3. If "tail", copy bit from Parent 1 to Offspring 2, Parent 2 to Offspring 1.

#### Crossover Parameters
- Application probability $p_c\in[0, 1]$. Chooses two individuals with probability $p_c$ for crossover.
- Number of crossover points $z$.

#### Partially Mapped Crossover (PMX)
For traveling salesman problem (TSP) like problems.
1. Selection of two crossover points. 
2. Copy the middle segment. 
3. Determine mapping relationship to legalize offspring. 
4. Legalize offspring with the mapping relationship.

An example from [Baeldung post](https://www.baeldung.com/cs/ga-pmx-operator):

Step 1. 

$$
\begin{align*}
\text{Parent 1} &= (1, 2, \underline{3, 4, 5, 6}, 7, 8, 9)\\ 
\text{Parent 2} &= (5, 4, \underline{6, 9, 2, 1}, 7, 8, 3)
\end{align*}
$$

Step 2.

$$
\begin{align*}
\text{Offspring 1} &= (1, 2, \underline{6, 9, 2, 1}, 7, 8, 9)\\
\text{Offspring 2} &= (5, 4, \underline{3, 4, 5, 6}, 7, 8, 3)
\end{align*}
$$

Step 3.

|Offspring 1| | | |Offspring 2|
|:---:|:---:|:---:|:---:|:---:|
|1|$\leftrightarrow$|6|$\leftrightarrow$|3|
|2||$\leftrightarrow$||5|
|9||$\leftrightarrow$||4|

Step 4.

$$
\begin{align*}
\text{Offspring 1} &= (1, 5, \underline{6, 9, 2, 1}, 7, 8, 4)\\
\text{Offspring 2} &= (2, 9, \underline{3, 4, 5, 6}, 7, 8, 1)
\end{align*}
$$

#### Order Crossover (OX1)
For order-based permutation tasks.
1. Randomly select gene segments in $P_0$.
2. As a child permutation, a permutation is generated that contains the selected gene segments of $P_0$ in the same position.
3. The remaining missing genes are now also transferred, but in the order in which they appear in $P_1$.
4. This results in the completed child genome.

An example from [Wikipedia](https://www.wikiwand.com/en/Crossover_(genetic_algorithm)#Order_crossover_(OX1)):

Step 1.

$$
\begin{align*}
\text{Parent 1} &= (\underline{1, 2}, 3, 4, 5, \underline{6, 7, 8}, 9,10)\\ 
\text{Parent 2} &= (2, 4, 1, 8, 10, 3, 5, 7, 6, 9)
\end{align*}
$$

Step 2.

$$
\text{Offspring} = (\underline{1, 2}, ?, ?, ?, \underline{6, 7, 8}, ?, ?)
$$

Step 3.

$$
\begin{align*}
P_\text{missing} &= (3, 4, 5, 9,10)\\ 
P_\text{in order from Parent 2} &= (4, 10, 3, 5, 9)
\end{align*}
$$

Step 4.

$$
\text{Offspring} = (\underline{1, 2}, 4, 10, 3, \underline{6, 7, 8}, 5, 9)
$$

### Mutation
Bit-flip mutation
- Alter each gene with probability $p_m$.
- The standard choice is $p_m = 1 / l$.
- Only values of $1 / l \leq p_m \leq 1 / 2$ make sense.
    - At least one bit on average should mutate.
    - $p = 1/2$ corresponds with random generation of offspring.

Crossover or mutation?
- Cooperation and competition between exploration and exploitation.
    - Exploration: discovering promising areas in the search space, i.e., gaining information on the problem.
    - Exploitation: optimizing within a promising area, i.e., using information.
- If we define "distance in the search space" as Hamming distance:
    - Crossover is explorative, it makes a big jump to an area somewhere "in between" two (parent) areas.
    - Mutation is exploitative, it creates random small deviations, thereby staying near (i.e., in the area of) the parent.
    - To hit the optimum you often need a "lucky" mutation (or multiple mutations).

Mutation and small deviations
- Assumptions
    1. Binary representations and decoding function $h(a_1, \dots, a_l) = \sum_{i = 0}^{l - 1} a_{i + 1}\cdot 2^i$ are used.
    2. Mutation with $p_m = 1/l$ is applied.
    3. Objective function $f = h$.
- Results.

    $$
    \begin{align*}
    P(\vert \Delta f\vert = 2^i) &= \frac{1}{l},\quad \forall i\in \{0, \dots, l - 1\} \\
    \Delta f = f(\mathbf{a}) &- f(\mathbf{a}') = h(\mathbf{a}) - h(\mathbf{a}')
    \end{align*}
    $$

    - $\mathbf{a}$: parent individual.
    - $\mathbf{a}'$: result of mutating $\mathbf{a}$.

### Schema Theory
#### Schema
**Definition**\
A schema $H\in\mathbb{B}^l$ is a partial instantiation of a string. Usually the uninstantiated elements are denoted by "$*$", sometimes called "don't care" symbol or "wild card". A schema defines a subset of $\mathbb{B}^l: H\in \{0, 1, *\}$.

- Set of all instances of schema $H = (h_1, \dots, h_l)$:

    $$
    I(H) = \{(a_1, \dots, a_l)\in\mathbb{B}^l|h_i\neq \ast \Rightarrow a_i = h_i\}
    $$

- Order of the schema: number of instantiated elements.

    $$
    o(H) = \vert\{i|h_i\in\{0, 1\} \}\vert
    $$

- Length of the schema: length of the substring starting at the first and ending at the last instantiated element.

    $$
    d(H) = \max\{i|h_i\in\{0, 1\}\} - \min\{i|h_i\in\{0, 1\}\}
    $$
- Facts.
    - In total there are $3^l$ different schemata.
    - Each chromosome (element of $\mathbb{B}^l$) is an instance of $2^l$ different schemata (examine the total number of schemata of which this element is an instance).

        $$
        \begin{align*}
        \binom{l}{0} + \binom{l}{1} + \binom{l}{2} + \cdots + \binom{l}{l} &= \sum_{i = 0}^l \binom{l}{i}\\
        &= \sum_{i = 0}^l \binom{l}{i} \cdot 1^i\\
        &= 2^l
        \end{align*}
        $$

    - There are at most $\mu\cdot 2^l$ schemata represented in a population of size $\mu$.
- A schema can be viewed as a hyperplane of an $l$-dimensional space.

#### Schema Theorem

|Notation|Meaning|
|:---:|:---:|
|$f$|to be maximized|
|$\bar{f}$|mean fitness in population|
|$l$|length of the string|
|$H$|a schema|
|$d(H)$|defining length|
|$o(H)$|order of the schema|
|$p_m$|mutation rate|
|$p_c$|crossover rate|
|$f(H)$|(estimated) schema fitness|
|$m(H, t)$|expected number of instantiations of $H$ in generation $t$|

$$
\begin{equation}
m(H, t + 1) \geq m(H, t)\cdot \frac{f(H)}{\bar{f}}\cdot \left(1 - p_c\frac{d(H)}{l - 1} \right)\cdot (1 - p_m)^{o(H)}
\tag{1}\label{eq:st}
\end{equation}
$$

- Expected number of instantiations of $H$ selected for crossover:

    $$
    m(H, t) \cdot \frac{f(H)}{\bar{f}}
    $$

- Probability that crossover does not occur within the defining length:

    $$
    1-p_c \frac{d(H)}{l-1}
    $$

- Probability that the schema is not mutated:

    $$
    \left(1-p_m\right)^{o(H)}
    $$

- The schema function inequality uses "$\geq$" because schema instances can appear from the crossover and the mutation of other patterns.

Exponential growth of building blocks
- Assumptions. 
    - $H$ is a short, low-order, highly fit schema.
    - $f(H) > \bar{f}$ assumed: $f(H) = \bar{f} + c$.
    - Assume $H$ is not destroyed by crossover or mutation.
    - Assume this remains valid for a number of generations.
- Equation \eqref{eq:st} can be simplified to:

    $$
    m(H, t + 1) = m(H, t)\cdot \frac{\bar{f} + c}{\bar{f}} = m(H, t)\cdot (1 + c')
    $$
- For a number of generations:

    $$
    m(H, t) = m(H, 0)\cdot (1 + c')^t
    $$

- Exponential growth of $H$ in the population.

Criticism
- Most of Holland’s approximations are only true for very large numbers (trials and population size).
- Within finite populations, exponentially increasing the number of schema instances leads to entirely filling the population.
- Within finite populations, exponentially decreasing the number of schema instances leads to complete elimination.
- Not all schemata are represented in a typical population.
- Schemata of large defining length are likely to be destroyed by crossover (even highly fit ones).

Almost sure covergence
- Prerequisites.
    1. $\max_{\mathbf{x}\in P(t)} f(\mathbf{x}) \geq \max_{\mathbf{x}\in P(t - 1)} f(\mathbf{x})$, e.g., through elitist selection.
    2. Any point is accessible from any other point (i.e., with mutation $p_m > 0$).
- Theorem.

    $$
    \lim_{t\rightarrow \infty} Pr\{\mathbf{x}^*\in P(t)\} = 1
    $$

    - $P(t)$: population at time $t$.
    - $\mathbf{x}^*$: global optimum.

#### Implicit Parallelism
- A lot of different schemata are effectively processed in parallel by a Genetic Algorithm. Individuals are instantiations of more than one schema.
- Effectively processing of a schema: reproduced at the desirable exponentially increasing rate.
- Why wouldn't a schema be processed effectively?
    - Reason: schema disruption by genetic operators.
    - Holland's estimate: $O(\mu^3)$ schemata are processed effectively when using a population of size $\mu$.

#### The Building Block Hypothesis (BBH)
- GAs are able to detect short, low order, and highly fit schemata and combine these into highly fit individuals.
- Building blocks are schemata that have:
    - A small defining length, $d(H)$.
    - Low order, $o(H)$.
    - High estimated fitness, $f(H)$.
- Implicit parallelism and the BBH were seen as explanations
for the power of GAs.

### Convergence Velocity
#### Definition
- Success probability:

    $$
    p_{\vec{a}}^+ = P\{f(m(\vec{a})) > f(\vec{a}) \}
    $$

- $k$-step success probability ($0\leq k \leq k_{\max}$):

    $$
    p_{\vec{a}}^+ = P\{f(m(\vec{a})) = f(\vec{a}) + k \}
    $$

- Convergence velocity:

    $$
    \begin{align*}
    \varphi &= E\left[f_{\max}(P(t + 1)) - f_{\max}(P(t)) \right] \\
    \Rightarrow \varphi_{1 + 1}(l, \vec{a}, p) &= \sum_{k = 0}^{k_{\max}} k\cdot p_{\vec{a}}^+ (k)
    \end{align*}
    $$

    |Notation|Meaning|
    |:---:|:---:|
    |$l$|bit-string length|
    |$\vec{a}$|current parent|
    |$p$|mutation rate|
    |$k$|improvement, $k\geq 0$ because of $(1+1)$|
    |$p_{\vec{a}}^+$|probability to get $k$ improvement|

#### Counting Ones
- Objective function:

    $$
    f(\vec{a}) = \sum_{i = 1}^{l} a_i
    $$

- Assume $(1 + 1)$-GA with mutation rate $p$. Let $q = 1 - p$ and $f_a:=f(\vec{a})$:

    $$
    \begin{align*}
    p_{\vec{a}}^0 &= \sum_{i = 0}^{\min\{f_a, l - f_a \}} \binom{f_a}{i}\binom{l - f_a}{i} p^{2i}q^{l - 2i}\\
    p_{\vec{a}}^+(k) &= \sum_{i = 0}^{\min\{f_a, l - f_a - k \}} \binom{f_a}{i}\binom{l - f_a}{i + k} p^{2i + k}q^{l - 2i - k},\quad 0\leq k \leq l - f_a \\
    p_{\vec{a}}^-(k) &= \sum_{i = 0}^{\min\{f_a - k, l - f_a \}} \binom{f_a}{i + k}\binom{l - f_a}{i} p^{2i + k}q^{l - 2i - k},\quad 0\leq k\leq f_a\\
    \varphi_{(1+1)} &= \sum_{k = 0}^{l - f_a} k\cdot p_{\vec{a}}^+(k)
    \end{align*}
    $$

    |Notation|Meaning|
    |:---:|:---:|
    |$f_a$|$f(\vec{a})$, the fitness value, the number of 1s|
    |$l$|the length of the bitstring|
    |$l - f_a$|the number of 0s|
    |$i$|the number of $1\rightarrow 0$ / $0\rightarrow 1$|
    |$k$|$k$-step improvements (decline) $0 \rightarrow 1$ ($1 \rightarrow 0$)|


- Approximated optimum mutation rate as a function of $f_a$:

    $$
    \begin{align*}
    p^* \approx \frac{1}{2(f_a + 1) - l}
    \end{align*}
    $$

- Absorption time.
    - Define $1 + 1$ states: $$z_k = \{\vec{a}: f(\vec{a}) =l - k \}\quad (0\leq k\leq l)$$, which represent solutions with $k$ zeros.
    - Transition probabilities

        $$
        \begin{align*}
        p_{i j}&=P\{f(m(\vec{a}))=l-j \mid f(\vec{a})=l-i\}: \\
        p_{i j}&=\left\{\begin{array}{lll}
        p_{l-i}^{+}(i-j) & , i>j & \text { Improvement } \\
        1-\sum_{k=1}^j p_{l-i}^{+}(k) & , i=j & \text { Stagnation } \\
        0 & , i<j & \text { Worsening }
        \end{array}\right.
        \end{align*}
        $$

- State 0 ( $l$ bits are correct) is absorbing.
- $\tau=\{1, \ldots, l\}$:gathered transient class of state.
- The transition matrix:

    $$
    P = \begin{pmatrix}I & 0\\ R & Q \end{pmatrix}
    $$

    in block form, according to $$E_i = \sum_{j\in T} n_{ij}$$ where $$N = (n_{ij}) = (I - Q)^{-1}$$.
    
    - $T$: transient states.
    - $I: the unity matrix.
    - $Q$: an $(l - 1) \times (l - 1)$ matrix.
    - $E_i(t)$: expected time to absorption if started in state $i$.

- Time to absorption: $O(l\cdot \ln l)$.
- Effect of different mutation rate settings:
    - $p_m$ is too large: exponential complexity (evolution $\rightarrow$ random search).
    - $p_m$ is too small: time to absorption almost constant.

#### $(1, \lambda)$-GA / $(1 + \lambda)$-GA
Convergence velocity:

$$
\begin{align*}
\varphi_{(1,/+ \lambda)} &=\sum_{k=k_{\min }}^{l-f_a} k \cdot P_{k^{\prime}=k}(\lambda)\\
&= \sum_{k=k_{\min }}^{l-f_a} k \cdot \sum_{i = 1}^{\lambda}\binom{\lambda}{i} p_{k'=k}^i\cdot p_{k'< k}^{\lambda - i}
\end{align*}
$$

- For $(1 + \lambda)$ and $(1, \lambda)$-GA, we have a single parent $\vec{a}$ and generate $\lambda$ offspring $$O = \{\vec{o}_1, \dots, \vec{o}_{\lambda} \}$$.
- "$,$": the next generation parent is chosen as the best among offspring, i.e., $\vec{a}_{t + 1} = \text{best}(O_t)$.
- "$+$": the next generation parent is chosen as the best among offspring and the old parent, i.e., $$\vec{a}_{t + 1} = \text{best}(O_t\cup \{a_t\})$$.
- $(1, \lambda)$-GA: $k_{\min} = -f_a$, $(1 + \lambda)$-GA: $k_{\min} = 0$.
- $P_{k'=k}(\lambda)$: the probability of at least one of  $\lambda$ offsprings to realize a $k$-step improvement, while the improvement of others is less than $k$. Notice that $(1,/+\lambda)$-GA will only keep the best offspring. Following notations in [$(1 + 1)$-GA (Counting Ones)](#counting-ones), we can express probabilities $$p_{k'=k}, p_{k'>k}, p_{k'< k}$$:

    $$
    \begin{align*}
    p_{k^{\prime}=k} &= \begin{cases}p_a^{+}(k) & , k \geq 0 \\
    p_a^{-}(-k) & , k<0\end{cases} \\
    p_{k^{\prime}>k} &= \begin{cases}\sum_{i=k+1}^{l-f_a} p_a^{+}(i) &  , k \geq 0 \\
    \sum_{i=k+1}^{-1} p_a^{-}(-i)+\sum_{i=0}^{l-f_a} p_a^{+}(i) & , k<0\end{cases} \\
    p_{k^{\prime}<k} &= 1-p_{k^{\prime}=k}-p_{k^{\prime}>k}
    \end{align*}
    $$

#### $(\mu, \lambda)$-GA / $(\mu + \lambda)$-GA
Convergence velocity:

$$
\begin{align*}
\varphi_{(\mu,/+\lambda)} &= \frac{1}{\mu}\sum_{k = k_{\min}}^{l - f_a} k\cdot \sum_{\nu = \lambda - \mu + 1}^{\lambda} p_{\nu} (k)\\
&= \frac{1}{\mu}\sum_{k = k_{\min}}^{l - f_a} k\cdot \sum_{\nu = \lambda - \mu + 1}^{\lambda} \sum_{i = 0}^{\nu - 1}\binom{\lambda}{\nu - i - 1}\sum_{j = 0}^{\lambda - \nu}\binom{\lambda - (\nu - 1 - i)}{\lambda - \nu - j} p_{k'=k}^{i + j + 1}\cdot p_{k' < k}^{\nu - i - 1}\cdot p_{k' > k}^{\lambda - \nu - j}
\end{align*}
$$

- $\mu$ parents and $\lambda$ offspring.
- $p_\nu(k)$: the probability of the offspring of rank $\nu$ to improve the objective function value by $k$.
- The rank is the position that the individual is situated when we order all solutions by fitness value (allow ties):
    - Individuals: $$\vec{a}_1, \vec{a}_2, \dots, \vec{a}_i, \dots, \vec{a}_{v}, \dots, \vec{a}_j, \dots, \vec{a}_{\lambda}$$.
    - Fitness values: $$f(\vec{a}_1) \leq f(\vec{a}_2) \leq \dots \leq f(\vec{a}_i) = \dots = f(\vec{a}_\nu) = \dots = f(\vec{a}_j) \leq \dots \leq f(\vec{a}_\lambda) $$.
- $\nu - i - 1$ realizations smaller than $k$ ($i = 0, 1, \dots, \nu - 1$).
- $\lambda - \nu - j$ realizations larger than $k$ ($j = 0, 1, \dots, \lambda - \nu)$.
- $i + j + 1$ realizations equals $k$.

## Evolutionary Strategies
### Overview

## References
1. Slides of Evolutionary Algorithms course, 2023 Fall, Leiden University.
2. [Selection (genetic algorithm).](https://www.wikiwand.com/en/Selection_(genetic_algorithm))
3. Geof H. Givens and Jennifer A. Hoeting. *Computational Statistics*. John Wiley & Sons, Ltd, 2012.
4. [Tournament Selection (GA).](https://www.geeksforgeeks.org/tournament-selection-ga/)
5. [Tournament selection.](https://www.wikiwand.com/en/Tournament_selection)
6. [Crossover (genetic algorithm).](https://www.wikiwand.com/en/Crossover_(genetic_algorithm))
7. [Partially Mapped Crossover in Genetic Algorithms.](https://www.baeldung.com/cs/ga-pmx-operator)
8. [Gray code.](https://encyclopediaofmath.org/wiki/Gray_code)