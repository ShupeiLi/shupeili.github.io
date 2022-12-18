---
layout: post
title: "Review: Advances in Data Mining"
categories: machine learning
tags: machine-learning

---

## Basics
**Hash function**: A hash function maps hash-keys of some data type to integer bucket numbers. A good hash function distributes the possible hash-key values approximately evenly among buckets. Any data type can be the domain of a hash function.

**Hash table**: A data structure which can store objects from a universe $$U$$, such that the expected cost of inserting, deleting, and searching a single element takes $$O(1)$$ steps.

**Load factor**: The ratio $$K / N$$, where $$K$$ is the size of the universe, $$N$$ is the number of bins.

**Index**: An index is a data structure that allows us to store and retrieve data records efficiently, given the value in one or more of the fields of the record. Hashing is one way to build an index.

**Storage on Disk**: When data must be stored on disk (secondary memory), it takes very much more time to access a desired data item than if the same data were stored in main memory. When data is large, it is important that algorithms strive to keep needed data in main memory.

**TF.IDF**: Term frequency times inverse document frequency. Define $$f_{ij}$$ to be the frequency of term $$i$$ in document $$j$$.

$$
\begin{align*}
TF_{ij} = \frac{f_{ij}}{\max_{k}f_{kj}}
\end{align*}
$$

TF: Same document. Normalize $$f_{ij}$$. The most frequent term in document $$j$$ gets a TF of 1.\
Suppose term $$i$$ appears in $$n_i$$ of the $$N$$ documents in the collection.

$$
\begin{align*}
IDF_i = \log_2 (N / n_i)
\end{align*}
$$

IDF: Multiple documents.

$$
\begin{align*}
TF.IDF = TF_{ij} \times IDF_i
\end{align*}
$$

**Hadoop**: Haddop distributed file system (hdfs).
- Chunks, replicas, "read-only", "write / append once".
- Unlimited scalability (million of nodes).
- Very robust, can run on a cluster / grid / WAN.

**MapReduce**:
- Map: Process chunks of data.
- Shuffle and sort (implicit, pre-programmed).
- Reduce: Aggregate partial results.

**An important sequence**:

$$
\begin{align*}
\lim_{n\rightarrow \infty} \left(1 + \frac{1}{n} \right)^n &= e\\
\lim_{n\rightarrow \infty} \left(1 - \frac{1}{n} \right)^n &= \frac{1}{e}
\end{align*}
$$

## Recommender Systems
### Content-based / Memory-based Approach
- Construct for every item its profile - a vector (or set) of features that characterize the item.
- Construct for every user his / her profile - an "average" profile of the items he / she likes.
- Recommend items that are closest to the user profile.
- Closeness: Jaccard, cosine, Pearson, ...

Cosine formula:

$$
\begin{align*}
\cos(a, b) = \frac{a\cdot b}{\Vert a\Vert_2\cdot \Vert b\Vert_2}
\end{align*}
$$

Pearson formula:

$$
\begin{align*}
corr(a, b) = \frac{\sum_i(r_{ai} - \bar{r}_a)(r_{bi} - \bar{r}_b)}{\sqrt{\sum_i(r_{ai} - \bar{r}_a)^2\sum_i(r_{bi} - \bar{r}_{b})^2}}
\end{align*}
$$

**Pros**:
- Item-profiles constructed up front (without historical data).
- Natural way of item clustering.
- Intuitively simple.

**Cons**:
- Memory & cpu-time expensive $$O(N^2)$$.
- Low accuracy, e.g. RMSE on the test data.

### Model-based Approach
Once we have item profiles we can train, for every user, a classification (or regression) model which predicts rating from the item profile.
- Model: DF, NN, SVM, ...
- Input: Item profile.
- Output: User ratings.

**Cons**:
- Expensive to build and maintain.
- Low accuracy.
- Cold-start problem: New users.

### Collaborative Filtering
Recommend items to the user based on what other similar useds have liked.
- Similar users: Users that rate items in a similar way.

Find other user's preferences from data.
- Explicit method: Ratings.
- Implicit method: Observations.

User-user collaborative filtering / item-item collaborative filtering.

**Three phases**:
1. Represent data: Rating / utility matrix.
2. Define neighborhood: Cosine, Pearson correlation, Jaccard.
3. Make predictions or recommendations: Weighting scheme.

**Pros**:
- Can recommend items that are not linked to the user's earlier choices (useful for promotions).
- Consider the opinions of a wide spectrum of users.

**Cons**:
- Sparsity of data.
- Individual characteristics are not taken into account.
- Tends to recommend poplar items (convergence around few items).
- Computationally very expensive (time & memory).

### Matrix Factorization
#### UV-Decomposition
Metrics: RMSE
- Initialize $$U$$ and $$V$$.
- Ordering the optimization of the elements of $$U$$ and $$V$$. Firstly, update each element of $$U$$. After that, update each element of $$V$$.
- Converging to a minimum.

#### Gradient Descent
- Initialize all variables at random.
- Iterate over all records (user, item, rating):
    - Calculate the error: 

        $$
        \begin{align*}
        \text{error} = \text{rating} - u_{\text{user}} \cdot v_{\text{item}}
        \end{align*}
        $$
   - Update parameters $$u_{\text{user}}$$ and $$v_{\text{item}}$$:

        $$
        \begin{align*}
        u_{\text{user}} &= u_{\text{user}} + \alpha\cdot \text{error}\cdot v_{\text{item}}\\
        v_{\text{item}} &= v_{\text{item}} + \alpha\cdot \text{error}\cdot u_{\text{user}}
        \end{align*}
        $$

#### Gradient Descent with Regularization
Prevent overfitting.
- Initialize all variables at random.
- Iterate over all records (user, item, rating):
    - Calculate the error: 

        $$
        \begin{align*}
        \text{error} = \text{rating} - u_{\text{user}} \cdot v_{\text{item}}
        \end{align*}
        $$
   - Update parameters $$u_{\text{user}}$$ and $$v_{\text{item}}$$:

        $$
        \begin{align*}
        u_{\text{user}} &= u_{\text{user}} + \alpha (\text{error}\cdot v_{\text{item}} - \lambda \cdot u_{\text{user}})\\
        v_{\text{item}} &= v_{\text{item}} + \alpha (\text{error}\cdot u_{\text{user}} - \lambda \cdot v_{\text{item}})
        \end{align*}
        $$

## Reduction of Dimensionality
### Principal Component Analysis (PCA)
PCA is defined as an orthogonal linear transformation that transforms the data to a new coordinate system such that the greatest variance by some scalar projection of the data comes to lie on the first coordinate (called the first principal component), the second greatest variance on the second coordinate, and so on.

Given $$n$$ variables $$x_1, x_2, \dots, x_n$$, principal components are linear combinations of $$x_1, x_2, \dots, x_n$$ such that:
- They are orthogonal to each other.
- They maximize the "variance of projections".
- The first principal component explains most of the variance, second principal component less.
- In practice, the first few principal components (2 - 3) explain most of the variance.

### Multi-Dimensional Scaling
Given $$n$$ objects $$p_1,\dots, p_n$$ and a similarity (distance) measure between them. Find a mapping $$p_i\rightarrow q_i$$ ($$q_i$$ in a low dimensional space) that preserves the original distances as much as possible.

$$
\begin{align*}
\text{sim}(p_i, p_j) \approx \text{dist} (q_i, q_j),\ \forall i, j.
\end{align*}
$$

Total error: $$\sum\left((\text{sim}(p_i, p_j) - \text{dist}(q_i, q_j))^2\right)$$\
An optimization problem: Random initialization and SGD optimization.

### Locally Linear Embeddings
For each object $$X_i$$, find a few neighboring objects. Measure distances between $$X_i$$ and these neighbors. Find $$Y_i$$ in low dimensional space that preserve all mutual distances: A very simple optimization problem.

Key idea: Mapping high dimensional inputs $$X_i$$ to low dimensional outputs $$Y_i$$ via local linear reconstruction weights $$W_{ij}$$.

### t-Distributed Stochastic Neighbor Embedding (t-SNE)
The t-SNE algorithm comprises two main stages. First, t-SNE constructs a probability distribution over pairs of high-dimensional objects in such a way that similar objects are assigned a higher probability while dissimilar points are assigned a lower probability. Second, t-SNE defines a similar probability distribution over the points in the low-dimensional map, and it minimizes the Kullback–Leibler divergence (KL divergence) between the two distributions with respect to the locations of the points in the map. While the original algorithm uses the Euclidean distance between objects as the base of its similarity metric, this can be changed as appropriate.

### Uniform Manifold Approximation and Projection (UMAP)
UMAP, at its core, works very similarly to t-SNE - both use graph layout algorithms to arrange data in low-dimensional space. In the simplest sense, UMAP constructs a high dimensional graph representation of the data then optimizes a low-dimensional graph to be as structurally similar as possible. While the mathematics UMAP uses to construct the high-dimensional graph is advanced, the intuition behind them is remarkably simple.

In order to construct the initial high-dimensional graph, UMAP builds something called a "fuzzy simplicial complex". This is really just a representation of a weighted graph, with edge weights representing the likelihood that two points are connected. To determine connectedness, UMAP extends a radius outwards from each point, connecting points when those radii overlap. Choosing this radius is critical - too small a choice will lead to small, isolated clusters, while too large a choice will connect everything together. UMAP overcomes this challenge by choosing a radius locally, based on the distance to each point's $$n$$-th nearest neighbor. UMAP then makes the graph "fuzzy" by decreasing the likelihood of connection as the radius grows. Finally, by stipulating that each point must be connected to at least its closest neighbor, UMAP ensures that local structure is preserved in balance with global structure.

Once the high-dimensional graph is constructed, UMAP optimizes the layout of a low-dimensional analogue to be as similar as possible. This process is essentially the same as in t-SNE, but using a few clever tricks to speed up the process.

## Mining Data Streams
### Sampling a Data Stream
Suppose a bank would like to have a representative sample of $$3\%$$ of all their data.

#### A Good Approach
Use a hash function to map each client to an integer $$1, \dots, 100$$. Transactions with number $$1, 2, 3$$ enter the sample. Ignore others.

**Pros**
- No need to create and keep any hash table.
- New clients are also included in the sampling process.

**Cons**
- The sample size may exceed available RAM.

#### Sapmling a Stream in Limited RAM
- Instead of a pre-specified percentage of clients that we want to cover (e.g. $$3\%$$), specify the amount of RAM that can be used for storing the sample.
- Use a hash function with, say, $$L=10,000$$ buckets. And put all the records from buckets $$1, 2, \dots, L$$ to the sample. $$L$$ is initially set to $$10,000$$.
- Whenever the size of samples reaches the size of available RAM, remove all the records from bucket $$L$$ and set $$L$$ to $$L - 1$$.

### Reservoir Sampling
**Goal**\
A random, uniformly distributed sample of $$s$$ records. Every record from the stream has the same chance of being kept in RAM, i.e.,

$$
\begin{align*}
Pr = \frac{s}{\text{#records seen so far}}
\end{align*}
$$

**Initialization**\
Store the first $$s$$ records of the stream in RAM. At this moment $$n=s$$, the probability of an element entering RAM is $$s/n$$ (Accidentally, it's 1).

**Inductive Step**
- When the $$(n + 1)^{\text{th}}$$ element arrives, decide with probability $$s / (n + 1)$$ to keep the record in RAM. Otherwise, ignore it.
- If you choose to keep it, throw one of the previously stored record out, selected with equal probability, and use the freed space for the new record.

**Proof**\
Every element has chance of $$s / n$$ of being in buffer at moment $$n$$. When a new element arrives,

$$
\begin{align*}
\left(1 - \frac{s}{n + 1}\right)\left(\frac{s}{n}\right) + \left(\frac{s}{n + 1}\right)\left(\frac{s - 1}{s}\right)\left(\frac{s}{n}\right)
\end{align*}
$$

### Bloom Filter
A Bloom filter consists of:
1. An array of $$n$$ bits, initially all 0's.
2. A collection of hash functions $$h_1, h_2, \dots, h_k$$. Each hash function maps "key" values to $$n$$ buckets, corresponding to the $$n$$ bits of the bit-array.
3. A set $$S$$ of $$m$$ key values.

The purpose of the Bloom filter is to allow through all stream elements whose keys are in $$S$$, while rejecting most of the stream elements whose keys are not in $$S$$.
- To initialize the bit array, begin with all bits 0. Take each key value in $$S$$ and hash it using each of the $$k$$ hash functions. Set to 1 each bit that is $$h_i(K)$$ for some hash function $$h_i$$ and some key value $$K$$ in $$S$$.
- To test a key $$K$$ that arrives in the stream, check that all of

    $$
    \begin{align*}
    h_1(K), h_2(K), \dots, h_k(K)
    \end{align*}
    $$

    are 1's in the bit-array. If all are 1's, then let the stream element through. If one or more of these bits are 0, then $$K$$ could not be in $$S$$, so reject the stream element.

#### False Positive

**Concept**\
The element passes the filter even if its key value is not in $$S$$.

**Notations**
- $$n$$: The bit-array length.
- $$m$$: The number of members of $$S$$.
- $$k$$: The number of hash functions.

**Analysis**\
The model to use is throwing darts at targets. Suppose we have $$x$$ targets and $$y$$ darts. Any dart is equally likely to hit any target. After throwing the darts, how many targets can we expect to be hit at least once?
- The probability that a given dart will not hit a given target is $$(x - 1) / x$$.
- The probability that none of the $$y$$ darts will hit a given target is $$\left(\frac{x - 1}{x}\right)^y$$. Rewrite it as $$\left(1 - \frac{1}{x}\right)^{x\left(\frac{y}{x}\right)}$$.
- Use the approximation $$\left(1 - \epsilon\right)^{1 / \epsilon} = 1 / e$$ for small $$\epsilon$$.
    
    $$
    \begin{align*}
    \left(1 - \frac{1}{x}\right)^{x\left(\frac{y}{x}\right)} \rightarrow e^{-\frac{y}{x}}
    \end{align*}
    $$

Apply the model on the Bloom filter. Think of each bit as a target, and each member of $$S$$ as a dart. Then, the number of targets is $$x = n$$, and the number of darts is $$y = km$$. Thus, the probability that a bit remains 0 is $$e^{-km/n}$$. In general, the probability of a false positive is the probability of a 1 bit, which is $$1 - e^{-km/n}$$, raised to the $$k$$-th power, i.e., $$\left(1 - e^{-km/n}\right)^k$$.

$$
\begin{align*}
\left(1 - \left(1 - \frac{1}{n}\right)^{km}\right)^k = \left(1 - e^{-\frac{km}{n}}\right)^k
\end{align*}
$$

**Note**: Bloom filters never generate false negative result, e.g., telling you that a username doesn’t exist when it actually exists.

**Choosing $$k$$**\
Two competing forces:
- If $$k$$ is large,
    - More bits in array are 1 $$\rightarrow$$ Higher false positive rate.
    - Test more bits for key value $$\rightarrow$$ Lower false positive rate.
- If $$k$$ is small,
    - More bits in array are 0 $$\rightarrow$$ Lower false positive rate.
    - Test fewer bits for key value $$\rightarrow$$ Higher false positive rate.

**Optimizing $$k$$**\
For fixed $$m, n$$, choose $$k$$ to minimize the false positive rate$$f$$.

$$
\begin{align*}
g &= \ln(f) = k\ln\left(1 - e^{-km/n}\right)\\
\frac{\partial g}{\partial k} &= \ln\left(1 - e^{-\frac{km}{n}}\right) + \frac{km}{n}\frac{e^{-\frac{km}{n}}}{1 - e^{-\frac{km}{n}}}\\
k &= \frac{n}{m}\ln 2
\end{align*}
$$

#### Remarks
- Cascading: Using two or more filters one after another reduces errors exponentially fast.
- Inserting new elements to a filter is easy (never fail). However, the false positive rate increases steadily as elements are added until all bits in the filter are set to 1, at which point all queries yield a positive result.
- Removal of elements from a filter is almost impossible. Because if we delete a single element by clearing bits at indices generated by $$k$$ hash functions, it might cause deletion of few other elements.
- If two Bloom filters $$F_A$$ and $$F_B$$ represent sets $$A$$ and $$B$$, then the bitwise AND of $$F_A$$ and $$F_B$$ represents the intersection of $$A$$ and $$B$$. It turns out that this is not a perfect solution because it will lead to more false positives compared to directly querying the individual Bloom filters. The bitwise OR of $$F_A$$ and $$F_B$$ represents the union of $$A$$ and $$B$$. Note that OR operation resulting Bloom filter behaves exactly as when we directly query all original Bloom filters and only returning true if at least one individual Bloom filters returned true.

#### Applications
- Google BigTable and Apache Cassandra: Reduce the disk lookups for non-existant rows or columns.
- Google Chrome: Identify malicious URLs.
- Squid Web Proxy Cache: Cache digests.
- Bitcoin: Speed up wallet synchronization.
- Venti archival storage system: Detect previously stored data.

### Counting Distinct Elements
A data stream consists of elements chosen from a set of size $$n$$, where $$n$$ is very big number. How to maintain the count of the number of distinct elements seen so far?

#### Obvious Approach
Maintain the set of elements seen ($$O(n)$$ memory complexity).

#### MinTopK Estimate
- Hash incoming objects into doubles from the interval $$[0, 1]$$ and count them shrinking the interval if needed.
- Due to limited memory, maintain only the $$K$$ biggest values ("TopK").
- Let $$s$$ denote the minimum of our set. The number of distinct elements $$\approx K / (1 - s)$$.

#### Flajolet-Martin Algorithm
**Key idea**\
The more different elements we see in th stream, the more different hash-values we shall see. As we see more different hash-values, it becomes more likely that one of these values will be "unusual". The particular property we shall exploit is that the value ends in many 0's, although many other options exist.

**Algorithm**
- Apply a hash function $$h$$ to a stream element $$a$$, the bit string $$h(a)$$ will end in some number of 0's, possibly none.
- Denote this number the tail length for $$a$$ and $$h$$. Let $$R$$ be the maximum tail length of any $$a$$ seen so far in the stream.
- Use estimate $$2^R$$ for the number of distinct elements seen in the stream.

**Combine Estimates**
- Median of all estimates.
- Divide the hash functions into small groups and take their average. Then take the median of the averages.

## References
1. Slides of Advances in Data Mining course, 2022 Fall, Leiden University.
2. [Principal component analysis.](https://en.wikipedia.org/wiki/Principal_component_analysis)
3. [t-distributed stochastic neighbor embedding.](t-distributed stochastic neighbor embedding)
4. [Understanding UMAP.](https://pair-code.github.io/understanding-umap/)
5. Wasem, Din J.. *Mining of Massive Datasets*. 2014.
6. [Bloom filters – Introduction and implementation](https://www.geeksforgeeks.org/bloom-filters-introduction-and-python-implementation/)
7. [Bloom filters.](https://florian.github.io/bloom-filters/)
8. [Lecture 11: Bloom filters, final review.](https://courses.cs.washington.edu/courses/csep544/11au/lectures/lecture11-bloom-filters-final-review.pdf)
