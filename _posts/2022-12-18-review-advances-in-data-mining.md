---
layout: post
title: "Review: Advances in Data Mining"
categories: machine_learning
tags: machine_learning
math: true
pseudocode: true

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

**Hadoop**: Hadoop distributed file system (hdfs).
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

Jaccard formula:

$$
\begin{align*}
\text{sim}(C_1, C_2) = \frac{|C_1\cap C_2|}{|C_1\cup C_2|}
\end{align*}
$$

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
Metrics: RMSE.
- Initialize $$U$$ and $$V$$.
- Ordering the optimization of the elements of $$U$$ and $$V$$. Firstly, update each element of $$U$$. After that, update each element of $$V$$.
- Converge to a minimum.

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

Total error: $$\sum\left((\text{sim}(p_i, p_j) - \text{dist}(q_i, q_j))^2\right)$$.\
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
For fixed $$m, n$$, choose $$k$$ to minimize the false positive rate $$f$$.

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

## Similarity Search
**Three steps for similarity testing (documents)**
1. Shingling: Convert documents, emails, etc., to sets.
2. Minhashing: Convert large sets to short signatures, while preserving similarity.
3. Locality-sensitive hashing: Focus on pairs of signatures likely to be similar.

### Shingling
**Shingles**\
A $$k$$-shingle for a document is a sequence of $$k$$ consecutive characters that appear in the document.

**Example**\
$$k = 2$$, $$\text{doc} = abcab$$.\
Set of 2-shingles $$= \{ab, bc, ca\}$$.

**Choosing the shingle size**\
How large $$k$$ should be depends on how long typical documents are and how large the set of typical characters is. The important thing to remember is:
- $$k$$ should be picked large enough that the probability of any given shingle appearing in any given document is low.

**Hashing shingles**\
Represent a document by the set of hash values of its $$k$$-shingles.

### Minhashing
**Signatures**\
Hash each column $$C$$ to a small signature $$Sig(C)$$, such that:
- $$Sig(C)$$ is small enough that we can fit a signature in main memory for each column.
- $$Sim(C_1, C_2)\approx Sim(Sig(C_1), Sig(C_2))$$.

**Minhashing**\
Define the hash function $$h(C)=$$ the position of the first (in the permuted order) row in which column $$C$$ has 1. Use several independent hash functions to create a signature. Note that:

$$
\begin{align*}
Pr(h(C_1) = h(C_2)) = Sim(C_1, C_2) = \frac{B}{B + L + R}
\end{align*}
$$

The signature of length $$k$$ for a set $$C$$ is:

$$
\begin{align*}
\{h_1(C), h_2(C), \dots, h_k(C)\}
\end{align*}
$$

where $$h_1, h_2, \dots, h_k$$ are some randomly chosen permutations. The similarity of signatures is the fraction of the positions in which they agree.

**Implementation**\
Represent the document by row-column paires and sort once by row.\
Time complexity: $$O(N)$$ ($$N=$$ total size of documents).

<pre id="minhashing" class="pseudocode">
\begin{algorithm}
\caption{Minhashing}
\begin{algorithmic}
\FOR{each row $r$}
    \FOR{each column $c$}
        \IF{$c$ has 1 in row $r$}
            \FOR{each hash function $h_i$}
                \STATE $Sig(i, c) = \min(Sig(i, c), h_i(r))$
            \ENDFOR
        \ENDIF
    \ENDFOR
\ENDFOR
\end{algorithmic}
\end{algorithm}
</pre>

### Local-Sensitive Hashing
**Key ideas**
- Split columns of signature maytix $$M$$ into several blocks of the same size.
- If two columns are very similar, then it is very likely that at least at one block they will be identical. Instead of comparing blocks, send them to buckets with the help of hash functions.
- Candidate paires are those that hash at least once to the same bucket. Check if candidate paires are really similar.

**More formally**
- Divide matrix $$M$$ into $$b$$ bands of $$r$$ rows.
- For each band, hash its portion of each column to a hash table with $$k$$ buckets ($$k$$ is a big number).
- Candidate columns pairs are those that hash to the same bucket for at least one band.
- Tune $$b$$ and $$r$$ to catch most similar pairs but few non-similar pairs.

#### Analysis
**False positives**: Dissimilar pairs that do hash to the same bucket.

Suppose we use $$b$$ bands of $$r$$ rows each. And suppose that a particular pair of documents have Jaccard similarity $$s$$.
- The probability that the signatures agree on one row is $$s$$.
- The probability that they agree on all $$r$$ rows of a given band is $$s^r$$.
- The probability that they do not agree on any of the rows of a band is $$1 - s^r$$.
- The probability that for none of $$b$$ bands they agree in all rows of that band is $$(1 - s^r)^b$$.
- The probability that they will agree in all rows of at least one band is $$1 - (1 - s^r)^b$$. This function is the probability that signatures will be compared for similarity.

**Time complexity**: $$O(\text{#documents})$$.

Check if candidate pairs are really similar:
- Compare signatures.
- Compare original documents.

The number of detected pairs depends on:
- The shape of the "detection curve" (the steeper the better).
- The true distribution of similar pairs.

**Note**: If $$b\cdot r = \text{constant}$$, increasing $$b$$ will shift the detection curve towards left (the curve may be steeper or flatter). On the contrary, increasing $$r$$ will shift the detection curve towards right and make it steeper.

### Distance Measures
**Properties**
1. $$d(x, y) \geq 0$$.
2. $$d(x, y) = 0$$ if and only if $$x = y$$.
3. $$d(x, y) = d(y, x)$$.
4. $$d(x, y) \leq d(x, z) + d(z, y)$$.

**Euclidean distance**

$$
\begin{align*}
d(x, y) = \Vert x - y\Vert^2
\end{align*}
$$

**Jaccard distance**

$$
\begin{align*}
d(x, y) = 1 - Sim(x, y)
\end{align*}
$$

**Cosine distance**

$$
\begin{align*}
d(x, y) = \arccos\left(\frac{x\cdot y}{\Vert x\Vert\Vert y\Vert}\right)
\end{align*}
$$

**Hamming distance**: Hamming distance between two vectors is the number of components in which they differ. For example, the hamming distance between 10101 and 11110 is 3.

### Random Projection
**Key idea**
- Pick a random vector $$v$$, which determines a hash function $$h_v$$ with two buckets.

    $$
    \begin{align*}
    h_v(x) = 
    \begin{cases}
    1\quad & v\cdot x > 0,\\
    -1\quad & \text{Otherwise}.
    \end{cases}
    \end{align*}
    $$

- Claim:
    
    $$
    \begin{align*}
    Pr(h(x) = h(y)) = Cosine\ Sim(x, y)
    \end{align*}
    $$

**Sketches**\
Instead of chosing a random vector from all possible vectors, it turns out to be sufficiently random if we restrict our choice to vectors whose components are $$+1$$ and $$-1$$. The result $$h_v(x)$$ is called the sketch of $$x$$.

## PageRank
A random surfer model: Markov process.

### Original Version

$$
\begin{align*}
v = Mv
\end{align*}
$$

#### Convergence
**Conditions**
1. The graph is strongly connected, i.e., there is a path between any two nodes.
2. There are no dead ends (nodes with no links going out).

$$
\begin{align*}
v' &= \lim_{n\rightarrow \infty} M^n v\\
v' &= Mv'
\end{align*}
$$

$$v'$$ is the principal eigen vector of matrix $$M$$ (principal = biggest eigen value).

### Teleporting Version
Deal with dead ends and spider traps.

$$
\begin{align*}
v = \beta Mv + (1 - \beta)e / n
\end{align*}
$$

where $$e$$ is a vector of $$n$$ 1's and $$\beta$$ is a hyperparameter (usually 0.8 or 0.9).\
For graph with dead ends, $$\sum_i v_i$$ may be smaller than 1. Still, useful in practice.

An alternative algorithm for handling dead ends:
1. Recursively remove dead ends and corresponding links.
2. Calculate PageRank for nodes of the ramaining graph.
3. Propagate the values to removed nodes.

However, $$\sum_i v_i$$ may be bigger than 1.

### Implementation
Store $$M$$ and $$v_{\text{old}}$$ on the disk, while keep $$v_{\text{new}}$$ on RAM. Update $$v_{\text{new}}$$ in a single scan of $$M$$ and $$v_{\text{old}}$$.

**Sparse matrix encoding**\
Encode sparse matrix using only nonzero entries: [source node, out-degree, destination nodes].

Assume $$v_{\text{new}}$$ can be fitted into RAM.

<pre id="sparse-matrix-encoding" class="pseudocode">
\begin{algorithm}
\caption{Sparse Matrix Encoding}
\begin{algorithmic}
\STATE Initialize all entries of $v_{\text{new}} = (1 - \beta) / N$.
\FOR{each node $i$}
    \STATE Read into memory $i, d_i, \text{dest}_1, \dots, \text{dest}_k, v_{\text{old}}(i)$.
    \FOR{$j=1$ \TO $k$}
        \STATE $v_{\text{new}}(\text{dest}_j) += \beta v_{\text{old}}(i) / d_i$
    \ENDFOR
\ENDFOR
\end{algorithmic}
\end{algorithm}
</pre>

## Model Ensembles
### Bagging
Reduce the variance. Use when:
- Base learner is sensitive to small changes.
- Base learner overfits data.
- Data is scarce.

Given a training set $$T$$.
1. Sample $$T$$ with replacement to create $$n$$ versions of $$T$$.
2. For each $$T_i$$, build a model.
3. Aggregate predictions:
    - Classification: Majority voting.
    - Regression: Average.

**Unique examples**: $$1 - 1/e \approx 63.2\%$$ 

**Pros**
- No thinking, no tuning.
- Better accuracy (almost for free) / regularization.

**Cons**
- More computations (easily to run on parallel, however).
- Loss of interpretability.

### Boosting
Develop a number of models that specilize in different regions of data. The more difficult case gets more attention.

Build a sequence of classifiers $$C_1, \dots, C_k$$.
1. $$C_1$$ is trained on the original data.
2. $$C_{n + 1}$$ pays more attention to cases misclassified by $$C_1, \dots, C_n$$.

**Pros**
- Reduce exponentially fast the error on the training set.
- Do not overfit the training set.
- Most successful with primitive base classifier, e.g., decision stumps, linear regression.

**Cons**
- Models are expensive to build and difficult to interpret.

### Random Forest
For $$b=1$$ to $$B$$,
1. Draw a bootstrap sample $$Z^\ast$$ of size $$N$$ from the training data.
2. Grow a random-forest tree $$T_b$$ to the bootstrapped data, by recursively repeating the following steps for each terminal node of the tree, until the minimum node size $$n_{min}$$ is reached.
    - Select $$m$$ variables at random from the $$p$$ variables.
    - Pick the best variable / split-point among $$m$$.
    - Split the node into two child nodes.

Output: the ensemble of trees $$\{T_b\}$$.
- Regression: $$\hat{f}(x) = E[T_b(x)]$$.
- Classification: Majority voting.

**Pros**
- Superior accuracy.
- No cross-validation needed (Out-of-Bag error estimate).
- Few parameters to tune.
- Hign robust (not very sensitive).
- Trivial to parallelize.
- Provide a heuristic measure of variable importance.

#### Main Hyperparameters
1. $$m$$: The number of variables used for node split process.
2. $$n_{min}$$: The minimum node size.
3. $$B$$: The number of trees in the ensemble.

By default,
- Classification: $$m = \lfloor \sqrt{p}\rfloor$$, $$n_{min} = 1$$.
- Regression: $$m = \lfloor p / 3 \rfloor$$, $$n_{min} = 5$$.

#### Out-of-Bag
**Out-of-bag samples**\
For each observation $$z_i = (x_i, y_i)$$, construct its random forest predictor by averaging only those trees corresponding to bootstrap samples in which $$z_i$$ did not appear.

**Out-of-bag error**\
The out-of-bag (OOB) error is the average error for each $$z_i$$ calculated using predictions from the trees that do not contain $$z_i$$ in their respective bootstrap sample. 

#### Variable Importance
At each split in each tree, the improvement in the split-criterion is the importance measure attributed to the splitting variable, and is accumulated over all the trees in the forest separately for each variable.

Alternatively, we can use the OOB samples to consruct a different variable importance measure. When the $$b$$-th tree is grown, the OBB samples are passed down the tree, and the prediction accuracy is recorded. Then the values for the $$j$$-th variable are randomly permuted in the OOB samples, and the accuracy is again computed. The decrease in accuracy as a result of this permuting is averaged over all trees, and is used as a measure of the importance of variable $$j$$ in the random forest.

### XGBoost
**Notations**
- $$K$$: The number of trees.
- $$f_k$$: A function in the functional space $$\mathcal{F}$$.
- $$\mathcal{F}$$: The set of all possible CARTs.
- $$\omega(f_k)$$: The complecity of the tree $$f_k$$.
- $$w$$: The vector of scores on leaves.
- $$q$$: A function assigning each data point to the corresponding leaf.
- $$T$$: The number of leaves.

**Algorithm**
1. Build an ensemble of CARTs.

    $$
    \begin{align*}
    \hat{y}_i = \sum_{k=1}^K f_k(x_i), f_k\in \mathcal{F}
    \end{align*}
    $$

2. Optimize the objective function.

    $$
    \begin{align*}
    \text{obj}(\theta) = \sum_i^n l(y_i, \hat{y}_i) + \sum_{k = 1}^{K} \omega(f_k)
    \end{align*}
    $$

**Additive training**\
It is intractable to learn all the trees at once. Instead, we use an additive strategy: fix what we have learned, and add one new tree at a time. 

$$
\begin{align*}
\hat{y}_i^{(t)} &= \sum_{k = 1}^t f_k (x_i) = \hat{y}_i^{(t - 1)} + f_t(x_i)\\
\text{obj}^{(t)} &= \sum_{i=1}^n l(y_i, \hat{y}_i^{(t)}) + \sum_{i = 1}^t \omega(f_i)\\
&= \sum_{i = 1}^n l(y_i, \hat{y}_i^{(t - 1)} + f_t(x_i)) + \omega(f_t) + \text{constant}
\end{align*}
$$

In the general case, we take the Taylor expansion of the loss function up to the second order:

$$
\begin{align*}
\text{obj}^{(t)} = \sum_{i=1}^n \left[l(y_i, \hat{y}_i^{(t - 1)}) + g_if_t(x_i) + \frac{1}{2}h_if_t^2(x_i) \right] + \omega(f_t) + \text{constant}
\end{align*}
$$

where the $$g_i$$ and $$h_i$$ are defined as:

$$
\begin{align*}
g_i &= \partial_{\hat{y}_i^{(t-1)}} l(y_i. \hat{y}_i^{(t - 1)})\\
h_i &= \partial_{\hat{y}_i^{(t-1)}}^2 l(y_i. \hat{y}_i^{(t - 1)})
\end{align*}
$$

After we remove all the constants, the specific objective at step $$t$$ becomes:

$$
\begin{align*}
\sum_{i = 1}^n \left[g_if_t(x_i) + \frac{1}{2}h_if_t^2(x_i) \right] + \omega(f_t) 
\end{align*}
$$

This becomes our optimization goal for the new tree.

**Model complexity**\
The definition of the tree $$f(x)$$:

$$
\begin{align*}
f_t(x) = w_{q(x)}, w\in R^T, q: R^d\rightarrow \{1, 2, \dots, T\}.
\end{align*}
$$

In XGBoost, we define the complexity as:

$$
\begin{align*}
\omega(f) = \gamma T + \frac{1}{2} \lambda\sum_{j = 1}^T w_j^2
\end{align*}
$$

## Basics of Time Series Analysis
### Patterns
The first we can do to identify patterns in a time series is separate it into components with easily understandable characteristics:

$$
\begin{align*}
X_t = T_t + S_t + C_t + I_t
\end{align*}
$$

where:
- $$T_t$$: The trend shows a general direction of the time series data over a long period of time. It represents a long-term progression of the series (secular variation).
- $$S_t$$: The seasonal component with fixed and known period. It is observed when there is a distinct repeated pattern observed between regular intervals due to seasonal factors: annual, monthly or weekly. Obvious examples include daily power consumption patterns or annual sales of seasonal goods.
- $$C_t$$ (optional): Cyclical component is a repetitive pattern which does not occur at fixed intervals - usually observed in an economic context like business cycles.
- $$I_t$$: The irregular component (residuals ) consists of the fluctuations in the time series that are observed after removing trend and seasonal / cyclical variations.

**Multiplicative decomposition**

$$
\begin{align*}
X_t = T_t * S_t * I_t \Leftrightarrow \log X_t = \log T_t + \log S_t + \log I_t
\end{align*}
$$

### Prophet

$$
\begin{align*}
X_t = T_t + S_t + H_t + \epsilon_t
\end{align*}
$$

where:
- $$T_t$$: Trend component.
- $$S_t$$: Seasonal component.
- $$H_t$$: Deterministic irregular component (holidays).
- $$\epsilon$$: Noise.

## Exploratory Data Analysis
Data mining process: selecting, exploring, modeling.

### CRISP Model
Business understanding, data understanding, **data preparation**, data modeling, model evaluation and deployment.

## Anomalies
**Three scenarios**
- Supervised anomaly detection: Labels available for both normal data and anomalies. Similar to rare class mining.
- Semi-supervised anomaly detection: Labels available only for normal data.
- Unsupervised anomaly detection: No labels assumed. Based on the assumption that anomalies are very rare compared to normal data.

**Three types of anomalies**
- Point anomalies: An individual data instance is anomalous if it deviates significantly from the rest of the data set.
- Contextual anomalies: An individual data instance is anomalous within a context. Require notion of context. Also referred to as conditional anomalies.
- Collective anomalies: A collection of related data instances is anomalous. Require a relationship among data instances (sequential data, spatial data, graph data). The individual instances within a collective anomaly are not anomalous by themselves.

### Classification Based Techniques
#### Supervised Classification Techniques
**Pros**
- Models that can be easily understood.
- Many algorithms available.
- High accuracy in detecting many kinds of known anomalies.

**Cons**
- Require both labels from both normal and anomaly class.
- Cannot detect unknown and emerging anomalies.

#### Semi-supervised Classification Techniques
**Pros**
- Models that can be easily understood.
- Normal behavior can be accurately learned.
- Many techniques available.

**Cons**
- Require labels from normal class.
- Possible high false alarm rate - previously unseen (yet legitimate) data records may be recognized as anomalies.

#### Nearest Neighbor Based Techniques
**Key assumption**: Normal points have close neighbors while anomalies are located far from other points.

**General two-step approach**
1. Compute neighborhood for each data record.
2. Analyze the neighborhood to determine whether data record is anomaly or not.

**Categories**
- Distance based methods: Anomalies are data points most distant from other points.
- Density based methods: Anomalies are data points in low density regions.
    - LOF approach: For each point, compute the density of its local neighborhood. Compute local outlier factor (LOF) of a sample $$p$$ as the average of ratios of the density of sample $$p$$ and the density of its nearest neighbors. Outliers are points with largest LOF value.

**Terms**
- $$k\_NN(x)$$: The average of distances from point $$x$$ to $$k$$ closest points.
- $$kth\_NN(x)$$: The biggest distance among distances from point $$x$$ to $$k$$ closest points.
- $$\rho(x) = 1 / kth\_NN(x)$$ (approx.).
- $$LOF(x) = E[\rho(y) / \rho(x)]$$, where $$y$$ is in the $$k$$-neigborhood of $$x$$.

**Cons**
- Applicable to specific type of data: n-dimensional "points" (vectors), with a meaningful "distance" or "similarity" measure.
- Computationally very expensive: $$O(n^2)$$.
- If "normal points" do not have sufficient number of neighbors, the techniques may fail.
- In high dimensional spaces, data is sparse and the concept of similarity may not be meaningful anymore.

#### Isolation Forest
- Given a set of points in $$R^n$$, build a "random tree" with leaves containing single points. When you have to split data select a variable/dimension and a splitting point at random. Iterate till all points are assigned to leaves.
- Repeat this process many times (build many trees) and measure for each data point its "average depth" in the forest.
- The smaller the "average depth" of a point the more likely it is an outlier.

## References
1. Slides of Advances in Data Mining course, 2022 Fall, Leiden University.
2. [Principal component analysis.](https://en.wikipedia.org/wiki/Principal_component_analysis)
3. [t-distributed stochastic neighbor embedding.](https://www.wikiwand.com/en/T-distributed_stochastic_neighbor_embedding)
4. [Understanding UMAP.](https://pair-code.github.io/understanding-umap/)
5. Wasem, Din J.. *Mining of Massive Datasets*. 2014.
6. [Bloom filters – Introduction and implementation](https://www.geeksforgeeks.org/bloom-filters-introduction-and-python-implementation/)
7. [Bloom filters.](https://florian.github.io/bloom-filters/)
8. [Lecture 11: Bloom filters, final review.](https://courses.cs.washington.edu/courses/csep544/11au/lectures/lecture11-bloom-filters-final-review.pdf)
9. Hastie, Trevor, Tibshirani, Robert and Friedman, Jerome. *The Elements of Statistical Learning*. New York, NY, USA: Springer New York Inc., 2001.
10. [OOB errors for random forests.](https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html)
11. [Introduction to boosted trees.](https://xgboost.readthedocs.io/en/stable/tutorials/model.html)
