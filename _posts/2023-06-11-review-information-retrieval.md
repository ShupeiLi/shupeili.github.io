---
layout: post
title: "Review: Information Retrieval"
categories: machine_learning
tags: machine_learning
math: true

---

## Introduction
### Search VS Recommendation

|Search engines|Recommender systems|
|:---:|:---:|
|User has an information need|User has an interest|
|User types a query|User goes to a service / app / website|
|System returns item sorted by relevance to that query|System provides items that are relevant to the user|
|Based on match to the query|Based on user history/profile|
|Based on popularity|Based on popularity|

### Two-stage Retrieval
Supervised learning with BERT: Not scalable to retrieval and ranking from large collections.

**Common strategy**: Use the embedding-based model to re-rank the top documents retrieved by a lexical IR model.
1. Lexical retrieval and ranking with BM25 or LM.
2. Re-ranking with a supervised BERT model.

|First stage|Second stage|
|:---:|:---:|
|From large (huge) collection|Ranking top-n documents from 1st stage|
|Unsupervised|Supervised|
|Often term-based (sparse)|Often based on embeddings (dense)|
|Priority: recall|Priority: precision at high ranks|

### Index Time VS Query Time

|Index time|Query time|
|:---:|:---:|
|Collect (new) documents|Process the user query|
|Pre-process the documents|Match the query to the index|
|Create the documents representation|Retrieve the documents that are potentially relevant|
|Store the documents in the index|Rank the documents by relevance score|
|Indexing can take time|Retrieval cannot take time (< 1s)|

### Retrieval
**Lexical matching**: Exact match, lexical models.\
**Semantic matching**: Vocabulary mismatch, embeddings.

## Evaluation
### Set Metrics
- Precision@k.

    $$
    \begin{align*}
    \text{precision@k} = \frac{\text{# retrieved & relevant documents in top-k}}{\text{# retrieved documents in top-k}}
    \end{align*}
    $$

- Recall@k.

    $$
    \begin{align*}
    \text{recall@k} = \frac{\text{# retrieved & relevant documents in top-k}}{\text{# relevant documents}}
    \end{align*}
    $$

- F-measure.
    - A tradeoff between precision and recall.
        
        $$
        \begin{align*}
        F_{\beta} = \frac{(1 + \beta^2) P \cdot R}{\beta^2 P + R}
        \end{align*}
        $$

    - $$\beta = 1$$: Harmonic mean of precision and recall (most used).

        $$
        \begin{align*}
        F_1 = 2\cdot \frac{P \cdot R}{P + R}
        \end{align*}
        $$

### Ranking Metrics
- Mean reciprocal rank (MRR).
    - Reciprocal rank:
        
        $$
        \begin{align*}
        𝑅𝑅 = \frac{1}{\text{rank of highest ranked relevant item}}
        \end{align*}
        $$

    - Mean reciprocal rank (MRR) = average over a set of queries.
- Average precision (AP).

    $$
    \begin{align*}
    AP = \frac{\sum_{k = 1}^n (P(k)\cdot rel(k))}{\text{# of relevant items in the collection}}
    \end{align*}
    $$

    - $$n$$: The number of results in the ranked list that we consider. 
    - $$P(k)$$: Precision at position $$k$$. 
    - $$rel(k)$$: An indicator function equaling 1 if the item at rank $$k$$ is a relevant document, zero otherwise.
    - Mean average precision (MAP) = the mean over all queries.
 
### Multi-level Judgements
**Assumptions**
1. Highly relevant results contribute more than slightly relevant results. $$\Rightarrow$$ Cumulative gain (CG).
2. The gain of a document should degrade with its rank. $$\Rightarrow$$ Discounted cumulative gain (DCG).
3. Better would be to have a 0-1 scale so that scores for queries can be compared. $$\Rightarrow$$ Normalized discounted cumulative gain (nDCG).

- CG.
    
    $$
    \begin{align*}
    CG(L) = \sum_{i = 1}^n r_i
    \end{align*}
    $$

- DCG.

    $$
    \begin{align*}
    DCG(L) = r_1 + \sum_{i = 2}^n \frac{r_i}{\log_2 i}
    \end{align*}
    $$

- iDCG: DCG for the ideally ranked list.
- nDCG.

    $$
    \begin{align*}
    nDCG(L) = \frac{DCG(L)}{iDCG}
    \end{align*}
    $$

### MS MARCO
- The relevance assessments are sparse (shallow): many queries, but on average, only one relevant judgment per query.
- Consequences:
    1. Model training requires both positive as well as negative examples.\
    Solution: sample random negatives. But these negatives are not necessarily irrelevant; they are just not labelled as relevant.
    2. Difficult to see differences between models with only one explicit relevance label for each query.

## Boolean Model and Index Construction
### Boolean Retrieval
- Index construction (offline).
    - Data structures: Term-document incidence matrix, inverted index / inverted file (record the non-zero positions, postings).
    - Indexer steps:
        1. Token sequence.
        2. Sort.
        3. Dictionary & postings.
- Query evaluation (online).
    - Merging algorithm: Postings sorted by docID. $$O(n+m)$$ time complexity.
    - Query processing optimization.
        - (k AND m) AND n: Cost (k + m) + (min(k,m) + n).
        - (k OR m) AND n: Cost (k + m) + n.
        - Process in order of increasing frequency.
- Pros and cons of Boolean IR.
    - Pros: 
        - Precise semantics, easy to understand why documents are in result list. 
        - Simpler index structure. 
        - Computationally efficient query evaluation. 
    - Con: 
        - No ranked results.
        - No notion of partial matching.
        - Information need has to be translated into a Boolean expression which many users find awkward.
- Phrase queries and positional indexes.
    - In the postings, store for each term the position(s) in which tokens of it appear:\
        <term, number of docs containing term;\
        doc1: position1, position2 … ;\
        doc2: position1, position2 … ;\
        etc.>
    - This significantly increases the index size.
- Zipf’s law and power law term distributions.
    - The k-th most frequent term has frequency proportional to $$1/k$$.

    $$
    \begin{align*}
    cf_k = \frac{c}{k} = \frac{cf_1}{k} = cf_1\cdot k^{-1}
    \end{align*}
    $$

    - $$cf_k$$: Collection frequency, the number of occurrences of the term $$t_k$$ in the collection.
    - $$c$$ is the normalizing constant.
    - This is a power law (power = -1).

### Compression
- Why compression? (Search engines)
    - Keep more stuff in memory (increases speed).
    - Increase data transfer from disk to memory.
    - Premise: Decompression algorithms are fast.
- The Huffman compression algorithm.
- Entropy: The information value, the minimum number of bits per symbol for lossless encoding. $$H(P)$$ is the entropy function for a probability distribution $$P(X)$$.

    $$
    \begin{align*}
    H(P) = -\sum_{x\in X}P(x)\log_2 P(x)
    \end{align*}
    $$

- Postings compression.
    - Delta encoding: Encoding differences between document numbers (d-gaps).
    - Variable length encoding: For a gap value G, compute its value in bits, determine the fewest bytes needed to hold $$log_2 G$$ bits.
        1. Begin with one byte to store G and dedicate 1 bit in it to be a continuation bit c.
        2. If G $$\leq$$ 127, binary-encode it in the 7 available bits and set c = 1.\
           Else encode G’s lower-order 7 bits and then use additional bytes to encode the higher order bits using the same algorithm.
        3. At the end set the continuation bit of the last byte to 1 (c = 1) and of the other bytes to 0 (c = 0).

## Vector Space Model
**RSV**: Retrieval status value.
- For each query $$Q_i$$ and document $$D_j$$, compute a score $$RSV(Q_i,D_j)$$.
- At least ordinal scale (should support $$>$$, $$<$$, $$=$$).
- Value should increase with relevance relation between the query $$Q_i$$ and document $$D_j$$.
- Rank documents by RSV.

### Set-based Approach
**Jaccard similarity**

$$
\begin{align*}
\text{Jaccard}(A, B) = \frac{|A \cap B|}{|A \cup B|}
\end{align*}
$$

Issues with Jaccard for scoring:
1. We need a more sophisticated way of **normalizing for length**.
2. Jaccard doesn’t consider **term frequency**.
3. (Not too) **rare** terms in a collection are more informative than (too) **frequent** terms. Jaccard doesn’t consider this information.

**Dice similarity**

$$
\begin{align*}
\text{Dice}(A, B) = 2\cdot \frac{|A \cap B|}{|A| + |B|} = \frac{2J}{J + 1}
\end{align*}
$$

### Term-weighting Approach
Bag of words model.
- Term frequency (tf), internal weight: The term frequency $$tf_{t,d}$$ of term $$t$$ in document $$d$$ is defined as the number of times that $$t$$ occurs in $$d$$.
    - Log-frequency weighting: The log frequency weight of term $$t$$ in $$d$$ is,
        
        $$
        \begin{align*}
        w_{t, d} = \begin{cases}
        1 + \log_{10}(tf_{t, d})\quad &\text{if } tf_{t, d} > 0,\\
        0\quad &\text{otherwise}.
        \end{cases}
        \end{align*}
        $$

- Inverse document frequency (idf), external weight: Term rarity. $$idf_t$$ quantifies how surprised we are to see term $$t$$ in a document.
    
    $$
    \begin{align*}
    idf_t = \log_{10} (N / df_t)
    \end{align*}
    $$

    - idf has no effect on ranking one term queries.
- tf.idf weighting.

    $$
    \begin{align*}
    tf.idf(t, d) = (1 + \log_{10} tf_{t, d}) \times \log_{10} (N / df_t)
    \end{align*}
    $$

    - Increases with the number of occurrences within document / the rarity of the term in the collection.
    - $$tf.idf(t, d)$$ is the evidence of $$d$$ being relevant when looking for $$t$$.
- Document ranking: Documents and queries are vectors of term weights.
    - High-dimensional and sparse.
    - Proximity = similarity of vectors $$\approx$$ inverse of distance.
    - Euclidean distance is large for "similar" vectors of different lengths.
    - Cosine similarity for length-normalized vectors.
        
        $$
        \begin{align*}
        \cos(q, d) = q\cdot d = \sum_{i = 1}^{|V|} q_id_i
        \end{align*}
        $$

    - Length normalization: Dividing a vector by its $$L_2$$ norm.

        $$
        \begin{align*}
        \Vert x\Vert_2 = \sqrt{\sum_i x_i^2}
        \end{align*}
        $$

    - Cosine generalized for any $$d$$ and $$q$$.

        $$
        \begin{align*}
        \cos(q, d) = \frac{q\cdot d}{\Vert q\Vert\cdot \Vert d\Vert} = \frac{\sum_{i = 1}^{|V|}q_id_i}{\sqrt{\sum_{i = 1}^{|V|}q_i^2}\sqrt{\sum_{i = 1}^{|V|}d_i^2}}
        \end{align*}
        $$

- Hypotheses for long documents.
    - Scope hypothesis: Cover more material than others.
    - Verbosity hypothesis: Covers a similar scope to a short document, but simply uses more words.

## Neural IR
**Word embeddings**: Dense representations of words that have semantic interpretation.
- Continuous, dense vector space.
- Learned from unlabelled data: Masked language modelling with self-supervision.

|One-hot / Term vector spaces|Embeddings|
|:---:|:---:|
|Sparse|Dense|
|High-dimensional ($$\vert V\vert$$)|Lower-dimensional (still 100s)|
|Observable|Latent|
|Can be used for exact matching|Can be used for in-exact matching|

**Distributional hypothesis**: Terms that occur in similar contexts tend to be semantically similar.

**BERT**: Used in a transfer learning setting.
- Pre-training: Learning embeddings from huge unlabeled data (self-supervised).
- Fine-tuning: Learning the classification model from smaller labeled data (supervised) for any NLP task (e.g. sentiment, named entities, classification).
- The embeddings are dynamic: Being updated during fine-tuning (as opposed to earlier, static embeddings, like word2vec).
- Input.
    - The first token of the input sequence is a special token called [CLS]. Used for classification.
    - Single-input tasks (e.g. sentence classification): End of input is [SEP].
    - Two-input tasks (e.g. sentence similarity): Texts are concatenated, separated by [SEP]. Another [SEP] token appended to the end.

### Training Approaches
**Pointwise**: Learning a ranking from individual (q, d, rel) pairs.
- Regression loss.

    $$
    \begin{align*}
    \mathcal{L}_{squared} &= \Vert rel_q(d) - score(q, d)\Vert^2\\
    \mathcal{L}_{total} &= \frac{1}{N}\sum_{i = 1}^N\mathcal{L}_{squared}
    \end{align*}
    $$

- Loss function does not consider relative ranking between items in the same list, only absolute numbers.

**Pairwise**: Consider pairs of relevant and nonrelevant documents for the same query. Minimize the number of incorrect inversions in the ranking.
- Pairwise hinge loss.

    $$
    \begin{align*}
    \mathcal{L}_{hinge} &= \max (0, 1 - (score(q, d_i) - score(q, d_j)))\\
    \mathcal{L}_{total} &= \sum_{y(d_i) > y(d_j)} \mathcal{L}_{hinge}
    \end{align*}
    $$

### MonoBERT
- Two-input classification, [[CLS], q, [SEP], $$d_i$$, [SEP]]:
    - Fine-tuning: Learn the relevance label.
    - Inference: Predict the relevance label.
- Cross-encoder: The general style of concatenating query and candidate texts into a transformer.
- Output (the representation of the [CLS] token).
    - Used as input to a single-layer, fully connected neural network.
    - To obtain a probability $$s_i$$ that the candidate $$d_i$$ is relevant to $$q$$.
    - Followed by a softmax function for the relevance classification.
- The input has to be truncated if tokencount $$q + d_i < 512$$: Computational complexity (memory) / Ranking and reranking has to be fast (< 1 second).
- Pointwise learning-to-rank method: The loss function takes into account only one candidate text at a time.

**Ranking long documents**
- The length limitation of BERT (and transformers in general) makes it necessary to truncate input texts. Or split documents in passages. $$\Rightarrow$$ From passages to documents.
- Challenges.
    - Training: Unclear what to feed to the model.
    - Inference: After getting a score for each passage we need to aggregate to a document score.
- Solutions.
    - BERT-MaxP: Passage score aggregation.
        - Training: Treat all passages from a relevant document as relevant and all passages from a non-relevant document as not relevant.
        - Inference: Estimate the relevance of each passage, then take the maximum passage score (MaxP) as the document score.
    - PARADE: Passage representation aggregation.
        - Training: The same.
        - Inference: Aggregate the representations of passages rather than aggregating the scores of individual passages (averaging the [CLS] representation from each passage).
    - Alternatives for aggregation:
        - Passage-level relevance labels.
        - Transformer architectures for long texts (e.g. Longformer, BigBird).
        - Use of lexical models.

### Dense Retrieval
- Why? If we use exact (lexical matching) in the 1st stage we might miss relevant documents.
- Dense retrieval is neural first-stage retrieval (replaces lexical retriever). Using embeddings in the first stage retrieval. Potentially solves the vocabulary mismatch problem by not requiring exact matches in the first stage (e.g. bike/bicycle).
    - Use a neural query encoder & nearest neighbor vector index.
    - Can be used as part of a larger pipeline.
- Bi-encoder: Encode the query and document independently. Then compute relevance.
    - Metric: Estimate the mutual relevance. $$\eta$$ = encoder. $$\phi$$ = similarity function (e.g. dot product, cosine similarity), lightweight. $$\eta_q$$, $$\eta_d$$, $$\phi$$ are learned in a supervised manner.

        $$
        \begin{align*}
        P(Relevance = 1|d_1, q) = \phi(\eta_q(q), \eta_d(d_i))
        \end{align*}
        $$
    - Advantages.
        1. Fast at query time because only the query has to be encoded.
        2. Possibility to do end-to-end training. Removes mismatch between training and inference on a set pre-retrieved by BM25.

    - Commonly used bi-encoder model: Sentence-BERT, originally designed for sentence similarity.
        - Learn a softmax classifier based on the concatenation of $$u$$ and $$v$$ and their difference $$\vert u - v\vert$$.

            $$
            \begin{align*}
            o = softmax(W_t\cdot (u \oplus v \oplus |u - v|))
            \end{align*}
            $$

        - Loss function: Standard cross entropy loss between predicted relevance and true relevance labels.
        - At inference: $$\phi(u, v) = \cos(u, v)$$.
        - Pointwise. Because we only take one $$d$$ into account per learning item. At inference, we measure the similarity between $$q$$ and each $$d$$, then sort the $$d$$s by this similarity.
    - Why are bi-encoders less effective compared to cross-encoders?
        - Because cross-encoders can learn relevance signals from attention between the query and candidate texts (terms) at each transformer encoder layer.

**Cross-encoder VS Bi-encoder**

|Cross-encoder|Bi-encoder|
|:---:|:---:|
|One encoder for q and d (concatenation)|Separate encoders for q and d|
|Full interaction (attention) between words in q and d|No interaction between words in q and d|
|High quality ranker|Lower quality ranker (less effective)|
|But only possible in re-ranking (limited doc set)|But highly efficient (also in 1st stage retrieval)|

### ColBERT
- A model that has the effectiveness of cross-encoders and the efficiency of bi-encoders.
- Based on the idea of "late interaction". Encode the queries and the documents separately. Then use the MaxSim operator to compute the similarity between $$d$$ and $$q$$.
    - MaxSim: The maximally similar token from the document for each query term. $$\eta(t)_i$$ is the $$i$$-th token of the text $$t$$ (query or document).
        
        $$
        \begin{align*}
        s_{q, d} = \sum_{i\in \eta(q)} \max_{j\in \eta(d)} \eta(q)_i\cdot \eta(d)_j
        \end{align*}
        $$

    - Index time: Inverted index.
    - Query time: Two stage retrieval.
- Compatible with nearest-neighbor search techniques.

### Challenges of Neural IR Models
- Long documents.
    - **Memory burden** of reading the whole document in the encoder.
    - **Mixture** of many topics, query matches may be spread.
    - Neural model must **aggregate** the relevant matches from different parts.
- Short documents.
    - **Fewer** query matches.
    - But neural model is more **robust** towards the **vocabulary mismatch** problem than term-based matching models.

**The long tail problem**
- Learn good representations of text. But rare terms and rare queries.
- A good IR method:
    - Be able to retrieve infrequently searched-for documents.
    - Perform reasonably well on queries with rare terms.
    - Incorporate both lexical and semantic matching signals.

## Probabilistic Information Retrieval
### Binary Independence Model (BIM)
**Binary**: Boolean, documents are represented as binary incidence vectors of terms.\
**Independence**: Terms occur in documents independently.

**Model design**
- Interested only in ranking. Linked dependence assumption.
- Binary term incidence vectors $$q$$. $$x$$ is the binary term incidence vector representing $$d$$. Use odds and Bayes' Rule:

    $$
    \begin{align*}
    O(R|q, x) &= \frac{p(R = 1|q, x)}{p(R = 0|q, x)}\\
              &= \frac{p(R = 1|q)}{p(R = 0|q)}\cdot\frac{p(x|R = 1, q)}{p(x|R = 0, q)}\\
              &= O(R|q)\cdot \prod_{t = 1}^n\frac{p(x_t|R = 1, q)}{p(x_t|R = 0, q)}
    \end{align*}
    $$
    
    Simplication: Let $$p_t = p(x_t = 1\vert R = 1, q)$$, $$u_t = p(x_t = 1\vert R = 0, q)$$. Assume, for all doc terms not occurring in the query ($$q_t = 0$$): $$p_t = u_t$$. Isolate the constant.

    $$
    \begin{align*}
    O(R|q, x) &= O(R|q)\cdot \prod_{t, x_t = 1}\frac{p(x_t = 1|R = 1, q)}{p(x_t = 1|R = 0, q)}\cdot \prod_{t, x_t = 0} \frac{p(x_t = 0|R = 1, q)}{p(x_t = 0|R = 0, q)}\\
              &= O(R|q)\cdot\prod_{t, x_t = 1, q_t = 1} \frac{p_t}{u_t}\cdot \prod_{t, x_t = 0, q_t = 1}\frac{1 - p_t}{1 - u_t}\\
              &= O(R|q)\cdot\prod_{t, x_t = 1, q_t = 1} \frac{p_t}{u_t}\cdot\prod_{t, x_t = 1, q_t = 1}\left(\frac{1 - u_t}{1 - p_t}\cdot \frac{1 - p_t}{1 - u_t} \right) \cdot\prod_{t, x_t = 0, q_t = 1}\frac{1 - p_t}{1 - u_t}\\
              &= O(R|q)\cdot\prod_{t, x_t = q_t = 1} \frac{p_t(1 - u_t)}{u_t(1 - p_t)}\prod_{q_t = 1}\frac{1 - p_t}{1 - u_t}\\
              &\propto \prod_{t, x_t = q_t = 1}\frac{p_t (1 - u_t)}{u_t(1 - p_t)}
    \end{align*}
    $$

- Retrieval status value: Take log.

    $$
    \begin{align*}
    RSV &= \log\prod_{t, x_t = q_t = 1}\frac{p_t(1 - u_t)}{u_t(1 - p_t)}\\
        &= \sum_{t, x_t = q_t = 1} \log\frac{p_t(1 - u_t)}{u_t(1 - p_t)}
    \end{align*}
    $$

    Let $$c_t = \log\frac{p_t(1 - u_t)}{u_t(1 - p_t)}$$, 

    $$
    \begin{align*}
    RSV = \sum_{t, x_t = q_t = 1} c_t
    \end{align*}
    $$

    - $$c_t$$: Log odds ratios. They function as the term weights in this model.
- Parameter estimation: For each term $$t$$ look at the table of document counts.

    |Documents|Relevant|Non-relevant|Total|
    |:---:|:---:|:---:|:---:|
    |$$x_t = 1$$|$$s$$|$$n - s$$|n|
    |$$x_t = 0$$|$$S - s$$|$$N - n - S + s$$|N - n|
    |**Total**|S|N - S|N|

    $$
    \begin{align*}
    \hat{p}_t &= \frac{s}{S}\\
    \hat{u}_t &= \frac{n - s}{N - S}\\
    \hat{c}_t &= \log\frac{\frac{s}{S - s}}{\frac{n - s}{N - n - S + s}}
    \end{align*}
    $$

- Smoothing: Avoid situations where $$p_t = 0$$ or $$u_t = 0$$.
    - Laplace smoothing: Add a small quantity (0.5) to the counts in each cell.

        $$
        \begin{align*}
        p_t &= \frac{s + 1 / 2}{S + 1}\\
        u_t &= \frac{n - s + 1 / 2}{N - S + 1}
        \end{align*}
        $$

    - Use a constant $$p_t = 0.5$$ resulting in idf weighting of terms. In this case RSV can be simplified to:

        $$
        \begin{align*}
        RSV = \sum_{t, x_t = q_t = 1} \log \frac{N}{n_t}
        \end{align*}
        $$

### Probabilistic Relevance Feedback
- Improve the estimate of $$p_t$$.
    1. Initial ranking using $$p_t=0.5$$.
    2. Take top $$\vert V\vert$$ documents from the ranked result list.
    3. User assesses for relevance: Resulting in two sets --- VR and VNR (Relevant / Non relevant).
    4. Re-estimate $$p_t$$ and $$u_t$$ on the basis of these

        $$
        \begin{align*}
        p_t &= |VR_t + 1 / 2| / |VR + 1|\\
        u_t &= |VR_t + 1 / 2| / |VR + 1|
        \end{align*}
        $$
- Pseudo relevance feedback: Assume that $$VR = V$$ (all top docs are relevant).

### BM25
- Goal: Be sensitive to term frequency and document length while not adding too many parameters.
- Generative model for documents: Distribution of term frequencies across documents (tf) follows a binomial distribution --- approximated by a Poisson distribution.
- Poisson distribution.
    - Occurrences are independent, do not occur simultaneously.
    - Rate is independent of any occurrence

    $$
    \begin{align*}
    p(k) = \frac{\lambda^k}{k!}e^{-\lambda}
    \end{align*}
    $$

- One Poisson model flaw: A reasonable fit for "general" terms, but is a rather poor fit for topic-specific terms.
- Eliteness: Binary, represents aboutness. The document is about the concept denoted by the term. Documents are composed of "topical (elite)" terms and supportive more common terms.
    - Elite terms: good indexing terms. Let $$\pi = p(E_i = 1\vert R)$$.
        
        $$
        \begin{align*}
        RSV^{elite} &= \sum_{i\in q, tf_i > 0} c_i^{elite}(tf_i)\\
        c_i^{elite}(tf_i) &= \log\frac{p(TF_i = tf_i|R = 1)p(TF_i = 0|R = 0)}{p(TF_i = 0|R = 1)p(TF_i = tf_i|R = 0)}\\
        p(TF_i = tf_i|R) &= \sum_{E_i = \{0, 1\}} p(E_i, TF_i = tf_i|R)\\
        &= \sum_{E_i\in\{0, 1\}} [\pi p(TF_i = tf_i|E_i = 1, R) + (1 - \pi)p(TF_i = tf_i|E_i = 0, R)]
        \end{align*}
        $$
- 2-Poisson model: $$\pi$$ is probability that term is elite for document. The distribution is different depending on whether the term is elite or not.

    $$
    \begin{align*}
    p(TF_i = k_i|R) = \pi\frac{\lambda^k}{k!}e^{-\lambda} + (1 - \pi)\frac{\mu^k}{k!}e^{-\mu}
    \end{align*}
    $$

- Saturation function.
    - Approximate parameters for the 2-Poisson model with a simple parametric curve that has the same qualitative properties.
        
        $$
        \begin{align*}
        \frac{tf}{k_1 + tf}
        \end{align*}
        $$
    
    - For high values of $$k_1$$, increments in $$tf_i$$ continue to contribute significantly to the score.
    - Contributions tail off quickly for low values of $$k_1$$.
- Document length normalization.
    - A real document collection probably has both effects. Should apply some kind of partial normalization.
        - Verbosity: Suggest observed $$tf_i$$ too high.
        - Larger scope: Suggest observed $$tf_i$$ may be right.
    - Length normalization component.
        
        $$
        \begin{align*}
        B = \left((1 - b) + b\frac{dl}{avdl}\right),\quad 0 \leq b \leq 1
        \end{align*}
        $$

        - $$b = 1$$: Full document length normalization, 100% verbosity.
        - $$b = 0$$: No document length normalization, 100% scope.
        - Document length $$dl$$: $$dl = \sum_{i\in V} tf_i$$.
        - $$avdl$$: Average document length over collection.
- Okapi BM25.
    - Normalize $$tf$$ using document length.

        $$
        \begin{align*}
        tf_i' &= \frac{tf_i}{B}\\
        c_i^{BM25}(tf_i) &= \log \frac{N}{df_i}\cdot \frac{(k_1 + 1)tf_i'}{k_1 + tf_i'}\\
        &= \log \frac{N}{df_i}\cdot \frac{(k_1 + 1)tf_i}{k_1\left((1 - b) + b\frac{dl}{avdl} \right) + tf_i}
        \end{align*}
        $$

    - BM25 ranking function.

        $$
        \begin{align*}
        RSV^{BM25} = \sum_{i\in q} c_i^{BM25}(tf_i)
        \end{align*}
        $$

    - $$k_1$$ controls term frequency scaling (saturation function). $$k_1 = 0$$ is binary model. $$k_1$$ large is raw term frequency.
    - $$b$$ controls document length normalization. $$b = 0$$ is no length normalization (scope). $$b = 1$$ is relative frequency (fully scale by document length, verbose).

## Language Modeling for IR
### Query-Likelihood Model
- Rank documents by the probability that the query could be generated by the document model (i.e. same topic).

    $$
    \begin{align*}
    P(D|Q) = \frac{P(Q|D)P(D)}{P(Q)} \propto P(Q|D)P(D)
    \end{align*}
    $$

- Assumptions.
    1. Prior is uniform. 
    2. Unigram model: Sampling with replacement.
- Query likelihood.

    $$
    \begin{align*}
    RSV(Q, D) = \prod_{i = 1}^n P(q_i|D)
    \end{align*}
    $$

- Maximum likelihood estimate.

    $$
    \begin{align*}
    P(q_i|D) = \frac{f_{q_i, D}}{|D|}
    \end{align*}
    $$

- If query words are missing from document, score will be zero.
- Sparse data problem. $$\Rightarrow$$ Smoothing.
    - Feature space is large.
    - Relatively small amount of data for estimation.

**Smoothing**
- Smoothing by discounting: Laplace.
    
    $$
    \begin{align*}
    P_{laplace}(w) = \frac{c(w) + \alpha}{\sum_{w\in V}c(w) + \alpha|V|}
    \end{align*}
    $$

    - All unseen terms are assigned an equal probability.
- Smoothing by linear interpolation.
    - Estimate for unseen words: $$\alpha_D P(q_i\vert C)$$.
    - Estimate for words that occur is: $$(1 - \alpha_D)P(q_i\vert C) + \alpha_D P(q_i\vert C)$$.
    - Jelinek Mercer smoothing: $$\alpha_D$$ is a constant $$\lambda$$.
        
        $$
        \begin{align*}
        P(q_1, q_2, \dots, q_n|D) &= \prod_{j = 1}^k P(q_j|D)\\
        \Rightarrow P(q_1, q_2, \dots, q_n|D) &= \prod_{j = 1}^n [(1 - \lambda)P(q_j|D) + \lambda P(q_j|C)]\\
        \Rightarrow \log P(q_1, q_2, \dots, q_n|D) &= \sum_{j = 1}^n \log[(1 - \lambda)P(q_j|D) + \lambda P(q_j|C)]
        \end{align*}
        $$

        - $$\log P(Q\vert D)$$ proportional to the term frequency and inversely proportional to the collection frequency.
    - Dirichlet smoothing: $$\alpha_D$$ depends on document length.

        $$
        \begin{align*}
        \alpha_D &= \frac{\mu}{|D| + \mu}\\
        P(q_i|D) &= \frac{f_{q_i, D} + \mu\frac{c_{q_i}}{|C|}}{|D| + \mu}\\
        \log P(Q|D) &= \sum_{i = 1}^n \log\frac{f_{q_i, D} + \mu\frac{c_{q_i}}{|C|}}{|D| + \mu}
        \end{align*}
        $$

### Relevance Model
**Key idea**: Use document collection for query expansion, formalized as re-estimating a relevance model.\
**Relevance model**: Language model representing information need.\

- Assumption: Query and relevant documents are samples from this model.
- (negative) KL divergence as ranking score.
- Pseudo-relevance feedback for LM: Query expansion technique.
- Estimating the relevance model.
    
    $$
    \begin{align*}
    P(w|R) &\approx P(w|q_1, \dots, q_n) = \frac{P(w, q_1, \dots, q_n)}{P(q_1, \dots, q_n)}\\
    P(w, q_1, \dots, q_n) &= \sum_{D\in C}P(D)P(w, q_1, \dots, q_n|D) \\
    &= \sum_{D\in C}P(D)P(w|D)\prod_{i = 1}^n P(q_i|D)
    \end{align*}
    $$
    
    - $$P(w, q_1, \dots, q_n\vert D)$$ is simply a weighted average of the language model probabilities for $$w$$ in a set of documents, where the weights are the query likelihood scores for those documents. Approximation: Take top N documents, since weights are very small for low rank documents.

## Web Search and Recommender Systems
Two intuitions about **hyperlinks**:
1. The anchor text pointing to page B is a good description of page B (textual information).
2. The hyperlink from A to B represents an endorsement of page B, by the creator of page A (quality signal).

Both signals contain noise.

### PageRank
- Pages visited more frequently in a random walk on the web are the more important pages.
- A random walk on the Web: Incoming link counts + indirect citations + smoothing.

    $$
    \begin{align*}
    PR(p) = \frac{\alpha}{N} + (1 - \alpha)\sum_{q\rightarrow p}\frac{PR(q)}{O(q)}
    \end{align*}
    $$

    - $$N$$: Total number of pages in Web graph.
    - $$PR(q)$$: PageRank score of $$q$$ in current iteration.
    - $$O(q)$$: Number of outgoing links of page $$q$$.

### Diversification
**Reasons**
1. Queries are often short and **ambiguous**.
2. If we take query-document similarity as the most important ranking criterion, there might be a lot of **redundance** in the top-ranked results.

**Maximal marginal relevance**: Novelty-based diversification approach. Discount the relevance by the document's maximum similarity.

$$
\begin{align*}
f_{MMR}(q, d, D_q) = \lambda f_1(q, d) - (1 - \lambda)\max_{d_j\in D_q}f_2(d, d_j)
\end{align*}
$$

- $$f_1(q, d)$$: Relevance of $$d$$ to $$q$$.
- $$f_2(q, d)$$: Similarity of $$d_j$$ to $$d$$.

### Recommender Systems
Three types of models:
1. Collaborative filtering systems: Use user-item interactions.
2. Content-based recommender systems: Use the attribute information about the users and items (textual).
3. Knowledge-based recommender systems: The recommendations are based on explicitly specified user requirements. 

Evaluation:
1. Offline evaluation with benchmarks.
2. User studies.
3. Online evaluation (A/B testing).

Societal relevance:
1. Bias: Recommended items get more exposure.
2. Transparency: Explanations help the user.
3. Privacy: What information is stored about the user and how can the system minimize personal information use.
4. Filter bubbles / rabbit holes: Aim for some diversity in the recommendations.

## User Interaction and Conversational Search
### Learn from User Interactions
Find related queries in the query log:
- Based on common substring (starting with the same word).
- Based on co-occurrence in a session.
- Based on term clustering (embeddings).
- Based on clicks.
- Or a combination of these.

### Learning from Interaction Data
- Implicit feedback is noisy.
- Implicit feedback is biased.
    1. Position bias.
       - How to measure the effect of the position bias?\
         Intervention in the ranking: Swap documents + A/B testing.
       - Inverse Propensity Scoring (IPS) estimators can remove position bias. Propensity of observation = probability that a document is observed by the user. Weigh clicks depending on their observation probability.
    2. Selection bias.
    3. Presentation bias.

### Click Models
**Dependent click model (DCM)**
- Assumptions:
    - Cascade assumption: Examination is in strictly sequential order with no breaks.
    - Click probability depends on document relevance.
    - Examination probability depends on click.
    - Users are homogeneous: Their information needs are similar given the same query.
- From top to down, traverse - examine - decide - continue.

Simulation of Interaction
- Pros:
    - How the system behaves.
    - A large amount of data.
    - Low cost.
    - Replicated.
    - Understanding.
- Cons:
    - Models can become complex if we want to mirror realistic user
behaviour.
    - Choose from too many possibilities.
    - Represent actual user behavior / performance.
    - Application context.

### Query Generation
- Query expansion: No changes to the index are required.
    - Pseudo-relevance feedback (PRF), e.g. RM3
        - Top-ranked documents are assumed to be relevant, thus providing a source for additional query terms.
        - Get the most relevant terms from the top-n documents.
        - Add these to the query (expanded query).
        - Reissue this expanded query to obtain a new ranked list.
    - Does not always improve BERT-based rankers, because contextual embeddings work better with natural language queries.
- Document expansion: More context for a model to choose appropriate expansion terms. Can be applied at index time.
    - **Doc2query**: For document expansion.
        - Train a sequence-to-sequence model that, given a text from a corpus, produces queries for which that document might be relevant.
        - Train on relevant pairs of documents-queries.
        - Use the model to predict relevant queries for documents.
        - Append predicted queries to the original texts (document expansion).

### Conversational Search
**Approaches**
- Retrieval-based methods:
    - Select the best possible response from a collection of responses.
    - Common in Question Answering systems.
    - Uses IR techniques.
    - Pros: Source is transparent. Efficient. Evaluation straightforward.
    - Cons: Answer space is limited. Potentially not fluent. Less interactive.
- Generation-based methods:
    - Generate a response in natural language.
    - Common in social chatbots.
    - Uses generative language models.
    - Pros: Fluent and human-like. Tailored to user and input. More interactive.
    - Cons: Not necessarily factual, potentially toxic. GPU-heavy. Evaluation challenging.
- Hybrid methods: Retrieve information, then generate the response.

**Challenges**
- Logical self-consistency: Semantic coherence and internal logic (not contradicting earlier utterances).
- Safety, transparency, controllability: It is difficult to control the output of a generative model. This could lead to toxic language or hate speech.
- Efficiency: Time- and memory-consuming training and inference.

## Domain-specific IR
**Characteristics**
- Complex, highly specific search tasks.
- Long sessions (many queries) with not only query search but also browsing and link following.
- User-specific and context-specific tasks.
- Users who need control over the search process.

**High-recall (or even: full-recall) tasks**: No relevant information should be missed. 
- Prior art retrieval: Previously published patents. 
- eDiscovery: All relevant information in the case of a law suit or legal investigation. 
- Systematic reviewing: Collect an exhaustive summary of current evidence relevant to a research question.

**Shallow relevance assessments**: The collected assessments are biased toward the pool, and thereby towards the initial rankers. If we run completely different ranking models, we retrieve documents without relevance assessment.

**Challenges of query-by-document retrieval**
1. Long documents in the collection.
2. Long queries.
3. Domain-specific language.
4. Lack of labelled data.

**Dealing with (long) documents as queries**
- Use word-based methods (BM25) on the full text.
- Extract query terms and the use regular retrieval models.
- Truncate all documents or use only the abstract.
- Automatic summarization.
- Use paragraph-level retrieval and aggregation.

## References
1. Slides of Information Retrieval course, 2023 Spring, Leiden University.
