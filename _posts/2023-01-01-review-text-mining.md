---
layout: post
title: "Review: Text Mining"
categories: machine learning
tags: machine-learning

---

## Tasks
Case: Discover side effects for hypertension medications.\
The text mining pipeline:
1. Filter the data: Retrieve relevant messages.
2. Process the data: Clean, anonymize.
3. Create training data: Human labelling.
4. Identify medication names: Named entity recognition.
5. Identify side effects: Named entity recognition.
6. External knowledge needed: Ontology.
7. Relations between medications and side effects: Relation extraction.

### Pre-processing
Challenges of text data:
1. Text data is unstructured / semi-structured.
2. Text data can be multi-lingual.
3. Text data is noisy.
    - Noisy encoding and typography might give challenges in processing.
    - Noisy attributes: Spelling errors, OCR errors.
4. Language is infinite.
    - Heaps' law: $$V_R(n) = Kn^\beta$$, where $$V_R$$ is the number of distinct words in an instance text of size $$n$$. $$K$$ and $$\beta$$ are free parameters determined empirically. With English text corpora, typically $$K$$ is between 10 and 100, and $$\beta$$ is between 0.4 and 0.6. 

#### Go from Raw Text to Clean Text
**Written text:**
- Digitized documents: Optical character recognition (OCR).
- Born-digital documents.

**Noises:**
- Scanned text and born-digital PDFs: OCR errors, character encoding errors.
    - Character encoding: The way that a computer displays text in a way that humans can understand. That is, translate a string of 0s and 1s to a character.\
    Example: ASCII, a 7-bit encoding based on English alphabet. Unicode, the universal standard for all writing systems (e.g. UTF-8).
- Semi-structured text: Text with markup.
    - Markup: Meta-information in a text file that is clearly distinguishable from the textual content. In the case of XML and json, markup often provides useful information in text processing.

#### Edit Distance
Measure string similarity:
- Spelling correction / normalization.
- Match names or terms to databases that might contain spelling errors or typos.

Minimal edit distance: Minimal number of editing operations (insertion, deletion, substitution) needed: string 1 $$\rightarrow$$ string 2.

Computing minimal edit distance: Dynamic programming.

$$
\begin{align*}
D[i, j] = \min
\begin{cases}
D[i - 1, j] + \text{del-cost}(source[i])\\
D[i, j - 1] + \text{ins-cost}(target[j])\\
D[i - 1, j - 1] + \text{sub-cost}(source[i], target[j])\\
\end{cases}
\end{align*}
$$

In slides, the cost of all operations is 1 (Levenshtein distance).

**Exercises**
1. Levenshtein distance: "casts" to "fast".
    <details>
    <summary>Solution</summary>
      3.
    </details>
2. Levenshtein distance: "a cat" to "an act".
    <details>
    <summary>Solution</summary>
      3.
    </details>

#### Tokenization
**Token**: An instance of a word or term occurring in a document.\
**Term**: A token when used as features (or a index), generally in normalized form (e.g. lowercased).\
**Token count**: The number of words (running) in a collection / document, including duplicates.\
**Vocabulary size**: The number of unique terms. The feature size when we use words as features.

**Tokenization**
1. Remove punctuation.
2. Split on whitespaces characters.

**Stop words**
- Definition: Extremely common words that don't carry any content.
- Remove stop words in: topic modelling, keyword extraction.
- NEVER remove stop words in: sequence labelling tasks, classification tasks with small data.

#### Sentence Splitting
Need sentence splitting for tasks that require sentence-level analysis, for example:
- Find the most similar sentences in a collection.
- Extract the relations between two entities within one sentence.
- Sentence-level sentiment analysis.
- Input for Transformer models with limited input length.

#### Lemmatization and Stemming
Normalize specific word forms to the same term:
1. Reduce the number of features.
2. Generalize better, especially for small datasets.

Lemma and stem are two types of basic word forms.\
**Lemma**: Dictionary form of a word.\
**Stem**: The portion of a word that is common to a set of (inflected) forms when all affixes are removed and is not further analyzable into meaningful elements.

Prefer lemmas over stems. Stemming can be effective for very small collections.

### Classification
1. Task definition.
    - Text unit (aka. document): Complete documents, sections, sentences.
    - Categories.
2. Example data (Refer to Resources).
3. Pre-processing.
    - Words as features: Tokenization. For most applications, apply lowercasing and removal of punctuation.
    - Additional pre-processing steps might include: Remove stop words, lemmatization or stemming, add phrases as features (e.g. "PhD defense").
4. Feature extraction.
    - Decide on vocabulary size: Feature selection.
        - Goal: Reduce dimensionality and overfitting.
        - Global term selection: Overall term frequency is used as cutoff.
        - Local term selection: Each term is scored by a scoring function that captures its degree of correlation with each class it occurs in (e.g. $$\chi^2$$ test).
        - Only use top-n terms for classifier training.
    - Term-document matrix: Assign weights to terms defined in vocabulary (feature values $$\rightarrow$$ features).
5. Classifier learning.
    - Use word features: Need an estimator that is well-suited for high-dimensional and sparse data, e.g., naive Bayes, SVM, random forest.
    - Dense embeddings: Can use neural networks.
    - Transfer learning: Pre-trained embeddings with transformers.
6. Evaluation.

**Exercises**
1. "you will. To" 4-grams.
    <details>
    <summary>Solution</summary>
     you_ , ou_w, u_wi, _wil, will, ill., ll._, l._T, ._To
    </details>

### Information Extraction
#### Named Entity Recognition
**Name entity**: A sequence of words that designates some real world entity (typically a name).\
**NER** is a machine learning task based on sequence labelling (one label per token):
- Word order matters.
- One entity can span multiple words.
- Multiple ways to refer to the same concept.

$$\rightarrow$$ The extracted entities often need to be linked to a standard form (ontology or knowledge base).

Format of training data: **IOB tagging**.

Challenge of NER:
1. Ambiguity of segmentation (boundaries).
2. Type ambiguity (e.g. JFK).
3. Shift of meaning (e.g. president of US).

Limitations of using a list of names to recognize entities:
1. Entities are typically multi-word phrases.
2. List is limited.
3. Variants.

#### Relation Extraction
**Co-occurrence based**
- Rely on the law of big numbers: There will be noise, but the output can still be useful.
- Assumptions: Entities that frequently co-occur are semantically connected.
- Use a context window (e.g. sentence) to determine co-occurrence. We can create a network based on this.

**Supervised learning**
- The most reliable. However, supervised learning requires labelled data.
- Assumptions: Two entities, one relation. Relation is verbalized in one sentence, or one passage.
- Classification problem: Find pairs of named entities and apply a relation classification on each pair.

**Distant supervision**
- If labelled data is limited.
- Use the knowledge base to identify relations in the text and discover relations that are not yet in the knowledge base.
- Start with a large, manually created knowledge base. Find occurrences of pairs of related entities from the database in sentences. Train a supervised relation extraction classifier on the found entities and their context. Apply the classifier to sentences with yet unconnected other entities in order to find new relations.

### Summarization

### Evaluation
- Evaluation of complete application (extrinsic evaluation): Human.
- Evaluation of the components (intrinsic evaluation): Need ground truth labels. Existing labels or human-assigned labels.

#### Metrics
$$A$$ is the set of labels assigned by algorithms, while $$T$$ is the set of true labels.
1. Accuracy: Proportion of samples that is correctly labled.
    - Cons: Imbalance data. May be more interested in correctness of labels than completeness of labels.
2. Precision: Proportion of the assigned labels that are correct. How many selected items are relevant?

   $$\begin{align*}\text{Precision} = \frac{|A\cap T|}{|A|} = \frac{tp}{tp + fp}\end{align*}$$
3. Recall: Proportion of the relevant labels that were assigned. How many relevant items are selected?

   $$\begin{align*}\text{Recall} = \frac{|A\cap T|}{|T|} = \frac{tp}{tp + fn}\end{align*}$$
4. $$F_1$$ score.

   $$\begin{align*}F_1 = 2\cdot \frac{\text{precision}\cdot \text{recall}}{\text{precision} + \text{recall}}\end{align*}$$

## Models
### Vector Semantics
- Linguistics: Distributional hypothesis.
    - Distributional hypothesis: The context of a word defines its meaning. Words that occur in similar contexts tend to be similar.
- Information retrieval: Vector space model (VSM).
    - VSM: Documents and queries represented in a vector space, where the dimensions are the words.
    - Problems of VSM:
        1. Synonymy (e.g. bicycle and bike) and polysemy (e.g. bank, chips).
        2. Encodings are arbitrary.
        3. Provide no useful information to the system.
        4. Lead to data sparsity.

Alternatives: Do not represent documents by words, but by:
- Topics: Topic modelling.
- Concepts: Word embeddings.

### Text Representation
#### Bag of Words
Text as classification object. Each word in the collection becomes a dimension in the vector space (feature). Word vectors are high-dimensional and sparse.\
Word order, punctuation, sentence / paragraph borders are not relevant.

**Note**: Text as sequence. If we want to extract knowledge from text, sequential information matters: word order (sequence), punctuation, capitalization.

**Zipf's Law**\
Given a text collection, the frequency of any word is inversely proportional to its rank in the frequency table.

#### Word Embeddings
The dense representation for text. The vector space is lower-dimensional and dense. The dimensions are latent (not individually interpretable) and learnt from data. Similar words are close to each other in the space (Distributional Hypothesis).

**Word2vec**
- A particularly computationally-efficient predictive model for learning word embeddings from raw text. A supervised problem on unlabeled data --- self supervision. Its inputs are encoded as one-hot vectors.
- Train a neural classifier on a binary prediction task. Method: Skip-gram with negative sampling.
    1. Positive sample: Target word and a neighboring context word.
    2. Negative sample: Randomly sample other words in the lexicon.
    3. Classifier: Distinguish two cases. Optimize the similarity (dot product) with SGD.
- Take the learned classifier weights on the hidden layer as the word embeddings.

Advantages:
1. Scalability: Size, time, parallelization.
2. Pre-trained word embeddings trained by one can be used by others.
3. Incremental training: Train on one piece of data, save results, continue training later on.

Note: For most applications, we need lemmatization for word2vec. Because we don't want representations for both "bicycle" and "bicycles" in our vector space.

From word embeddings to document embeddings:
1. Take the average vector over all words in the document.
2. Use doc2vec.
3. Use BERT embeddings, i.e., SentenceBERT, which is highly efficient but only for short documents / passages.

### Term Weighting
#### Term Frequency (TF)
The term count $$tc_{t, d}$$ of term $$t$$ in document $$d$$ is defined as the number of times that $$t$$ occurs in $$d$$.

**TF formula**:

$$
\begin{align*}
tf_{t, d} = 
\begin{cases}
1 + \log_{10} tc_{t, d}\quad &\text{if } tc_{t, d} > 0\\
0\quad &\text{otherwise}
\end{cases}
\end{align*}
$$

#### Inverse Document Frequency (IDF)
- The most frequent terms are not very informative.
- $$df_t$$: The document frequency, i.e., the number of documents that $$t$$ occurs in.
- $$df_t$$ is an inverse measure of the informativaness of term $$t$$.

**IDF formula**:

$$
\begin{align*}
idf_t = \log_{10} \frac{N}{df_t}
\end{align*}
$$

**TF-IDF formula**:

$$
\begin{align*}
tf\text{-}idf = tf * idf
\end{align*}
$$

### Naive Bayes
Assumptions: Conditional independence and positional independence.\
Maximize the posterior probability:

$$
\begin{align*}
c_{MAP} = argmax_{c\in C} P(d|c)P(c)
\end{align*}
$$

**Notations**
- $$P(c)$$: Prior probability. Approximate by $$\vert D_c\vert / \vert D \vert$$, where $$D_c$$ is the number of documents for class $$c$$ and $$D$$ is the total number of documents.
- $$P(d\vert c) = P(t_1,\dots, t_k\vert c) = P(t_1\vert c)\cdots P(t_\vert c)$$, where $$P(t\vert c) = \frac{T_{ct} + 1}{\left(\sum_{t'\in V} T_{ct'}\right) + \vert V\vert}$$ (add-one / Laplace smoothing).
- $$T_{ct}$$: The number of occurrences of $$t$$ in training documents from class $$c$$.
- $$\vert V\vert$$: The size of the vocabulary.

**Exercises**
1. Calculate $$P(\text{no spam}\vert 5)$$ and $$P(\text{spam}\vert 5)$$.

   |ID|Content|Class|
   |---|---|---|
   |1|request urgent interest urgent|spam|
   |2|assistance low interest deposit|spam|
   |3|symposium defence june|no spam|
   |4|siks symposium deadline june|no spam|
   |5|registration assistance symposium deadline|?|

<details>
<summary>Solution</summary>
$$
\begin{align*}
P(\text{no spam}\vert 5) &= \frac{2}{4}\cdot\frac{1}{18}\cdot\frac{1}{18}\cdot\frac{3}{18}\cdot\frac{2}{18} = 2.86 \times 10^{-5}\\
P(\text{spam}\vert 5) &= \frac{2}{4}\cdot\frac{1}{19}\cdot\frac{2}{19}\cdot\frac{1}{19}\cdot\frac{1}{19} = 7.67 \times 10^{-6}
\end{align*}
$$

Therefore, message 5 is no spam.
</details>

### Sequence Labelling
#### Hidden Markov Model (HMM)
A probabilistic sequence model. Denote the tag as $$t$$ and th word as $$w$$:

$$
\begin{align*}
\hat{t}_{1:n} = argmax_{t_1, \dots, t_n} P(t_1, \dots, t_n\vert w_1, \dots, w_n) \approx argmax_{t_1, \dots, t_n} \prod_{i=1}^n P(w_i\vert t_i)P(t_i\vert t_{i-1})
\end{align*}
$$

- $$P(w_i\vert t_i)$$: Emission probability.
- $$P(t_i\vert t_{i - 1})$$: Transition probability.

Probabilities are estimated by counting on a labelled training corpus.\
**Decoding**: Determine the hidden variables sequence corresponding to the sequence of observations.

#### Feature-based NER
Supervised learning: Features, IOB-labelled texts.

**Typical features**:
1. Part-of-speech
2. Gazatteer.
3. Word shape.

#### Conditional Random Fields (CRF)
**Motivation**: It is hard for generative models like HMMs to add features directly into the model.
- A discriminative undirected probabilistic graphical model.
- Feature vectors: Take rich representations of observations.
- Take previous labels and context observations into account.
- Optimize the sequence as a whole: Viterbi algorithm.

$$
\begin{align*}
P(\bar{y}\vert \bar{x}; w) = \frac{\exp \left(\sum_i\sum_j w_jf_j(y_{i - 1}, y_i, \bar{x}, i) \right)}{\sum_{y'\in Y}\exp\left(\sum_i\sum_j w_jf_j(y_{i-1}', y_i', \bar{x}, i)\right)}
\end{align*}
$$

#### Neural Sequence Models
**biLSTM-CRF**
- Use CRF layer on top of the bi-LSTM output to impose strong constraints for neighboring tokens (e.g., I-PER tag must follow I-PER or B-PER). Softmax optimization alone is insufficient.
- The nodes for the current timestamp (token) are connected to the nodes for the previous timestamp.

#### Transformer Models
The transformer architecture is the current state-of-the-art model for NER. In particular, BERT.
- Transformer architecture is an encoder-decoder architecture.
- Input is processed in parallel.
- Can model longer-term dependencies because the complete input is processed at once.
- Quadratic space complexity: $$O(n^2)$$, for input length of $$n$$ items.

**Attention mechanism**\
The model has access to all of the input items. Each input token is compared to all other input tokens (dot product).

**BERT**\
Pre-training of deep bidirectional transformers for language understanding. Self-supervised pre-training based on language modelling.\
Input: WordPiece = Token embeddings + Segment embeddings + Position embeddings.

BERT for sentence similarity:
1. Concatenate sentences in the input.
2. SBERT: Independent encoding of the two sentences with a BERT encoder. Then measure similarity between the two embeddings.

### Sequence-to-sequence

### Transfer Learning
Inductive transfer learning: Transfer the knowledge from pre-trained language models to any NLP task.
1. Pre-training: Unlabelled data, different tasks (self-supervision).
2. Fine-tuning: Initialized with pre-trained parameters. Labelled data from downstream tasks (supervised learning).

**Zero shot**: Use a pre-trained model without fine-tuning.\
**Few-shot learning**: Fine-tuning with a small number of samples.

Challenges of state-of-the-art methods:
1. Time and memory expensive: Pre-training, fine-tuning, inference.
2. Hyperparameter tuning:
    - Optimization on validation set (takes time).
    - Adoption of hyperparameters from pre-training task (might be suboptimal).
3. Interpretation / explainability: Additional effort.

### Resources
#### Labeled Data
**Existing labelled data**
- Benchmark data
    - Pros: High quality. Re-usable. Compare results to others.
    - Cons: Not available for every specific problem and data type.
- Existing human labels: Labels that were added to items by humans but not originally created for training machine learning models. 
    - Pros: High quality. Potentially large. Often freely available.
    - Cons: Not available for every specific problem and data type. Not always directly suitable for training classifiers.
- Labelled user-generated content.
    - Pros: Potentially large. Human-created. Freely available, depending on the platform.
    - Cons: Noisy, often inconsistent. May be low-quality. Indirect signal.

**Create labelled data**\
Keywords: Annotation guidelines, crowdsourcing (quality control), inter-rater agreement (Cohen's Kappa).

**Cohen's Kappa**

$$
\begin{align*}
\kappa = \frac{P(a) - P(e)}{1 - P(e)}
\end{align*}
$$

- $$P(a)$$: Actual agreement: Percentage agreed.
- $$P(e)$$: Expected agreement.

Interpretation of Kappa: Agreement.

|<0|0-0.20|0.21-0.40|0.41-0.60|0.61-0.80|0.81-1|
|---|---|---|---|---|---|
|No|Slight|Fair|Moderate|Substantial|Almost perfect|

**Exercises**
1. Calculate $$\kappa$$.

   |||A2||
   |---|---|---|---|
   |||Yes|No|
   |A1|Yes|20|5|
   ||No|10|15|

<details>
<summary>Solution</summary>
$$
\begin{align*}
\kappa = \frac{0.7 - 0.5}{1 - 0.5} = 0.4
\end{align*}
$$
</details>

#### Unlabeled Data

#### Other
- Dictionaries (gazetteers).
- Ontology / Knowledge bases.
    - Motivation: Multiple extracted mentions can refer to the same concept.
    - Normalization of extracted mentions: Ontology / knowledge bases.
    - Ontology linking approaches:
        1. Define it as text classification task with the ontology items as labels. Challenges: The label space is huge. We don't have training data for all items.
        2. Define it as term similarity task. Use embeddings trained for synonym detection.
- General domain or specific domain.

## Applications

### Opinionated Content

#### Sentiment Analysis

#### Stance Detection

### Industrial Text Mining
#### CV-vacancy Matching

### Domains

#### Biomedical Text Mining

## References
1. Slides of Text Mining course, 2022 Fall, Leiden University.
2. Jurafsky, Dan and James H. Martin. *Speech and Language Processing*. 2000.
3. [Heaps' law.](https://en.wikipedia.org/wiki/Heaps%27_law)
