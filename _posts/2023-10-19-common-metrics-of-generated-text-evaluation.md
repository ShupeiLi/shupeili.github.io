---
layout: post
title: "Common Metrics of Generated Text Evaluation"
categories: machine_learning
tags: machine_learning
math: true

---

## Perplexity (PPL)
Perplexity is defined as the exponentiated average negative log-likelihood of a sequence. If we have a tokenized sequence $$X=(x_0, x_1, \cdots, x_t)$$, then the perplexity of $$X$$ is,

$$
\operatorname{PPL}(X)=\exp \left\{-\frac{1}{t} \sum_i^t \log p_\theta\left(x_i \mid x_{<i}\right)\right\}
$$

where $$\log p_{\theta}(x_i∣x_{<i})$$ is the log-likelihood of the $$i$$-th token conditioned on the preceding tokens $$x_{<i}$$ according to our model. Intuitively, it can be thought of as an evaluation of the model’s ability to predict uniformly among the set of specified tokens in a corpus. Importantly, this means that the tokenization procedure has a direct impact on a model’s perplexity which should always be taken into consideration when comparing different models.

This is also equivalent to the exponentiation of the cross-entropy between the data and model predictions.

## BLEU
BLEU (bilingual evaluation understudy) is an algorithm for evaluating the quality of text which has been machine-translated from one natural language to another. Quality is considered to be the correspondence between a machine's output and that of a human: "the closer a machine translation is to a professional human translation, the better it is".

BLEU's output is always a number between 0 and 1. This value indicates how similar the candidate text is to the reference texts, with values closer to 1 representing more similar texts.

Fix a candidate corpus $$\hat{S} := \left(\hat{y}^{(1)}, \dots, \hat{y}^{M} \right)$$, and reference candidate corpus $$S = (S_1, \dots, S_M)$$, where each $$S_i := (y^{(i, 1)}, \dots, y^{(i, N_i)})$$.

**Modified n-gram precision**

Define the modified n-gram precision function to be:

$$
p_n(\hat{S} ; S):=\frac{\sum_{i=1}^M \sum_{s \in G_n\left(\hat{y}^{(i)}\right)} \min \left(C\left(s, \hat{y}^{(i)}\right), \max _{y \in S_i} C(s, y)\right)}{\sum_{i=1}^M \sum_{s \in G_n\left(\hat{y}^{(i)}\right)} C\left(s, \hat{y}^{(i)}\right)}
$$

where $$\sum_{s \in G_n(\hat{y})} C(s, y)=$$ number of n-substrings in $$\hat{y}$$ that appear in $$y$$.

**Brevity penalty**

$$
\mathrm{BP(\hat{S}; S)}=\left\{\begin{array}{ll}
1 & \text { if } c>r \\
e^{(1-r / c)} & \text { if } c \leq r
\end{array}\right.
$$

$$c$$ is the length of the candidate corpus, that is, 

$$
c := \sum_{i = 1}^M |\hat{y}^{(i)}|
$$

where $$\vert y \vert$$ is the length of $$y$$. $$r$$ is the effective reference corpus length, that is, 

$$
r := \sum_{i = 1}^M |y^{(i, j)}|
$$

where $$y^{(i, j)} = \text{argmin}_{y\in S_i}\vert \vert y\vert - \vert\hat{y}^{(i)}\vert\vert$$, that is, the sentence from $$S_{i}$$ whose length is as close to $$\vert \hat{y}^{(i)}\vert$$ as possible.

**Final definition of BLEU**

Given a weighting vector $$w := (w_1, w_2, \dots)$$, $$\sum_{i = 1}^\infty w_i = 1$$, $$w_i\in[0, 1]$$, BLEU is defined by:

$$
BLEU_w(\hat{S} ; S):=B P(\hat{S} ; S) \cdot \exp \left(\sum_{n=1}^{\infty} w_n \ln p_n(\hat{S} ; S)\right)
$$

In words, it is a weighted geometric mean of all the modified n-gram precisions, multiplied by the brevity penalty.

## Distinct-n

Distinct-{1, 2}: degree of diversity by calculating the number of distinct unigrams and bigrams in generated responses. The value is scaled by total number of generated tokens to avoid favoring long sentences.

There are typically two common methods to calculate "distinct-n" metrics, such as D1 (Distinct-1) and D2 (Distinct-2), and they are often referred to as "sample-based" and "system-level" calculations:

1. Sample-Based Distinct-n (intra-distinct): calculate distinct-n metrics on individual samples of generated text and then average the results. It's a more fine-grained assessment, measuring diversity at the text sample level. Each generated sample has its distinct-n score, and you take the average over all samples.

    $$
    \begin{align*}
    \text{D1_sample} &= \frac{\text{count of unique n-grams in sample}}{\text{total n-grams in sample}}\\
    \text{D1} &= \frac{\text{sum of D1_sample for all samples}}{\text{number of samples}}
    \end{align*}
    $$

2. System-Level Distinct-n (inter-distinct): treat the entire generated dataset as a single unit and calculate distinct-n metrics on the combined dataset. This approach provides a system-level measure of diversity and is typically easier to calculate.

    $$
    \begin{align*}
    \text{D1_system} &= \frac{\text{count of unique n-grams in the entire dataset}}{\text{total n-grams in the entire dataset}}\\
    \text{D2_system} &= \frac{\text{count of unique n-gram pairs in the entire dataset}}{\text{total n-gram pairs in the entire dataset}}
    \end{align*}
    $$

## References
1. [Hugging face documents: PPL](https://huggingface.co/docs/transformers/perplexity)
2. [Wikipedia: BLEU](https://en.wikipedia.org/wiki/BLEU)
3. Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a Method for Automatic Evaluation of Machine Translation. In *Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics*, pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics.
4. Jiwei Li, Michel Galley, Chris Brockett, Jianfeng Gao, and Bill Dolan. 2016. A Diversity-Promoting Objective Function for Neural Conversation Models. In *Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 110–119, San Diego, California. Association for Computational Linguistics.
5. Conversations with OpenAI ChatGPT-3.5.
