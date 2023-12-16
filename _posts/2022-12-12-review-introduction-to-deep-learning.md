---
layout: post
title: "Review: Introduction to Deep Learning"
categories: machine_learning
tags: machine_learning
math: true
pseudocode: true

---

## Building Blocks
### Brief History
**Early booming (50's - early 60's)**\
Rosenblatt (1958)
- Perceptron: Network of threshold nodes for pattern classification.
- Perceptron convergence theorem: Everything that can be represented by a perceptron can be learned.

**The setback (mid 60's - late 70's)**\
Single layer perceptron cannot represent simple functions such as XOR.\
Scaling problem: Connection weights may grow infinitely.

**Renewed enthusiasm and progress (80's - 90's)**\
Backpropagation: Multi-layer feed forward nets.\
Unsupervised learning.\
First applications:
- NetTalk: A text-to-speech network that learns to pronounce English text. 
    - Input: Phrase and phonetic representation.
    - 7 $\times$ 29 inputs $\rightarrow$ 80 hidden units $\rightarrow$ 26 output units
- ALVINN: Drive a car.
- FALCON: A fraud detection system.

Problems of adding layers: Overfitting, local minima entrapment, vanishing or exploding gradients.\
Why multiple layers?
1. In theory, one hidden layer is sufficient to model any function with an arbitary accuracy. However, the number of required nodes and weights grows exponentially fast.
2. Deeper network, less nodes.

**Dominance of new techniques (90's - 2005)**\
SVM, kernel methods, ensemble methods, random forests

**Deep Learning Revolution (2006 - present)**\
Enabling factors:
1. Availability of big data.
2. Powerful hardware (GPU).
3. New algorithms and architectures.

### Perceptron
#### Perceptron learning algorithm
<pre class="pseudocode">
\begin{algorithm}
\caption{Perceptron learning algorithm}
\begin{algorithmic}
\STATE Initial $w$ randomly
\WHILE{$\exists$ misclassified training examples}
    \STATE Select a misclassified example $(x, d)$
    \STATE $w_{new} = w_{old} + \eta dx$
\ENDWHILE
\end{algorithmic}
\end{algorithm}
</pre>

$\eta$: The learning rate.

#### Cover's Theorem
What is the chance that a randomly labeled set of $N$ points in $d$-dimensional space, is linearly separable?

$$
\begin{align*}
F(N, d) = 
\begin{cases}
1 &N \leq d + 1;\\
\frac{1}{2^{N-1}}\sum_{i=0}^d \binom{N-1}{i} & N > d+1.
\end{cases}
\end{align*}
$$

In highly dimensional spaces:
1. #points in $$d$$-dimensional space $$<$$ 2$$d$$ $$\rightarrow$$ almost always linearly separable.
2. #points in $$d$$-dimensional space $$>$$ 2$$d$$ $$\rightarrow$$ almost always not linearly separable.

#### Gradient Descent Algorithm
Goal: Find a minimum of a function $$f$$.\
Steps:
1. Start with an arbitrary point $$x$$.
2. Find a direction in which $$f$$ is decreasing most rapidly: $$-\nabla x$$.
3. Make a small step in this direction: $$x = x - \eta\nabla x$$.
4. Repeat the whole process.

#### Perceptron for Multi-class Problems
**Linear Separability for Multi-class Problems**\
There exist $$c$$ linear discriminant functions $$y_1(x), \dots, y_c(x)$$ such that each $$x$$ is assigned to class $$C_k$$ if and only if $$y_k(x) > y_j$$ for all $$j\neq k$$.

**Generalized Perceptron Convergence Theorem**\
If the $$c$$ sets of points are linearly separable then the generalized perceptron algorithm terminates after a finite number of iterations, separating all classes.

**Perceptron VS SVM**

||Perceptron|SVM|
|---|---|---|
|Inspiration|Simulating biological networks|Statistical learning theory|
|Training|Gradient descent algorithm|Quadratic optimization|
|Kernel trick|No|Yes|
|Multi-class classification|Yes|No|
|Probability of prediction|Yes|No|

## Training
### Backpropagation Algorithm
Key idea: Chain rule.\
Search for weight values that minimize the total error of the network.\
The repeated application of two passes:
- Forward pass: Compute the error of each neuron of the output layer.
- Backward pass: The error is propagated backwards through the network layer by layer via the generalized delta rule. Update all weights.

Highlights:
- No guarantees of convergence: Learning rate is too big or too small.
- In case of convergence: Local (or global) minimum. $$\rightarrow$$ Try several starting configurations and learning rates.

**Difference between gradient descent and backpropagation**\
“Gradient descent” is a general technique for finding (local minima) of a function which involves calculationg gradients (or partial derivatives) of the function, while “backpropagation” is a very efficient method for calculating gradients of “well-structured” functions such as multi-layered networks.

### Practical Strategies
#### Three Update Strategies
1. Full batch mode: All input at once.
2. (Mini) batch mode: A small random sample of inputs.
3. On-line mode: One input at a time.

#### Stopping Criteria
- Total mean squared error change.
- Generalization based criterion.\
  Early stopping: Stop trainig as soon at the error on the validation set increases.

#### Different Tasks

|Task|Activation Function|Error Function|
|---|---|---|
|Regression|Linear|Sum square error|
|Binary classification|Logistic|Cross entropy|
|Multi-class classification|Softmax|Cross entropy|

#### Activation Functions

|Function|Comment|
|---|---|
|Logistic|Vanishing gradients|
|Tanh|Vanishing gradients|
|Linear|A combination of linear functions is a linear function|
|ReLU|1. No multiplications, just "sign tests"; 2. Dying neurons when 0|
|Leaky ReLU|At most 1 multiplication, and just one "sign tests"|
|ELU|Better than Leaky ReLU|
|SELU|Preserve variance of activations when passing layers|

#### Weights Initialization
**Traditional strategies**\
Standard Gaussian ($$\mu$$ = 0, $$\sigma$$ = 1); Uniform $$[-1, 1]$$\
Drawback:\
Variance of outputs (of a single node / layer) increases, so the deeper we go the bigger the variance.

**Weights are initialized at random**\
If all weights are initialized to the same value then all the weights (in each layer) are updated in the same way, effectively reducing the whole network to a stack of single neurons.

**Solutions**\
For each layer calculate:\
$$\text{fan}_{\text{in}}$$: number of connections "to" the given layer\
$$\text{fan}_{\text{out}}$$: number of connections "from" the given layer\
$$\text{fan}_{\text{avg}} = (\text{fan}_{\text{in}} + \text{fan}_{\text{out}}) / 2$$

Then generate weights (for each layer) from a normal distribution with $$\mu$$ = 0 and:\
(Glorot) For logistic, Tanh, softmax: $$\sigma^2 = 1 / \text{fan}_{\text{avg}}$$\
(He) For ReLU and its variants: $$\sigma^2 = 2 / \text{fan}_{\text{in}}$$\
(LeCun) For SELU: $$\sigma^2 = 1 / \text{fan}_{\text{in}}$$

#### Batch Normalization
Key idea: Normalize, scale, and shift each batch.\
$$\gamma$$, $$\beta$$ are tuneable parameters and can be optimized with SGD.\
$$\mu$$, $$\sigma$$ are computed parameters.

$$
\begin{align*}
\mu_{\mathcal{B}} &= \frac{1}{m}\sum_{i=1}^{m} x_i\\
\sigma_{\mathcal{B}}^2 &= \frac{1}{m} \sum_{i=1}^m (x_i - \mu_{\mathcal{B}})^2\\
\hat{x}_i &= \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}\\
y_i &= \gamma\hat{x}_i + \beta \equiv \text{BN}_{\gamma, \beta}(x_i)
\end{align*}
$$

Apply the chain rule to the BN transformation to propagate gradients of the loss function.

**Test mode**\
During training, every batch is normalized separately but $$\beta$$ and $$\gamma$$ are tuned to be optimal for all batches.\
We need to estimate $$\mu$$ and $$\sigma$$ over the training set to normalize incoming data in the test mode. How?
1. Treat the whole training set as one big batch (very expensive).
2. Apply the moving average technique to estimate $$\mu$$ and $$\sigma$$ process batch after batch.

> **Exponential moving average**\
> For all batches:
> $$
> \begin{cases}
> \mu_1 = \mu_{\mathcal{B_1}}\\
> \mu_{n + 1} = 0.01 \mu_{\mathcal{B_{n+1}}} + 0.99 \mu_n
> \end{cases}
> $$
{: .prompt-info }

#### Gradient Clipping
Gradient clipping is a technique that tackles exploding gradients. The idea of gradient clipping is very simple: If the gradients get too large, we rescale it to keep it small. More precisely, if $$\Vert g \Vert > c$$, then

$$
\begin{align*}
g = c\cdot \frac{g}{\Vert g\Vert}
\end{align*}
$$

where $$c$$ is a hyperparameter, $$g$$ is the gradient, and $$\Vert g\Vert$$ is the norm of $$g$$. Since $$g / \Vert g\Vert$$ is a unit vector, after rescaling the new $$g$$ will have norm $$c$$. Note that if $$\Vert g\Vert < c$$, then we don’t need to do anything.

Gradient clipping ensures the gradient vector $$g$$ has norm at most $$c$$. This helps gradient descent to have a reasonable behaviour even if the loss landscape of the model is irregular.

#### Regularization Methods
- L1 or L2 regularization: Penalty on too big values of weights.
- Alternative loss functions: MSE, cross entropy, LogLoss, HuberLoss, etc.
- Batch normalization. 
- Node dropout: At every training step (processing a batch), each node has a chance $$0 < p < 1$$ to be disabled (not from the output layer). All nodes are active when testing (a form of bagging a collection of networks).
- Monte Carlo dropout: Normal dropout + Aggregate predictions over random sub-networks when testing.

#### Transfer Learning
Transfer learning is a machine learning technique where a model trained on one task is re-purposed on a second related task.\
Methods:
1. Train a model to reuse it.
2. Use a pre-trained model.
3. Feature extraction (representation learning).

**Transfer learning on grayscale images**
1. Adding additional channels to each greyscale image.
2. Modifying the first convolutional layer of the pretrained network.

#### Variants of the Gradient Descent Algorithm
**SGD with momentum**
1. Require: Initial parameter $$\theta$$, initial velocity $$v$$, learning rate $$\epsilon$$, momentum parameter $$\alpha$$.
2. Sample a minibatch of $$m$$ examples from the training set.
3. Compute gradient estimate $$g = \frac{1}{m}\nabla_{\theta}\sum L$$.
4. Compute velocity update $$v = \alpha v - \epsilon g$$.
5. Apply update $$\theta = \theta + v$$.

**SGD with Nesterov accelerated gradient**
1. Require: Initial parameter $$\theta$$, initial velocity $$v$$, learning rate $$\epsilon$$, momentum parameter $$\alpha$$.
2. Sample a minibatch of $$m$$ examples from the training set.
3. Apply interim update $$\tilde{\theta} = \theta + \alpha v$$.
4. Compute gradient estimate **at interim point** $$g = \frac{1}{m}\nabla_{\tilde{\theta}}\sum L$$.
5. Compute velocity update $$v = \alpha v - \epsilon g$$.
6. Apply update $$\theta = \theta + v$$.

## Convolutional Networks
### Concepts
**Randomly permuting all pixels**\
The accuracy achieved by a single layer perceptron (or MLP) on the random permuted data is the same as on the original data.\
Error function: $$n!$$ local minima.

**Filter**: A feature detector. Return high values when the corresponding patch is similar to the filter matrix.\
**Convolution**: An operation that takes as input a tensor and applies a "local convolution operation" (kernel, filter, convolutional matrix, convolutional tensor) to "all fragments" of the input.\
**Convolutional layer**: A layer of neurons that performs the same operation on fragments of the input.\
**Feature map**: The result of applying convolutional layer to data.\
**Stride**: The step size when moving a filter over the input. Reduce resolution. Shrink the image.\
**Padding**: Artificially increase the size of the input (e.g. by zeros, mirror reflections, ...) to preserve the original input size in the feature map. 'same' - add zeros when needed. 'valid' - accept the loss of some input. Don't pad the input.\
**Pooling**: Used to reduce the size of data by "subsampling". Reduce image resolution.

### LeNet5
**C1**\
Convolutional layer with 6 feature maps of size 28 $$\times$$ 28. Each unit of C1 has a 5 $$\times$$ 5 receptive field in the input layer.\
Shared weights: $$(5 \times 5 + 1) \times 6 = 156$$ parameters to learn.\
Connections: $$28 \times 28 \times (5 \times 5 + 1)\times 6 = 122304$$.\
If it was fully connected we had: $$(32\times 32 + 1) \times (28 \times 28)\times 6 = 4821600$$ parameters.

**S2**\
Subsampling layer with 6 feature maps of size 14 $$\times$$ 14. 2 $$\times$$ 2 nonoverlapping receptive fields in C1.\
Parameters: $$6\times 2=12$$.\
Connections: $$14\times 14 \times (2 \times 2 + 1)\times 6 = 5880$$.

**C3**\
C3 consists of: 6 filters of ddepth 3, 9 filters of depth 4, 1 filter of depth 6. 5 $$\times$$ 5 receptive fields at identical locations in S2.\
Parameters: $$6\times (1 + 5\times 5\times 3) + 9 \times (1 + 5\times 5\times 4) + 1 \times (1 + 5\times 5 \times 6) = 1516$$.

**S4**\
Subsampling layer with 16 feature maps of size 5 $$\times$$ 5. Each unit in S4 is connected to the corresponding 2 $$\times$$ 2 receptive field at C3.\
Parameters: $$16 \times 2 = 32$$.\
Connections: $$5\times 5 \times(2\times 2 + 1)\times 16 = 2000$$.

**C5**\
Convolutional layer with 120 feature maps of size 1 $$\times$$ 1. Each unit in C5 is connected to all 16 5 $$\times$$ 5 receptive fields in S4.\
Parameters and connections: $$120\times (16 \times 5\times 5 + 1) = 48120$$.

**F6**\
84 fully connected units.\
Parameters and connections: $$84\times (120 + 1) = 10164$$.

## Autoencoders and GANs
### Autoencoders
Extract the most concise representation of the input.\
Application: Data compression, dimensionality reduction, visualization, anomaly detection, data generator.

**Denoising autoencoders**: Add noise to inputs and train an autoencoder to denoise.\
**Sparse autoencoders**: Impose some constraints on the bottleneck layer to extract a few features that are really important. Kullback-Leibler divergence is used as part of the loss function.\
**Variational autoencoders**: Extend the bottleneck layer to include the mean value and the standard variance.

### GANs

$$
\begin{align*}
\min_G \max_D V(D, G)
\end{align*}
$$

It is formulated as a minimax game, where:
- The Discriminator is trying to maximize its reward $$V(D, G)$$.
- The Generator is trying to minimize Discriminator's reward (or maximize its loss).

$$
\begin{align*}
V(D, G) = E_{x\sim p(x)}[\log D(x)] + E_{z\sim q(z)}[\log(1 - D(G(z)))]
\end{align*}
$$

The Nash equilibrium of this particular game is achieved at:
- $$P_{data}(x) = P_{gen}(x), \forall x$$.
- $$D(x) = \frac{1}{2}, \forall x$$.

#### Problems with GANs
Probability distribution is implicit:
- Not straightforward to compute $$P(x)$$.
- Thus Vanilla GANs are only good for Sampling / Generation.

Training is hard:
- Non-convergence: For example, differential equation's solution has sinusoidal terms. Even with a small learning rate, it will not converge.
- Mode-collapse: Generator fails to output diverse samples.

Some solutions:
- Mini-batch GANs: Let the Discriminator look at the entire batch instead of single examples. If there is lack of diversity, it will mark the examples as fake.
- Supervision with labels: Label information of the real data might help. Empirically generates much better samples.

## Recurrent Networks
### Backpropagation Through Time
- The output value does depend on the state of the hidden layer, which depends on all previous states of the hidden layer (and thus, all previous inputs).
- Recurrent net can be seen as a (very deep) feedforward net with shared weights.
- Unfold the network over time and use SGD to find the minimum.

Problems:
- The longer the sequence the deeper error propagation.
- Vanishing / exploding gradients.

Therefore, only short (5 - 10) sequences can be modeled this way.

### Long-short Memory Networks (LSTM)
Key ideas:
- Extend the "short memory" hidden layer of a RNN by a mechanism that allows to "learn" which information should be preserved for a longer period and how should it be combined with the current data.
- This "meta-information" should be kept in a "cell state" vector.
- The hidden layer is replaced by a specially designed network.

## Transformers
### Motivation
#### Models for Sequential Data
- Can handle sequences of different lengths as input / output without adjusting parameters.
- These models are versatile - can handle seqence-to-label, label-to-sequence, autoregressive and sequence-to-sequence training.
- Can be causal or non-causal.

#### Drawbacks of RNNs
1. They are inherently sequential and cannot be easily parallelized - need to obtain the previous state of the network before moving on to the next (predict $$t_2$$ requires previous the prediction of $$t_1$$).
2. Long-term dependencies are hard to model (both in terms of gradients and context awareness).
3. Models do not scale well with GPUs - sequential processing.

### Transformer Architecture
**Encoder block**
1. Positional encodings / embeddings.
2. Self attention.
3. Layer normalization: Scaling outputs to have mean of 0 and standard deviation of 1.
4. Residual connections: Initial embedding added to the output of self-attention layer.
5. Simple feed-forward network: Outputs of self attention are processed individually.

**Pros**
- Shallower than RNNs, easier to train.
- Very efficient on GPUs.
- Attention mechanism is key - global context awareness.

**Cons**
- Costly for very long sequences.

### Self Attention Mechanism
Queries (Q), keys (K), and values (V).

$$
\begin{align*}
A(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d}}\right)V
\end{align*}
$$

**Multi-head self attention**
- There are many important semantic relationships to attend to.
- Having multiple heads solves this problem.
- A matter of splitting tensors and increasing their rank.
- The end result of all individual heads is concatenated.

## Reinforcement Learning
### Alpha
**AlphaGo**: Pre-trained on a huge database of historical games and then improved by playing against itself.\
**AlphaGo Zero**: Trained solely on self-played games surpasses AlphaGo.\
**Alpha Zero**: A generic architecture learning to play Go, Chess, Shogi.

#### AlphaGo Zero: Key Ideas
- Trained solely by self-play generated data (no human knowledge).
- Use a single Convolutional ResNet with two "heads" that model policy and value estimates.
    - Policy: Probability distribution over all possible next moves.
    - Value: Probability of winning from the current position.
- Extensive use of Monte Carlo Tree Search to get "better estimates".
- A tournament to select the best network to generate fresh training data.

### Concepts
Reinforcement learning (RL) is a general-purpose framework for artificial intelligence.
- RL is for an **agent** with capacity to act.
- Each **action** influences the agent's future **state**.
- Success is measured by a scalar **reward** signal.

RL in a nutshell: Select **actions** to maximize future **reward**.

**Policy**: $$\pi$$ is a behaviour function selecting actions given states $$a = \pi(s)$$.\
**Value function**: $$Q^{\pi}(s, a)$$ is a expected total reward from state $$s$$ and action $$a$$ under policy $$\pi$$,

$$
\begin{align*}
Q^{\pi}(s, a) = E\left[r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots | s, a \right]
\end{align*}
$$

#### Policy-based RL
- Search directly for the optimal policy $$\pi^*$$.
- This is the policy achieving maximum future reward.

#### Value-based RL
- Estimate the optimal value function $$Q^*(s, a)$$.
- This is the maximum value achievable under any policy.

#### Model-based RL
- Build a transition model of the environment.
- Plan (e.g. by look ahead) using model.

## References
1. Slides of Introduction to Deep Learning course, 2022 Fall, Leiden Univeristy.
2. [Difference between a SVM and a perceptron.](https://www.baeldung.com/cs/svm-vs-perceptron)
3. [What is gradient clipping?](https://towardsdatascience.com/what-is-gradient-clipping-b8e815cdfb48)
4. [What is transfer learning? Exploring the popular deep learning approach.](https://builtin.com/data-science/transfer-learning)
5. [A gentle introduction to transfer learning for deep learning.](https://machinelearningmastery.com/transfer-learning-for-deep-learning/)
6. Bengio, Yoshua. *Deep Learning*. MIT Press, 2016.
7. [Transfer learning on greyscale images: How to fine-tune pretrained models on black-and-white datasets.](https://towardsdatascience.com/transfer-learning-on-greyscale-images-how-to-fine-tune-pretrained-models-on-black-and-white-9a5150755c7a)
8. Geron, Aurelien. *Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems*. 2nd ed., O’Reilly, 2019.
