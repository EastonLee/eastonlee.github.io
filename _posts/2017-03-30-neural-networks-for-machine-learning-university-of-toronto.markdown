---
title: Neural Networks for Machine Learning University of Toronto
layout: post
published: true
last_modified_at: 2017-07-16
category: [Neural Network, Machine Learning, Course Notes]
---

Update on 7/16: This course is damn hard! And poor organized somewhere, for example, in week 13 one video is missing and the last two questions in the quiz are wrong themselves. But all in all it's still a course in depth.

This is a note for Course: Neural Networks for Machine Learning University of Toronto

I found Prof. Geoffrey Hinton's British English was a little hard for me to understand, but he definitely has the insight of neural network, the content is really of high quality and helped me a lot to understand neural network thoroughly.

<!--more-->

* TOC
{:toc}

# Week 1: Introduction

## Why do we need machine learning?

## What are neural networks?

## Some simple models of neurons

Linear, Binary Threshold, Logistic Sigmoid, Rectified Linear, Stochastic Binary

## A simple example of learning

## Three types of learning

* Supervised Learning
    * Regression
    * Classification
* Unsupervised Learning
    * It provides a compact, low-dimensional representation of the input
    * It provides an economical high-dimensional representation of the input in terms of learned features
    * It finds sensible clusters in the input
* Reinforced Learning

# Week 2: The Perceptron learning procedure

## Types of neural network architectures

* Feed-forward neural networks
* Recurrent networks
* Symmetrically connected networks

## Perceptrons: The first generation of neural networks

In machine learning, the perceptron is an algorithm for supervised learning of binary classifiers

## A geometrical view of perceptrons

## Why the learning works

## What perceptrons can't do

# Week 3: The backpropagation learning proccedure

## Learning the weights of a linear neuron

## The error surface for a linear neuron

## Learning the weights of a logistic output neuron

## The backpropagation algorithm

## Using the derivatives computed by backpropagation

A large number of different methods have been developed. –  Weight-decay

*  Weight-sharing
*  Early stopping
*  Model averaging
*  Bayesian fitting of neural nets –  Dropout
*  Generative pre-training

Linear hidden units don't add modeling capacity to the network.

# Week 4: Learning feature vectors for words

## Learning to predict the next word

One Hot

Why One Hot?


## A brief diversion into cognitive science

There has been a long debate in cognitive science between two rival theories of what it means to have a concept:

* The feature theory: A concept is a set of semantic features.
    * This is good for explaining similarities between concepts. –  Its convenient: a concept is a vector of feature activities.
* The structuralist theory: The meaning of a concept lies in its relationships to other concepts.
    * So conceptual knowledge is best expressed as a relational graph.
    * Minsky used the limitations of perceptrons as evidence against feature vectors and in favor of relational graph representations.

* These two theories need not be rivals. A neural net can use vectors of semantic features to implement a relational graph.
    * In the neural network that learns family trees, no explicit inference is required to arrive at the intuitively obvious consequences of the facts that have been explicitly learned.
    * The net can “intuit” the answer in a forward pass.
* We may use explicit rules for conscious, deliberate reasoning, but we do a lot of commonsense, analogical reasoning by just “seeing” the answer with no conscious intervening steps.
    * Even when we are using explicit rules, we need to just see which rules to apply.

## Another diversion: The softmax output function

Using squared error as logstic's cost function may not be a good idea, because the derivative is likely very near to zero, resulting into very slow learning.

Instead, we can use Softmax, its cost function is $$E=-t\log(y)-(1-t)\log(1-y)$$, also called cross-entropy.

## Neuro-probabilistic language models

## Ways to deal with the large number of possible outputs


# Week 5: Object recognition with neural nets.

## Why object recognition is difficult

## Achieving viewpoint invariance

Several different approaches to achieve viewpoint invariance:
* Use redundant invariant features
* Put a box around the object and use normalized pixels
* Use replicated features with pooling. This is called "convolutional neural network"
* Use hierarchy of parts of that have explicit poses relative to the camera

## Convolutional nets for digit recognition
## Convolutional nets for object recognition
Graded: Lecture 5 Quiz
Graded: Programming Assignment 2: Learning Word Representations.

# Week 6: Optimization: How to make the learning go faster

I find nothing worth taking notes of, many overlapping content with the course by Stanford.

## Overview of mini-batch gradient descent

## A bag of tricks for mini-batch gradient descent

## The momentum method

## Adaptive learning rates for each connection

## Rmsprop: Divide the gradient by a running average of its recent magnitude


# Week 7: Recurrent neural networks

## Modeling sequences: A brief overview

Linear dynamic systems and hidden Markov models are stochastic models, Recurrent neural networks are deterministic.

## Training RNNs with back propagation

## A toy example of training an RNN

A recurrent network can emulate a finite state automaton, but it is exponentially more powerful. With N hidden neurons it has 2^N possible binary activity vectors (but only N^2 weights).

## Why it is difficult to train an RNN

Four effective ways to learn an RNN

* Long Short Term Memory
* Hessian Free Optimization
* Echo State Network
* Good initialization with momentum

## Long-term Short-term-memory

# Week 8: More recurrent neural networks

## Video: Modeling character strings with multiplicative connections

## Video: Learning to predict the next character using HF

## Video: Echo State Networks

# Week 9

**Preventing overfitting**

* Approach 1: Get more data

    Almost always the best bet if you have enough compute power to train on more data

* Approach 2: Use a model that has the right capability

    * enough to fit the right regularity
    * not enough to fit spurious regularities (if they are weaker)

* Approach 3: Average many different models

    * Use models with different forms
    * Or train the model with different subsets of training data (this is called "bagging")

* Approach 4: (Bayesian) Use a single neural network architecture, but average different prediction made by many different weight vectors.

**The capability can be controlled by many ways**

* Architecture: Limit the number of hidden layers and the number of units per layer
* Early Stopping: Start with small weights and stop the learning before it overfits
* Weight decay: Penalize large weights using penalties or constrains on the their squared values (L2 penalty) or absolute values (L1 penalty)
* Noise: Add noise to the weights or the activities

Typically a combination of these methods is used.

**Cross-validation: a better way to choose meta parameters**

Divide the total dataset into three subsets:

* Training data: is used for learning the parameters of the model.
* Validation data: is not used for learning but is used to decide what settings of the meta parameters work best.
* Test data: is used to get a final, unbiased estimate of how well the network works. We expect this estimate to be worse than on the validation data.

**N-fold cross-validation** (Easton's note: This definition of N-fold cross-validation is different from elsewhere)

We could divide the total dataset into one final test set and N other subset and train on all but one of the subsets to get N different estimate of the validation error rate.

**Noise can be used as regularizer against overfit in input, output and activating functions**

# Week 10

**Making models differ by changing their training data**

* Bagging: Train different models on different subsets of the data
* Boosting: Train a sequence of low capability models. Weight the training cases differently for each model in the sequence.

# Week 11

Hopfield nets and Boltzmann machines
 
5 videos, 1 reading
Reading: Lecture Slides (and resources)
Video: Hopfield Nets
Video: Dealing with spurious minima
Video: Hopfield nets with hidden units
Video: Using stochastic units to improv search
Video: How a Boltzmann machine models data
Graded: Lecture 11 Quiz

# Week 12

Restricted Boltzmann machines (RBMs)
This module deals with Boltzmann machine learning  
5 videos, 1 reading
Reading: Lecture Slides (and resources)
Video: Boltzmann machine learning
Video: OPTIONAL VIDEO: More efficient ways to get the statistics
Video: Restricted Boltzmann Machines
Video: An example of RBM learning
Video: RBMs for collaborative filtering

# Week 13

Stacking RBMs to make Deep Belief Nets
 
3 videos, 1 reading
Reading: Lecture Slides (and resources)
Video: The ups and downs of back propagation
Video: Belief Nets
Video: The wake-sleep algorithm
Graded: Programming Assignment 4: Restricted Boltzmann Machines
Graded: Lecture 13 Quiz

# Week 14

Deep neural nets with generative pre-training
 
5 videos, 1 reading
Reading: Lecture Slides (and resources)
Video: Learning layers of features by stacking RBMs
Video: Discriminative learning for DBNs
Video: What happens during discriminative fine-tuning?
Video: Modeling real-valued data with an RBM
Video: OPTIONAL VIDEO: RBMs are infinite sigmoid belief nets

# Week 15

Modeling hierarchical structure with neural nets
 
6 videos, 1 reading
Reading: Lecture Slides (and resources)
Video: From PCA to autoencoders
Video: Deep auto encoders
Video: Deep auto encoders for document retrieval
Video: Semantic Hashing
Video: Learning binary codes for image retrieval
Video: Shallow autoencoders for pre-training

# Week 16

Recent applications of deep neural nets
 
3 videos
Video: OPTIONAL: Learning a joint model of images and captions
Video: OPTIONAL: Hierarchical Coordinate Frames
Video: OPTIONAL: Bayesian optimization of hyper-parameters


<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>