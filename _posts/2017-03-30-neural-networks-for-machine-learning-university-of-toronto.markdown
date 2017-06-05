---
title: Neural Networks for Machine Learning University of Toronto
layout: post
published: false
category: [Neural Network, Machine Learning, Course Notes]
---

This is a note for Course: Neural Networks for Machine Learning University of Toronto

I found Prof. Geoffrey Hinton's British English was a little hard for me to understand, but the content is really of high quality and definitely helped me understand neural network thoroughly.

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


# Week 5

Object recognition with neural nets. In this module we look at why object recognition is difficult.  

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

# Week 6

Optimization: How to make the learning go faster
We delve into mini-batch gradient descent as well as discuss adaptive learning rates. 
5 videos, 1 reading
Reading: Lecture Slides (and resources)
Video: Overview of mini-batch gradient descent
Video: A bag of tricks for mini-batch gradient descent
Video: The momentum method
Video: Adaptive learning rates for each connection
Video: Rmsprop: Divide the gradient by a running average of its recent magnitude
Graded: Lecture 6 Quiz

# Week 7

Recurrent neural networks
This module explores training recurrent neural networks 
5 videos, 1 reading
Reading: Lecture Slides (and resources)
Video: Modeling sequences: A brief overview
Video: Training RNNs with back propagation
Video: A toy example of training an RNN
Video: Why it is difficult to train an RNN
Video: Long-term Short-term-memory

# Week 8

More recurrent neural networks
We continue our look at recurrent neural networks 
3 videos, 1 reading
Reading: Lecture Slides (and resources)
Video: Modeling character strings with multiplicative connections
Video: Learning to predict the next character using HF
Video: Echo State Networks

# Week 9

Ways to make neural networks generalize better
We discuss strategies to make neural networks generalize better 
6 videos, 1 reading
Reading: Lecture Slides (and resources)
Video: Overview of ways to improve generalization
Video: Limiting the size of the weights
Video: Using noise as a regularizer
Video: Introduction to the full Bayesian approach
Video: The Bayesian interpretation of weight decay
Video: MacKay's quick and dirty method of setting weight costs
Graded: Lecture 9 Quiz
Graded: Programming assignment 3: Optimization and generalization

# Week 10

Combining multiple neural networks to improve generalization
This module we look at why it helps to combine multiple neural networks to improve generalization 
5 videos, 1 reading
Reading: Lecture Slides (and resources)
Video: Why it helps to combine models
Video: Mixtures of Experts
Video: The idea of full Bayesian learning
Video: Making full Bayesian learning practical
Video: Dropout
Graded: Lecture 10 Quiz

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