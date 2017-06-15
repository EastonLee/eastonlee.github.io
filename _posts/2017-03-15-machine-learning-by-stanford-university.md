---
title: Machine Learning by Stanford University
layout: post
category: [Machine Learning, Course Notes]
---

This course is one of the most famous courses on Coursera. Now I go two weeks ahead of the deadline and reach Week 5, I plan to finish it in the flowing few days. 

<!--more-->
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

Update 03-19:
I finished the course with full marks today, but this post is still incomplete, I will keep updating it as reviewing this great course.

This course is the perfect choice if you are not satisfied with just being able to drive some machine learning framework to work but also eager to know what is under the hood, this course will teach you the most concrete mathematical principles and equations underlying most AI applications. Overall in this course, Prof. Ng delivered profound knowledge in a comprehensive way. But this course isn't flawless, for example Week 5 uses intuition to explain backpropogation and example applications, which I would say verbose and useless. 

Bellow is my note of important concept, it may be incomplete and biased, feel free to leave comment and let me know, I will keep it updated.

The Syllabus skeleton is left, to remind readers in which section that concept is taught.

* TOC
{:toc}

# Week 1: Introduction

Supervised Learning and Unsupervised Learning

# Week 2:

## Multivariate Linear Regression

### Multiple Features

### Gradient Descent For Multiple Variables

### Gradient Descent in Practice I - Feature Scaling

### Gradient Descent in Practice II - Learning Rate

### Features and Polynomial Regression

## Computing Parameters Analytically

### Normal Equation

### Normal Equation Noninvertibility


# Week 3: Logistic Regression

Questions:

1. Is the gradient too small?

2. Why logistic regression has advantage over linear regression when it comes to classification.

http://www.theanalysisfactor.com/why-logistic-regression-for-binary-response/

## Classification and Representation

### Classification

### Hypothesis Representation

$$\theta(x)=g(\theta^Tx)$$
$$z=\theta^Tx$$
$$g(z)=\frac{1}{1+e^{-z}}$$

### Decision Boundary

TODO
convex function

## Logistic Regression Model

### Cost Function

$$J(\theta)=-\frac{1}{m}[\sum_{i=0}^{m}y^{(i)}\log h_\theta(x^{(i)})+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))] + \frac{\lambda}{2m}\sum_{j=1}^{n}\theta_j^2 $$

$$J(\Theta)=−\frac{1}{m}\sum_{i=1}^m\sum_{k=1}^K[y^{(i)}k\log((h_\Theta(x(i)))_k)+(1−y_ k^{(i)})\log(1−(h_Theta(x^{(i)}))_k)]+\frac{λ}{2m}\sum_{l=1}^{L−1}\sum_{i=1}^{sl}\sum_{j=1}^{sl+1}(\Theta^{(l)}_{j,i})^2$$

# Week 4: 

### Simplified Cost Function and Gradient Descent

### Advanced Optimization

fminunc in Octave is very useful to auto generate cost and gradient

## Multiclass Classification

### Multiclass Classification: one-vs-all

## Solving the Problem of Overfitting

### The Problem of Overfitting

### Cost function

### Regularized Linear Regression

### Regularized Logistic Regression

# Week 4

## Non-linear Hypotheses

## Neurons and Brains

## Neural Networks

### Model Representation

#### How to determine the dimension of one layer?
If network has $$s_j$$ units in layer j and $$s_j+1$$ units in layer j+1, then $$\theta(j)$$ will be of dimension $$s_{j+1}(s_j+1)$$.

#### Forward Propagation

## Applications

### Examples and Intuitions 

# Week 5: Neural Networks: Learning

## Cost Function and Backpropagation

### Cost Function

### Backpropagation Algorithm

Error(delta) of cost for Node

### Backpropagation Intuition

## Backpropagation in Practice

### Implementation Note: Unrolling Parameters

### Gradient Checking

gradApprox ≈ deltaVector

The code to compute gradApprox can be very slow

### Random Initialization 

Initialization theta can't be set all to 0, otherwise the backpropagation will get all same theta. So theta matrix should be initialize randomly. This is also called Symmetry Breaking.

One effective strategy for choosing $$\epsilon_{init}$$ is to base it on the number of units in the network. A good choice of $$\epsilon_{init}$$ is $$\epsilon_{init} = \frac{\sqrt6}{\sqrt{L_{in}+L_{out}}}$$ , where $$L_{in} = s_l$$ and $$L_{out} = s_l+1$$ are the number of units in the layers adjacent to $$\Theta^{(l)}$$.

### Putting It Together

Question: Can we just skip gradient checking?
A: No, we need to check the backpropogation is bug free.

First, pick a network architecture; choose the layout of your neural network, including how many hidden units in each layer and how many layers in total you want to have.

Number of input units = dimension of features x(i)

Number of output units = number of classes

Number of hidden units per layer = usually more the better (must balance with cost of computation as it increases with more hidden units)

Defaults: 1 hidden layer. If you have more than 1 hidden layer, then it is recommended that you have the same number of units in every hidden layer.

**Training a Neural Network**

1. Randomly initialize the weights
2. Implement forward propagation to get hΘ(x(i)) for any x(i)
3. Implement the cost function
4. Implement backpropagation to compute partial derivatives
5. Use gradient checking to confirm that your backpropagation works. Then disable gradient checking.
6. Use gradient descent or a built-in optimization function to minimize the cost function with the weights in theta.

## Application of Neural Networks

### Autonomous Driving

### Programming Assignment

Visualizing the hidden layer

One way to understand what your neural network is learning is to visualize what the representations captured by the hidden units.

# Week 6: 

## Advice for Applying Machine Learning

## Evaluating a Learning Algorithm

### Deciding What to Try Next

### Evaluating a Hypothesis

### Model Selection and Train/Validation/Test Sets

## Bias vs. Variance

### Diagnosing Bias vs. Variance

### Regularization and Bias/Variance

In order to choose the model and the regularization term λ, we need to:

Create a list of lambdas (i.e. λ∈{0,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28,2.56,5.12,10.24});
Create a set of models with different degrees or any other variants.
Iterate through the λs and for each λ go through all the models to learn some Θ.
Compute the cross validation error using the learned Θ (computed with λ) on the JCV(Θ) without regularization or λ = 0.
Select the best combo that produces the lowest error on the cross validation set.
Using the best combo Θ and λ, apply it on Jtest(Θ) to see if it has a good generalization of the problem.

### Learning Curves

### Deciding What to do Next Revisited

## Review

### Quiz: Advice for Applying Machine Learning5 questions

### Programming Assignment: Regularized Linear Regression and Bias/Variance3h

## Machine Learning System Design

## Building a Spam Classifier

### Prioritizing What to Work On

### Error Analysis

Accuracy = (true positives + true negatives) / (total examples)

Precision = (true positives) / (true positives + false positives)

Recall = (true positives) / (true positives + false negatives)

F1 score = (2 * precision * recall) / (precision + recall)

## Handling Skewed Data

### Error Metrics for Skewed Classes

### Trading Off Precision and Recall

### Using Large Data Sets

### Data For Machine Learning

## Review

### Quiz: Machine Learning System Design5 questions

TODO

# Week 7: Support Vector Machines

Question: What is SVM for?


## Large Margin Classification

### Optimization Objective

### Large Margin Intuition

### Mathematics Behind Large Margin Classification

## Kernels

kernel refers to similarity function.

### SVMs in Practice

### Using An SVM

Do not perform feature scaling before using the Gaussian kernel.

Gaussian kernel, linear kernel.

## Review

### Quiz: Support Vector Machines5 questions

### Programming Assignment: Support Vector Machines

# Week 8

## Unsupervised Learning

### Clustering

#### Unsupervised Learning: Introduction

#### K-Means Algorithm

#### Optimization Objective

#### Random Initialization

#### Choosing the Number of Clusters

## Dimensionality Reduction

### Motivation

#### Motivation I: Data Compression

#### Motivation II: Visualization

### Principal Component Analysis

#### Principal Component Analysis Problem Formulation

Preprocess is needed: Feature scaling and mean normalization

#### Principal Component Analysis Algorithm

### Applying PCA

#### Reconstruction from Compressed Representation

#### Choosing the Number of Principal Components

#### Advice for Applying PCA

### Review

#### Programming Assignment: K-Means Clustering and PCA

I'm excited about this exercise, about how images' pixels or other high dimension can be reduced to low dimension, I was shocked when the "eigenfaces" was drawn, look how well it did give the outlines of faces.

# Week 9

# Anomaly Detection

## Density Estimation

If $$p(x_i) < \epsilon$$, we say $$x_i$$ is anomalous. We use Gaussian Distribution to calculate $$p(x_i)$$.

### Problem Motivation

### Gaussian Distribution

$$p(x;\mu, \sigma)=\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}

### Algorithm

## Building an Anomaly Detection System

### Developing and Evaluating an Anomaly Detection System

### Anomaly Detection vs. Supervised Learning

What's difference between Anomaly Detection and Supervised Learning?

|                                                                                                 Anomaly detection                                                                                                 |                                                                       Supervised learning                                                                        |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Very small number of positive examples (y=1). (0-20 is common)                                                                                                                                                    | Large number of positive and negative examples.                                                                                                                  |
| Large number of negative (y=0) examples                                                                                                                                                                           |                                                                                                                                                                  |
| Many different "types" of anomalies. Hard for any algorithm to learn from positive examples what the anomalies look like; future anomalies may look nothing like any of the anomalous examples we've seen so far. | Enough positive examples for algorithm to get a sense of what positive examples are like, future positive examples likely to be similar to ones in training set. |
| Fraud detection                                                                                                                                                                                                   | Email spam classification                                                                                                                                        |
| Manufacturing (e.g. aircraft engines)                                                                                                                                                                             | Weather prediction (sunny/rainy/etc)                                                                                                                             |
| Monitoring machines in a data center                                                                                                                                                                              | Cancer classification                                                                                                                                            |

### Choosing What Features to Use

Choose features that might take on unusually large or small values in the event of an anomaly.

If features are not normally distributed, use 1/2 power or log function to normalize them.

## Multivariate Gaussian Distribution (Optional)

### Multivariate Gaussian Distribution

### Anomaly Detection using the Multivariate Gaussian Distribution

$$p(x;\mu ,\Sigma )=\frac { 1 }{ 2\pi ^{ n/2 }\vert \Sigma \vert ^{ 1/2 } } exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))$$

Flag an anomaly if $$p(x)<\epsilon$$

where $$\mu=\frac{1}{m}\sum_{i=1}^{m}x^{(i)}$$

$$\Sigma=\frac{1}{m}\sum_{i=1}^{m}(x^{(i)}-\mu)(x^{(i)}-\mu)^T$$

Origin model is like this:

$$p(x)=p(x_1,;\mu_1,\sigma_1^2)\times p(x_2,;\mu_1,\sigma_1^2)\times\cdots \times p(x_n,;\mu_n,\sigma_n^2)$$

|                                         Original model                                         |                Multivariate Gaussian                |
|------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| Manually create features to capture anomalies where x1, x2 take unusual combinations of values | Automatically capture correlations between features |
| Computationally cheaper (alternatively, scales betters to large n)                             | Computationally more expensive                      |
| OK even if m(training set size) is small                                                       | Must have m>n, or else $$\Sigma$$ is invertible     |
|                                                                                                |                                                     |

## Recommender Systems

### Predicting Movie Ratings

This section assumes we have movies features, then we train parameters for every user.

### Problem Formulation

### Content Based Recommendations

## Collaborative Filtering

### Collaborative Filtering

This section talks about feature learning, assuming we don't have movies' features yet. But once some users rate one unfeatured movie, we can calculate the movie's feature which makes the cost function minimum.

### Collaborative Filtering Algorithm

x and theta matrix should both be initialized randomly to break symmetry.

### Low Rank Matrix Factorization

### Vectorization: Low Rank Matrix Factorization

### Implementational Detail: Mean Normalization

For new users who haven't any rating, thus haven't any theta, you could assign the average rating and theta to them.

## Review

### Programming Assignment: Anomaly Detection and Recommender Systems

This assignment is not so interesting, I recommend you another Machine Learning specialization by Washington University on Cousera, I think they teach Recommending System better.

# Week 10: Large Scale Machine Learning

## Gradient Descent with Large Datasets

Observe learning curves over training set and cross validation set, if they converge and reach to the same level, that means your training set is large enough.

### Learning With Large Datasets

Batch gradient descent is more suitable for large dataset.

### Stochastic Gradient Descent

Stochastic gradient descent:

1. randomly shuffle training examples
2. use single training example to update theta
3. repeat step 2

Stochastic gradient descent is much faster than batch gradient descent, but is not guaranteed to reach optimum eventually. I think it's hard for stochastic gradient descent to pick a proper learning rate.

### Mini-Batch Gradient Descent

A compromise of batch and stochastic gradient descent.

### Stochastic Gradient Descent Convergence

You can decrease learning rate to guarantee the cost function converge.

## Advanced Topics

### Online Learning

Similar to stochastic gradient descent, every time you use new training set to adjust your model, and only for once. The method can adapt your model when user preference changes.

### Map Reduce and Data Parallelism

Parallel computing gradient on batch, then sum up on central node and update your sigma. This way also speeds up the learning process and enable you to deal with large scale dataset.

# Week 11: Application Example: Photo OCR

## Photo OCR

### Problem Description and Pipeline

### Sliding Windows

### Getting Lots of Data and Artificial Data

### Ceiling Analysis: What Part of the Pipeline to Work on Next

Analyze every component, assume that component and the components before it are perfect, calculate the accuracy of the whole pipeline, so you can find improvement space at every component, if the improvement is little, then it's not very worth it to improve it.