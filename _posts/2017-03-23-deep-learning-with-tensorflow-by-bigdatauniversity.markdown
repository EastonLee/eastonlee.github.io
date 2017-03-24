---
title: Deep Learning with TensorFlow by BigDataUniversity
layout: post
published: false
category: [Deep Learning, Machine Learning]
---

# Grading Scheme

1. The minimum passing mark for the course is 70% with the following weights:

    * 50% - All Review Questions 

    * 50% - The Final Exam

2. Though Review Questions and the Final Exam have a passing mark of 60% respectively, the only grade that matters is the overall grade for the course.

3. Review Questions have no time limit. You are encouraged to review the course material to find the answers.  Please remember that the Review Questions are worth 50% of your final mark.

4. The final exam has a 1 hour time limit.

5. Attempts are per question in both, the Review Questions and the Final Exam:

    * One attempt - For True/False questions

    * Two attempts - For any question other than True/False

6. There are no penalties for incorrect attempts.

7. Clicking the "Final Check" button when it appears, means your submission is FINAL.  You will NOT be able to resubmit your answer for that question ever again.

8. Check your grades in the course at any time by clicking on the "Progress" tab.

# Module 1 - Introduction to TensorFlow

## Learning Objectives current section

In this lesson you will learn about:

* Introduction to TensorFlow
* Linear, Nonlinear and Logistic Regression with Tensorflow
* Activation Functions


## Introduction to TensorFlow

## TensorFlow's Hello World

Tensor (Matrix), Node (Operation), Session
```python
import tensorflow as tf # import
a = tf.constant([2]) # Tensor
b = tf.constant([3])
c = tf.add(a, b) # Node
session = tf.Session() # Session

result = session.run(c) # Execution / Evaluation
print(result)

session.close() # release resource

# with block will release resource automatically after end of block
with tf.Session() as session:
    result = tf.run(session)
    print(result)

```

## Tensors, Variables and Placeholders

```python
Scalar = tf.constant([2])
Vector = tf.constant([5,6,2])
Matrix = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
Tensor = tf.constant( [ [[1,2,3],[2,3,4],[3,4,5]] , [[4,5,6],[5,6,7],[6,7,8]] , [[7,8,9],[8,9,10],[9,10,11]] ] )
with tf.Session() as session:
    result = session.run(Scalar)
    print "Scalar (1 entry):\n %s \n" % result
    result = session.run(Vector)
    print "Vector (3 entries) :\n %s \n" % result
    result = session.run(Matrix)
    print "Matrix (3x3 entries):\n %s \n" % result
    result = session.run(Tensor)
    print "Tensor (3x3x3 entries) :\n %s \n" % result

Matrix_one = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
Matrix_two = tf.constant([[2,2,2],[2,2,2],[2,2,2]])

# two ways of addition
first_operation = tf.add(Matrix_one, Matrix_two)
second_operation = Matrix_one + Matrix_two

with tf.Session() as session:
    result = session.run(first_operation)
    print "Defined using tensorflow function :"
    print(result)
    result = session.run(second_operation)
    print "Defined using normal expressions :"
    print(result)
    # the results are the same

# tf.matmul
Matrix_one = tf.constant([[2,3],[3,4]])
Matrix_two = tf.constant([[2,3],[3,4]])

first_operation = tf.matmul(Matrix_one, Matrix_two)

with tf.Session() as session:
    result = session.run(first_operation)
    print "Defined using tensorflow function :"
    print(result)

# variable
state = tf.Variable(0)
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)
# Variables must be initialized by running an initialization operation after having launched the graph. We first have to add the initialization operation to the graph:
init_op = tf.global_variables_initializer()
with tf.Session() as session:
  session.run(init_op)
  print(session.run(state))
  for _ in range(3):
    session.run(update)
    print(session.run(state))

# placeholder
a=tf.placeholder(tf.float32)
b=a*2
with tf.Session() as sess:
    result = sess.run(b,feed_dict={a:3.5})
    print result

dictionary={a: [ [ [1,2,3],[4,5,6],[7,8,9],[10,11,12] ] , [ [13,14,15],[16,17,18],[19,20,21],[22,23,24] ] ] }

with tf.Session() as sess:
    result = sess.run(b,feed_dict=dictionary)
    print result

# operations
a = tf.constant([5])
b = tf.constant([2])
c = tf.add(a,b)
d = tf.subtract(a,b)

with tf.Session() as session:
    result = session.run(c)
    print 'c =: %s' % result
    result = session.run(d)
    print 'd =: %s' % result
```

## Linear Regression with Tensor Flow

When more than one independent variable is present the process is called multiple linear regression. When multiple dependent variables are predicted the process is known as multivariate linear regression.
```python
import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = (10, 6)

X = np.arange(0.0, 5.0, 0.1)
X

##You can adjust the slope and intercept to verify the changes in the graph
a=1
b=0

Y= a*X + b 

plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 3 + 2
y_data = np.vectorize(lambda y: y + np.random.normal(loc=0.0, scale=0.1))(y_data)

a = tf.Variable(1.0)
b = tf.Variable(0.2)
y = a * x_data + b
loss = tf.reduce_mean(tf.square(y - y_data)) # This function finds the mean of a multidimensional tensor, and the result can have a diferent dimension.
optimizer = tf.train.GradientDescentOptimizer(0.5) # 0.5 is the learning rate
train = optimizer.minimize(loss)
train_data = []
for step in range(100):
    evals = sess.run([train,a,b])[1:]
    if step % 5 == 0:
        print(step, evals)
        train_data.append(evals)

converter = plt.colors
cr, cg, cb = (1.0, 1.0, 0.0)
for f in train_data:
    cb += 1.0 / len(train_data)
    cg -= 1.0 / len(train_data)
    if cb > 1.0: cb = 1.0
    if cg < 0.0: cg = 0.0
    [a, b] = f
    f_y = np.vectorize(lambda x: a*x + b)(x_data)
    line = plt.plot(x_data, f_y)
    plt.setp(line, color=(cr,cg,cb))

plt.plot(x_data, y_data, 'ro')


green_line = mpatches.Patch(color='red', label='Data Points')

plt.legend(handles=[green_line])

plt.show()

```

## Logistic Regression

## Activation Functions

## Lab

## Graded Review Questions

## Review Questions This content is graded

# Module 2 - Convolutional Networks

## Learning Objectives

## Introduction to Convolutional Networks

## Convolution and Feature Learning

## Convolution with Python and Tensor Flow

## The MNIST Database

## Multilayer Perceptron with Tensor Flow

## Convolutional Network with Tensor Flow

## Lab

## Graded Review Questions

## Review Questions This content is graded

# Module 3 - Recurrent Neural Network

## Learning Objectives

## The Sequential Problem

## The Recurrent Neural Network Model

## The Long Short-Term Memory Model

## Recursive Neural Tensor Networks

## Applying Recurrent Networks to Language Modelling

## Lab

## Graded Review Questions

## Review Questions This content is graded

# Module 4 - Unsupervised Learning

## Learning Objectives

## Introduction to Unsupervised Learning

## RBMs and Autoencoders

## Initializing a Restricted Boltzmann Machine

## Training a Restricted Bolztmann Machine

## Recommendation System with a Restrictive Boltzmann Machine

## Lab

## Graded Review Questions

## Review Questions This content is graded

# Module 5 - Autoencoders

## Learning Objectives

## Introduction to Autoencoders

## Autoencoder Structure

## Autoencoders with Tensor Flow

## Deep Belief Networks

## Lab

## Graded Review Questions

## Review Questions This content is graded

## Course Summary

## Course Summary

## Appendix

## Resources and Materials

## Final Exam

## Instructions

## Final Exam

## Timed Exam

## Completion Certificate

## Completion Certificate
