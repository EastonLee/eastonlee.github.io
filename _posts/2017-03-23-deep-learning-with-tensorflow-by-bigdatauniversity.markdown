---
title: Deep Learning with TensorFlow by BigDataUniversity
layout: post
published: true
category: [Deep Learning, Machine Learning]
---
<script src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

Just a note of this Course, prepared for the coming Quizzes and Final Exam.

<!-- Update: 2017-03-24

The review questions and final exam are not hard at all, I've passed them with full marks. The lab is indeed good for learner to get hand on the Deep Learning process! After this course you will be familiar with TensorFlow and I encourage you to move on and learn higher level framework Keras. -->

* TOC
{:toc}


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

While Linear Regression is suited for estimating continuous values (e.g. estimating house price), it isn’t the best tool for predicting the class of an observed data point. In order to estimate a classification, we need some sort of guidance on what would be the most probable class for that data point. For this, we use Logistic Regression.

Despite the name logistic regression, it is actually a probabilistic classification model. Logistic regression fits a special s-shaped curve by taking the linear regression and transforming the numeric estimate into a probability with the following function:

$$ProbabilityOfaClass=\Theta(y)=\frac{e^y}{1+e^y}=\exp(y)/(1+\exp(y))=p$$

```python
import tensorflow as tf
import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

iris = load_iris()
iris_X, iris_y = iris.data[:-1,:], iris.target[:-1]
iris_y= pd.get_dummies(iris_y).values
trainX, testX, trainY, testY = train_test_split(iris_X, iris_y, test_size=0.33, random_state=42)
```

Why use Placeholders?
1) This feature of TensorFlow allows us to create an algorithm which accepts data and knows something about the shape of the data without knowing the amount of data going in. 

2) When we insert “batches” of data in training, we can easily adjust how many examples we train on in a single step without changing the entire algorithm.

```python
# numFeatures is the number of features in our input data.
# In the iris dataset, this number is '4'.
numFeatures = trainX.shape[1]

# numLabels is the number of classes our data points can be in.
# In the iris dataset, this number is '3'.
numLabels = trainY.shape[1]


# Placeholders
# 'None' means TensorFlow shouldn't expect a fixed number in that dimension
X = tf.placeholder(tf.float32, [None, numFeatures]) # Iris has 4 features, so X is a tensor to hold our data.
yGold = tf.placeholder(tf.float32, [None, numLabels]) # This will be our correct answers matrix for 3 classes.

W = tf.Variable(tf.zeros([4, 3]))  # 4-dimensional input and  3 classes
b = tf.Variable(tf.zeros([3])) # 3-dimensional output [0,0,1],[0,1,0],[1,0,0]
#Randomly sample from a normal distribution with standard deviation .01

weights = tf.Variable(tf.random_normal([numFeatures,numLabels],
                                       mean=0,
                                       stddev=0.01,
                                       name="weights"))

bias = tf.Variable(tf.random_normal([1,numLabels],
                                    mean=0,
                                    stddev=0.01,
                                    name="bias"))
```

$$\hat{y}=sigmoid(WX+b)$$

```python
# Three-component breakdown of the Logistic Regression equation.
# Note that these feed into each other.
apply_weights_OP = tf.matmul(X, weights, name="apply_weights")
add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias") 
activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")

# Training
# Number of Epochs in our training
numEpochs = 700

# Defining our learning rate iterations (decay)
learningRate = tf.train.exponential_decay(learning_rate=0.0008,
                                          global_step= 1,
                                          decay_steps=trainX.shape[0],
                                          decay_rate= 0.95,
                                          staircase=True)

#Defining our cost function - Squared Mean Error
cost_OP = tf.nn.l2_loss(activation_OP-yGold, name="squared_error_cost")

#Defining our Gradient Descent
training_OP = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_OP)
# Create a tensorflow session
sess = tf.Session()

# Initialize our weights and biases variables.
init_OP = tf.global_variables_initializer()

# Initialize all tensorflow variables
sess.run(init_OP)
# argmax(activation_OP, 1) returns the label with the most probability
# argmax(yGold, 1) is the correct label
correct_predictions_OP = tf.equal(tf.argmax(activation_OP,1),tf.argmax(yGold,1))

# If every false prediction is 0 and every true prediction is 1, the average returns us the accuracy
accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))

# Summary op for regression output
activation_summary_OP = tf.summary.histogram("output", activation_OP)

# Summary op for accuracy
accuracy_summary_OP = tf.summary.scalar("accuracy", accuracy_OP)

# Summary op for cost
cost_summary_OP = tf.summary.scalar("cost", cost_OP)

# Summary ops to check how variables (W, b) are updating after each iteration
weightSummary = tf.summary.histogram("weights", weights.eval(session=sess))
biasSummary = tf.summary.histogram("biases", bias.eval(session=sess))

# Merge all summaries
merged = tf.summary.merge([activation_summary_OP, accuracy_summary_OP, cost_summary_OP, weightSummary, biasSummary])

# Summary writer
writer = tf.summary.FileWriter("summary_logs", sess.graph)
# Initialize reporting variables
cost = 0
diff = 1
epoch_values = []
accuracy_values = []
cost_values = []

# Training epochs
for i in range(numEpochs):
    if i > 1 and diff < .0001:
        print("change in cost %g; convergence."%diff)
        break
    else:
        # Run training step
        step = sess.run(training_OP, feed_dict={X: trainX, yGold: trainY})
        # Report occasional stats
        if i % 10 == 0:
            # Add epoch to epoch_values
            epoch_values.append(i)
            # Generate accuracy stats on test data
            train_accuracy, newCost = sess.run([accuracy_OP, cost_OP], feed_dict={X: trainX, yGold: trainY})
            # Add accuracy to live graphing variable
            accuracy_values.append(train_accuracy)
            # Add cost to live graphing variable
            cost_values.append(newCost)
            # Re-assign values for variables
            diff = abs(newCost - cost)
            cost = newCost

            #generate print statements
            print("step %d, training accuracy %g, cost %g, change in cost %g"%(i, train_accuracy, newCost, diff))


# How well do we perform on held-out test data?
print("final accuracy on test set: %s" %str(sess.run(accuracy_OP, 
                                                     feed_dict={X: testX, 
                                                                yGold: testY})))
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.plot([np.mean(cost_values[i-50:i]) for i in range(len(cost_values))])
plt.show()
```

## Activation Functions

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

%matplotlib inline
def plot_act(i=1.0, actfunc=lambda x: x):
    ws = np.arange(-0.5, 0.5, 0.05)
    bs = np.arange(-0.5, 0.5, 0.05)

    X, Y = np.meshgrid(ws, bs)

    os = np.array([actfunc(tf.constant(w*i + b)).eval(session=sess) \
                   for w,b in zip(np.ravel(X), np.ravel(Y))])

    Z = os.reshape(X.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1)

#start a session
sess = tf.Session();
#create a simple input of 3 real values
i = tf.constant([1.0, 2.0, 3.0], shape=[1, 3])
#create a matrix of weights
w = tf.random_normal(shape=[3, 3])
#create a vector of biases
b = tf.random_normal(shape=[1, 3])
#dummy activation function
def func(x): return x
#tf.matmul will multiply the input(i) tensor and the weight(w) tensor then sum the result with the bias(b) tensor.
act = func(tf.matmul(i, w) + b)
#Evaluate the tensor to a numpy array
act.eval(session=sess)
plot_act(1.0, func)
```

### The Step Functions

The Step Function simply functions as a limiter. Every input that goes through this function will be applied to gets either assigned a value of 0 or 1. Tensorflow dosen't have a Step Function.

### The Sigmoid Functions

```python
plot_act(1, tf.sigmoid)
act = tf.sigmoid(tf.matmul(i, w) + b)
act.eval(session=sess)
```

### Tanh

```python
plot_act(1, tf.tanh)
act = tf.tanh(tf.matmul(i, w) + b)
act.eval(session=sess)
```

### The Linear Unit functions

#### Rectified Linear Unit / ReLU

The ReLU is a simple function which operates within the  [0,∞)[0,∞)  interval. For the entirety of the negative value domain, it returns a value of 0, while on the positive value domain, it returns  xx  for any  f(x)f(x) .

During the initialization process of a Neural Network model, in which weights are distributed at random for each unit, ReLUs will only activate approximately only in 50% of the times -- which saves some processing power. Additionally, the ReLU structure takes care of what is called the Vanishing and Exploding Gradient problem by itself. Another benefit -- if not only marginally relevant to us -- is that this kind of activation function is directly relatable to the nervous system analogy of Neural Networks (this is called Biological Plausibility).

```python
plot_act(1, tf.nn.relu)
act = tf.nn.relu(tf.matmul(i, w) + b)
act.eval(session=sess)
```

## Lab

## Graded Review Questions

Full marks!

# Module 2 - Convolutional Networks

## Learning Objectives

In this lesson you will learn about:

* Introduction to Convolutional Networks
* Convolution and Feature Learning
* Convolution with Python and Tensor Flow
* The MNIST Database
* Multilayer Perceptron with Tensor Flow
* Convolutional Network with Tensor Flow

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
