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

Update: 2017-03-24

I think the content of this post is not proper for publishing, because most of the content is not written by me, I just wrap them up.

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

The ReLU is a simple function which operates within the  [0,∞)[0,∞)  interval. For the entirety of the negative value domain, it returns a value of 0, while on the positive value domain, it returns x for any  f(x).

During the initialization process of a Neural Network model, in which weights are distributed at random for each unit, ReLUs will only activate approximately only in 50% of the times -- which saves some processing power. Additionally, the ReLU structure takes care of what is called the Vanishing and Exploding Gradient problem by itself. Another benefit -- if not only marginally relevant to us -- is that this kind of activation function is directly relatable to the nervous system analogy of Neural Networks (this is called Biological Plausibility).

```python
plot_act(1, tf.nn.relu)
act = tf.nn.relu(tf.matmul(i, w) + b)
act.eval(session=sess)
```

#### Softplus

$$f(x)=\ln(1+e^{x})$$

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

### Convolution: 1D operation with Python (Numpy/Scipy)

```python
import numpy as np

h = [2,1,0]
x = [3,4,5]
 

y = np.convolve(x,h)
y  
```

There are three methods to apply kernel on the matrix, with padding (full), with padding(same) and without padding(valid)


```python
import numpy as np

x= [6,2]
h= [1,2,5,4]

y= np.convolve(x,h,"full")  #now, because of the zero padding, the final dimension of the array is bigger
y= np.convolve(x,h,"same")  #it is same as zero padding, but withgenerates same 
y= np.convolve(x,h,"valid")  #we will understand why we used the argument valid in the next example
```

### Convolution: 2D operation with Python (Numpy/Scipy)

```python
from scipy import signal as sg

I= [[255,   7,  3],
    [212, 240,  4],
    [218, 216, 230],]

g= [[-1,1]]

print ('Without zero padding \n')
print ('{0} \n'.format(sg.convolve( I, g, 'valid')))
# The 'valid' argument states that the output consists only of those elements 
# that do not rely on the zero-padding.

print ('With zero padding \n')
print sg.convolve( I, g)
```

### Coding with TensorFlow

```python
import tensorflow as tf

#Building graph

input = tf.Variable(tf.random_normal([1,10,10,1]))
filter = tf.Variable(tf.random_normal([3,3,1,1]))
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
op2 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

#Initialization and session
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    print("Input \n")
    print('{0} \n'.format(input.eval()))
    print("Filter/Kernel \n")
    print('{0} \n'.format(filter.eval()))
    print("Result/Feature Map with valid positions \n")
    result = sess.run(op)
    print(result)
    print('\n')
    print("Result/Feature Map with padding \n")
    result2 = sess.run(op2)
    print(result2)
```

### Convolution applied on images

```python
# download standard image
!wget --quiet https://ibm.box.com/shared/static/cn7yt7z10j8rx6um1v9seagpgmzzxnlz.jpg --output-document bird.jpg

#Importing
import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image


### Load image of your choice on the notebook
print("Please type the name of your test image after uploading to \
your notebook (just drag and grop for upload. Please remember to \
type the extension of the file. Default: bird.jpg")

#raw= raw_input()

im = Image.open('bird.jpg')  # type here your image's name

# uses the ITU-R 601-2 Luma transform (there are several 
# ways to convert an image to grey scale)

image_gr = im.convert("L")    
print("\n Original type: %r \n\n" % image_gr)

# convert image to a matrix with values from 0 to 255 (uint8) 
arr = np.asarray(image_gr) 
print("After conversion to numerical representation: \n\n %r" % arr) 
### Activating matplotlib for Ipython
%matplotlib inline

### Plot image

imgplot = plt.imshow(arr)
imgplot.set_cmap('gray')  #you can experiment different colormaps (Greys,winter,autumn)
print("\n Input image converted to gray scale: \n")
plt.show(imgplot)

kernel = np.array([
                        [ 0, 1, 0],
                        [ 1,-4, 1],
                        [ 0, 1, 0],
                                     ]) 

grad = signal.convolve2d(arr, kernel, mode='same', boundary='symm')
%matplotlib inline

print('GRADIENT MAGNITUDE - Feature map')

fig, aux = plt.subplots(figsize=(10, 10))
aux.imshow(np.absolute(grad), cmap='gray')

type(grad)

grad_biases = np.absolute(grad) + 100

grad_biases[grad_biases > 255] = 255
%matplotlib inline

print('GRADIENT MAGNITUDE - Feature map')

fig, aux = plt.subplots(figsize=(10, 10))
aux.imshow(np.absolute(grad_biases), cmap='gray')
```

## The MNIST Database

### 1st part: classify MNIST using a simple model

You have two basic options when using TensorFlow to run your code:
* Build graphs and run session Do all the set-up and THEN execute a session to evaluate tensors and run operations (ops)
* Interactive session create your coding and run on the fly.

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()
```

## Multilayer Perceptron with Tensor Flow

**Creating placeholders**

It's a best practice to create placeholders before variable assignments when using TensorFlow. Here we'll create placeholders for inputs ("Xs") and outputs ("Ys").

Placeholder 'X': represents the "space" allocated input or the images.

* Each input has 784 pixels distributed by a 28 width x 28 height matrix   
* The 'shape' argument defines the tensor size by its dimensions.  
* 1st dimension = None. Indicates that the batch size, can be of any size.  
* 2nd dimension = 784. Indicates the number of pixels on a single flattened MNIST image. 

Placeholder 'Y':_ represents the final output or the labels.

* 10 possible classes (0,1,2,3,4,5,6,7,8,9) 
* The 'shape' argument defines the tensor size by its dimensions. 
* 1st dimension = None. Indicates that the batch size, can be of any size.  
* 2nd dimension = 10. Indicates the number of targets/outcomes

dtype for both placeholders: if you not sure, use tf.float32. The limitation here is that the later presented softmax function only accepts float32 or float64 dtypes. For more dtypes, check TensorFlow's documentation here

```python
x  = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
# Weight tensor
W = tf.Variable(tf.zeros([784,10],tf.float32))
# Bias tensor
b = tf.Variable(tf.zeros([10],tf.float32))
# run the op initialize_all_variables using an interactive session
sess.run(tf.initialize_all_variables())
#mathematical operation to add weights and biases to the inputs
tf.matmul(x,W) + b
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#Load 50 training examples for each training iteration   
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}) * 100
print("The final accuracy for the simple ANN model is: {} % ".format(acc) )
sess.close() #finish the session
```

## Convolutional Network with Tensor Flow

### 2nd part: Deep Learning applied on MNIST

Architecture of our network is:

* (Input) -> [batch_size, 28, 28, 1] >> Apply 32 filter of [5x5]
* (Convolutional layer 1) -> [batch_size, 28, 28, 32]
* (ReLU 1) -> [?, 28, 28, 32]
* (Max pooling 1) -> [?, 14, 14, 32]
* (Convolutional layer 2) -> [?, 14, 14, 64]
* (ReLU 2) -> [?, 14, 14, 64]
* (Max pooling 2) -> [?, 7, 7, 64]
* [fully connected layer 3] -> [1x1024]
* [ReLU 3] -> [1x1024]
* [Drop out] -> [1x1024]
* [fully connected layer 4] -> [1x10]

0) Input - MNIST dataset

1) Convolutional and Max-Pooling

2) Convolutional and Max-Pooling

3) Fully Connected Layer

4) Processing - Dropout

5) Readout layer - Fully Connected

6) Outputs - Classified digits

```python
import tensorflow as tf

# finish possible remaining session
sess.close()

#Start interactive session
sess = tf.InteractiveSession()
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
width = 28 # width of the image in pixels 
height = 28 # height of the image in pixels
flat = width * height # number of pixels in one image 
class_output = 10 # number of possible classifications for the problem
x  = tf.placeholder(tf.float32, shape=[None, flat])
y_ = tf.placeholder(tf.float32, shape=[None, class_output])
x_image = tf.reshape(x, [-1,28,28,1])  # The input image is a 28 pixels by 28 pixels and 1 channel (grayscale). In this case the first dimension is the batch number of the image (position of the input on the batch) and can be of any size (due to -1)
```

**[Every layer explanation](https://courses.bigdatauniversity.com/courses/course-v1:BigDataUniversity+ML0120EN+2016/courseware/76d637cbe8024e509dc445df847e6c3a/d2529e01786a412fb009daef4b002a48/)**

```python
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32])) # need 32 biases for 32 outputs
convolve1= tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
h_conv1 = tf.nn.relu(convolve1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2

W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64])) #need 64 biases for 64 outputs
convolve2= tf.nn.conv2d(layer1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')+ b_conv2
h_conv2 = tf.nn.relu(convolve2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2
layer2= h_pool2
layer2_matrix = tf.reshape(layer2, [-1, 7*7*64])
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024])) # need 1024 biases for 1024 outputs
fcl3=tf.matmul(layer2_matrix, W_fc1) + b_fc1
h_fc1 = tf.nn.relu(fcl3)
layer3= h_fc1
keep_prob = tf.placeholder(tf.float32)
layer3_drop = tf.nn.dropout(layer3, keep_prob)
W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1)) #1024 neurons
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10])) # 10 possibilities for digits [0,1,2,3,4,5,6,7,8,9]
fcl4=tf.matmul(layer3_drop, W_fc2) + b_fc2
y_conv= tf.nn.softmax(fcl4)
layer4= y_conv
```

Define functions and train the model

```python
import numpy as np
layer4_test =[[0.9, 0.1, 0.1],[0.9, 0.1, 0.1]]
y_test=[[1.0, 0.0, 0.0],[1.0, 0.0, 0.0]]
np.mean( -np.sum(y_test * np.log(layer4_test),1))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(layer4), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(layer4,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

for i in range(1100):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, float(train_accuracy)))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
kernels = sess.run(tf.reshape(tf.transpose(W_conv1, perm=[2, 3, 0,1]),[32,-1]))

from utils import tile_raster_images
import matplotlib.pyplot as plt
from PIL import Image
%matplotlib inline
image = Image.fromarray(tile_raster_images(kernels, img_shape=(5, 5) ,tile_shape=(4, 8), tile_spacing=(1, 1)))
### Plot image
plt.rcParams['figure.figsize'] = (18.0, 18.0)
imgplot = plt.imshow(image)
imgplot.set_cmap('gray')  
import numpy as np
plt.rcParams['figure.figsize'] = (5.0, 5.0)
sampleimage = mnist.test.images[1]
plt.imshow(np.reshape(sampleimage,[28,28]), cmap="gray")

# check the first convolution layer

ActivatedUnits = sess.run(convolve1,feed_dict={x:np.reshape(sampleimage,[1,784],order='F'),keep_prob:1.0})
filters = ActivatedUnits.shape[3]
plt.figure(1, figsize=(20,20))
n_columns = 6
n_rows = np.math.ceil(filters / n_columns) + 1
for i in range(filters):
    plt.subplot(n_rows, n_columns, i+1)
    plt.title('Filter ' + str(i))
    plt.imshow(ActivatedUnits[0,:,:,i], interpolation="nearest", cmap="gray")

# check the second convolution layer
ActivatedUnits = sess.run(convolve2,feed_dict={x:np.reshape(sampleimage,[1,784],order='F'),keep_prob:1.0})
filters = ActivatedUnits.shape[3]
plt.figure(1, figsize=(20,20))
n_columns = 8
n_rows = np.math.ceil(filters / n_columns) + 1
for i in range(filters):
    plt.subplot(n_rows, n_columns, i+1)
    plt.title('Filter ' + str(i))
    plt.imshow(ActivatedUnits[0,:,:,i], interpolation="nearest", cmap="gray")

sess.close() #finish the session

```

# Module 3 - Recurrent Neural Network

## Learning Objectives

In this lesson you will learn about:

* The Recurrent Neural Network Model 
* Long Short-Term Memory
* Recursive Neural Tensor Network Theory
* Applying Recurrent Networks to Language Modelling

## The Sequential Problem

Long Short-Term Memory, or LSTM for short, is one of the proposed solutions or upgrades to the Recurrent Neural Network model. The Recurrent Neural Network is a specialized type of Neural Network that solves the issue of maintaining context for Sequential data -- such as Weather data, Stocks, Genes, etc. At each iterative step, the processing unit takes in an input and the current state of the network, and produces an output and a new state that is re-fed into the network.

## The Recurrent Neural Network Model

## The Long Short-Term Memory Model

### LSTM basic

```python
import numpy as np
import tensorflow as tf
import tensorflow as tf

sample_input = tf.constant([[1,2,3,4,3,2],[3,2,2,2,2,2]],dtype=tf.float32)

LSTM_CELL_SIZE = 3  #2 hidden nodes

lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_CELL_SIZE, state_is_tuple=True)
state = (tf.zeros([2,LSTM_CELL_SIZE]),)*2

with tf.variable_scope("LSTM_sample4"):
    output, state_new = lstm_cell(sample_input, state)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print (sess.run(sample_input))
print (sess.run(state))
print (sess.run(output))
print (sess.run(state_new))
sample_LSTM_CELL_SIZE = 3  #3 hidden nodes (it is equal to time steps)
sample_batch_size = 2
num_layers = 2 # Lstm layers
sample_input = tf.constant([[[1,2,3,4,3,2], [1,2,1,1,1,2],[1,2,2,2,2,2]],[[1,2,3,4,3,2],[3,2,2,1,1,2],[0,0,0,0,3,2]]],dtype=tf.float32)
lstm_cell = tf.contrib.rnn.BasicLSTMCell(sample_LSTM_CELL_SIZE, state_is_tuple=True)
stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell] * num_layers)
_initial_state = stacked_lstm.zero_state(sample_batch_size, tf.float32)
with tf.variable_scope("Stacked_LSTM_sample8"):
    outputs, new_state =  tf.nn.dynamic_rnn(stacked_lstm, sample_input,dtype=tf.float32, initial_state=_initial_state)


sess = tf.Session()
sess.run(tf.global_variables_initializer()) 

print (sess.run(sample_input))
sess.run(_initial_state)
print (sess.run(new_state))
print (sess.run(output))
```

## Recursive Neural Tensor Networks

### MNIST DATA CLASSIFICATION WITH RNN/LSTM

```python
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True)
trainimgs = mnist.train.images
trainlabels = mnist.train.labels
testimgs = mnist.test.images
testlabels = mnist.test.labels 

ntrain = trainimgs.shape[0]
ntest = testimgs.shape[0]
dim = trainimgs.shape[1]
nclasses = trainlabels.shape[1]
print "Train Images: ", trainimgs.shape
print "Train Labels  ", trainlabels.shape
print
print "Test Images:  " , testimgs.shape
print "Test Labels:  ", testlabels.shape

samplesIdx = [100, 101, 102]  #<-- You can change these numbers here to see other samples

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()

ax1 = fig.add_subplot(121)
ax1.imshow(testimgs[samplesIdx[0]].reshape([28,28]), cmap='gray')


xx, yy = np.meshgrid(np.linspace(0,28,28), np.linspace(0,28,28))
X =  xx ; Y =  yy
Z =  100*np.ones(X.shape)

img = testimgs[77].reshape([28,28])
ax = fig.add_subplot(122, projection='3d')
ax.set_zlim((0,200))

offset=200
for i in samplesIdx:
    img = testimgs[i].reshape([28,28]).transpose()
    ax.contourf(X, Y, img, 200, zdir='z', offset=offset, cmap="gray")
    offset -= 100

    ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

plt.show()


for i in samplesIdx:
    print "Sample: {0} - Class: {1} - Label Vector: {2} ".format(i, np.nonzero(testlabels[i])[0], testlabels[i])

n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)


learning_rate = 0.001
training_iters = 100000
batch_size = 100
display_step = 10

x = tf.placeholder(dtype="float", shape=[None, n_steps, n_input], name="x")
y = tf.placeholder(dtype="float", shape=[None, n_classes], name="y")

weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Define a lstm cell with tensorflow
lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)

#initial state
#initial_state = (tf.zeros([1,n_hidden]),)*2

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input) [100x28x28]

    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
   

    # Get lstm cell output
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, inputs=x, dtype=tf.float32)

    # Get lstm cell output
    #outputs, states = lstm_cell(x , initial_state)
    
    # The output of the rnn would be a [100x28x128] matrix. we use the linear activation to map it to a [?x10 matrix]
    # Linear activation, using rnn inner loop last output
    # output [100x128] x  weight [128, 10] + []
    output = tf.reshape(tf.split(outputs, 28, axis=1, num=None, name='split')[-1],[-1,128])
    return tf.matmul(output, weights['out']) + biases['out']

with tf.variable_scope('forward3'):
    pred = RNN(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred ))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:

        # We will read a batch of 100 images [100 x 784] as batch_x
        # batch_y is a matrix of [100x10]
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        
        # We consider each row of the image as one sequence
        # Reshape data to get 28 seq of 28 elements, so that, batxh_x is [100x28x28]
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
    

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        
        
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

sess.close()
```

## Applying Recurrent Networks to Language Modelling

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf

!wget -q -O /resources/data/ptb.zip https://ibm.box.com/shared/static/z2yvmhbskc45xd2a9a4kkn6hg4g4kj5r.zip
!unzip -o /resources/data/ptb.zip -d /resources/
!cp /resources/ptb/reader.py .
import reader
!wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz 
!tar xzf simple-examples.tgz -C /resources/data/
#Initial weight scale
init_scale = 0.1
#Initial learning rate
learning_rate = 1.0
#Maximum permissible norm for the gradient (For gradient clipping -- another measure against Exploding Gradients)
max_grad_norm = 5
#The number of layers in our model
num_layers = 2
#The total number of recurrence steps, also known as the number of layers when our RNN is "unfolded"
num_steps = 20
#The number of processing units (neurons) in the hidden layers
hidden_size = 200
#The maximum number of epochs trained with the initial learning rate
max_epoch = 4
#The total number of epochs in training
max_max_epoch = 13
#The probability for keeping data in the Dropout Layer (This is an optimization, but is outside our scope for this notebook!)
#At 1, we ignore the Dropout Layer wrapping.
keep_prob = 1
#The decay for the learning rate
decay = 0.5
#The size for each batch of data
batch_size = 30
#The size of our vocabulary
vocab_size = 10000
#Training flag to separate training from testing
is_training = 1
#Data directory for our dataset
data_dir = "/resources/data/simple-examples/data/"

session=tf.InteractiveSession()

# Reads the data and separates it into training data, validation data and testing data
raw_data = reader.ptb_raw_data(data_dir)
train_data, valid_data, test_data, _ = raw_data

itera = reader.ptb_iterator(train_data, batch_size, num_steps)
first_touple=itera.next()
x=first_touple[0]
y=first_touple[1]

_input_data = tf.placeholder(tf.int32, [batch_size, num_steps]) #[30#20]
_targets = tf.placeholder(tf.int32, [batch_size, num_steps]) #[30#20]

feed_dict={_input_data:x, _targets:y}
session.run(_input_data,feed_dict)

lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0)
stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell] * num_layers)
session.run(_initial_state,feed_dict)
embedding = tf.get_variable("embedding", [vocab_size, hidden_size])  #[10000x200]
session.run(tf.global_variables_initializer())
session.run(embedding, feed_dict)
# Define where to get the data for our embeddings from
inputs = tf.nn.embedding_lookup(embedding, _input_data)  #shape=(30, 20, 200) 

session.run(inputs[0], feed_dict)

outputs, new_state =  tf.nn.dynamic_rnn(stacked_lstm, inputs, initial_state=_initial_state)

session.run(tf.global_variables_initializer())
session.run(outputs[0], feed_dict)

output = tf.reshape(outputs, [-1, size])
output

session.run(output[0], feed_dict)

softmax_w = tf.get_variable("softmax_w", [size, vocab_size]) #[200x1000]
softmax_b = tf.get_variable("softmax_b", [vocab_size]) #[1x1000]
logits = tf.matmul(output, softmax_w) + softmax_b

session.run(tf.global_variables_initializer())
logi = session.run(logits, feed_dict)
logi.shape

First_word_output_probablity = logi[0]
First_word_output_probablity.shape

embedding_array= session.run(embedding, feed_dict)
np.argmax(First_word_output_probablity)

y[0][0]
_targets
targ = session.run(tf.reshape(_targets, [-1]), feed_dict) 
first_word_target_code= targ[0]
first_word_target_code

first_word_target_vec = session.run( tf.nn.embedding_lookup(embedding, targ[0]))
first_word_target_vec

loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(_targets, [-1])],[tf.ones([batch_size * num_steps])])
session.run(loss, feed_dict)
cost = tf.reduce_sum(loss) / batch_size

session.run(tf.global_variables_initializer())
session.run(cost, feed_dict)

# 
final_state = new_state
# Create a variable for the learning rate
lr = tf.Variable(0.0, trainable=False)
# Create the gradient descent optimizer with our learning rate
optimizer = tf.train.GradientDescentOptimizer(lr)
# Get all TensorFlow variables marked as "trainable" (i.e. all of them except _lr, which we just created)
tvars = tf.trainable_variables()
tvars
tvars=tvars[3:]
[v.name for v in tvars]
var_x = tf.placeholder(tf.float32)
var_y = tf.placeholder(tf.float32) 
func_test = 2.0*var_x*var_x + 3.0*var_x*var_y
session.run(tf.global_variables_initializer())
feed={var_x:1.0,var_y:2.0}
session.run(func_test, feed)

var_grad = tf.gradients(func_test, [var_x])
session.run(var_grad,feed)
var_grad = tf.gradients(func_test, [var_y])
session.run(var_grad,feed)
tf.gradients(cost, tvars)
grad_t_list = tf.gradients(cost, tvars)
#sess.run(grad_t_list,feed_dict)
max_grad_norm
# Define the gradient clipping threshold
grads, _ = tf.clip_by_global_norm(grad_t_list, max_grad_norm)
grads
session.run(grads,feed_dict)
# Create the training TensorFlow Operation through our optimizer
train_op = optimizer.apply_gradients(zip(grads, tvars))
session.run(tf.global_variables_initializer())
session.run(train_op,feed_dict)

class PTBModel(object):

    def __init__(self, is_training):
        ######################################
        # Setting parameters for ease of use #
        ######################################
        self.batch_size = batch_size
        self.num_steps = num_steps
        size = hidden_size
        self.vocab_size = vocab_size
        
        ###############################################################################
        # Creating placeholders for our input data and expected outputs (target data) #
        ###############################################################################
        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps]) #[30#20]
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps]) #[30#20]

        ##########################################################################
        # Creating the LSTM cell structure and connect it with the RNN structure #
        ##########################################################################
        # Create the LSTM unit. 
        # This creates only the structure for the LSTM and has to be associated with a RNN unit still.
        # The argument n_hidden(size=200) of BasicLSTMCell is size of hidden layer, that is, the number of hidden units of the LSTM (inside A).
        # Size is the same as the size of our hidden layer, and no bias is added to the Forget Gate. 
        # LSTM cell processes one word at a time and computes probabilities of the possible continuations of the sentence.
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0)
        
        # Unless you changed keep_prob, this won't actually execute -- this is a dropout wrapper for our LSTM unit
        # This is an optimization of the LSTM output, but is not needed at all
        if is_training and keep_prob < 1:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
        
        # By taking in the LSTM cells as parameters, the MultiRNNCell function junctions the LSTM units to the RNN units.
        # RNN cell composed sequentially of multiple simple cells.
        stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell] * num_layers)

        # Define the initial state, i.e., the model state for the very first data point
        # It initialize the state of the LSTM memory. The memory state of the network is initialized with a vector of zeros and gets updated after reading each word.
        self._initial_state = stacked_lstm.zero_state(batch_size, tf.float32)

        ####################################################################
        # Creating the word embeddings and pointing them to the input data #
        ####################################################################
        with tf.device("/cpu:0"):
            # Create the embeddings for our input data. Size is hidden size.
            embedding = tf.get_variable("embedding", [vocab_size, size])  #[10000x200]
            # Define where to get the data for our embeddings from
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        # Unless you changed keep_prob, this won't actually execute -- this is a dropout addition for our inputs
        # This is an optimization of the input processing and is not needed at all
        if is_training and keep_prob < 1:
            inputs = tf.nn.dropout(inputs, keep_prob)

        ############################################
        # Creating the input structure for our RNN #
        ############################################
        # Input structure is 20x[30x200]
        # Considering each word is represended by a 200 dimentional vector, and we have 30 batchs, we create 30 word-vectors of size [30xx2000]
        #inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, num_steps, inputs)]
        # The input structure is fed from the embeddings, which are filled in by the input data
        # Feeding a batch of b sentences to a RNN:
        # In step 1,  first word of each of the b sentences (in a batch) is input in parallel.  
        # In step 2,  second word of each of the b sentences is input in parallel. 
        # The parallelism is only for efficiency.  
        # Each sentence in a batch is handled in parallel, but the network sees one word of a sentence at a time and does the computations accordingly. 
        # All the computations involving the words of all sentences in a batch at a given time step are done in parallel. 

        ####################################################################################################
        # Instanciating our RNN model and retrieving the structure for returning the outputs and the state #
        ####################################################################################################
        
        outputs, state = tf.nn.dynamic_rnn(stacked_lstm, inputs, initial_state=self._initial_state)

        #########################################################################
        # Creating a logistic unit to return the probability of the output word #
        #########################################################################
        output = tf.reshape(outputs, [-1, size])
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size]) #[200x1000]
        softmax_b = tf.get_variable("softmax_b", [vocab_size]) #[1x1000]
        logits = tf.matmul(output, softmax_w) + softmax_b

        #########################################################################
        # Defining the loss and cost functions for the model's learning to work #
        #########################################################################
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(self._targets, [-1])],
                                                      [tf.ones([batch_size * num_steps])])
        self._cost = cost = tf.reduce_sum(loss) / batch_size

        # Store the final state
        self._final_state = state

        #Everything after this point is relevant only for training
        if not is_training:
            return

        #################################################
        # Creating the Training Operation for our Model #
        #################################################
        # Create a variable for the learning rate
        self._lr = tf.Variable(0.0, trainable=False)
        # Get all TensorFlow variables marked as "trainable" (i.e. all of them except _lr, which we just created)
        tvars = tf.trainable_variables()
        # Define the gradient clipping threshold
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)
        # Create the gradient descent optimizer with our learning rate
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        # Create the training TensorFlow Operation through our optimizer
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    # Helper functions for our LSTM RNN class

    # Assign the learning rate for this model
    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    # Returns the input data for this model at a point in time
    @property
    def input_data(self):
        return self._input_data

    # Returns the targets for this model at a point in time
    @property
    def targets(self):
        return self._targets

    # Returns the initial state for this model
    @property
    def initial_state(self):
        return self._initial_state

    # Returns the defined Cost
    @property
    def cost(self):
        return self._cost

    # Returns the final state for this model
    @property
    def final_state(self):
        return self._final_state

    # Returns the current learning rate for this model
    @property
    def lr(self):
        return self._lr

    # Returns the training operation defined for this model
    @property
    def train_op(self):
        return self._train_op

##########################################################################################################################
# run_epoch takes as parameters the current session, the model instance, the data to be fed, and the operation to be run #
##########################################################################################################################
def run_epoch(session, m, data, eval_op, verbose=False):

    #Define the epoch size based on the length of the data, batch size and the number of steps
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    #state = m.initial_state.eval()
    #m.initial_state = tf.convert_to_tensor(m.initial_state) 
    #state = m.initial_state.eval()
    state = session.run(m.initial_state)
    
    #For each step and data point
    for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size, m.num_steps)):
 
        #Evaluate and return cost, state by running cost, final_state and the function passed as parameter
        cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                     {m.input_data: x,
                                      m.targets: y,
                                      m.initial_state: state})
        
        #Add returned cost to costs (which keeps track of the total costs for this epoch)
        costs += cost
        
        #Add number of steps to iteration counter
        iters += m.num_steps

        if verbose and step % (epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" % (step * 1.0 / epoch_size, np.exp(costs / iters),
              iters * m.batch_size / (time.time() - start_time)))

    # Returns the Perplexity rating for us to keep track of how the model is evolving
    return np.exp(costs / iters)

# Reads the data and separates it into training data, validation data and testing data
raw_data = reader.ptb_raw_data(data_dir)
train_data, valid_data, test_data, _ = raw_data

#Initializes the Execution Graph and the Session
with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-init_scale,init_scale)
    
    # Instantiates the model for training
    # tf.variable_scope add a prefix to the variables created with tf.get_variable
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True)
        
    # Reuses the trained parameters for the validation and testing models
    # They are different instances but use the same variables for weights and biases, they just don't change when data is input
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False)
        mtest = PTBModel(is_training=False)

    #Initialize all variables
    tf.global_variables_initializer().run()

    for i in range(max_max_epoch):
        # Define the decay for this epoch
        lr_decay = decay ** max(i - max_epoch, 0.0)
        
        # Set the decayed learning rate as the learning rate for this epoch
        m.assign_lr(session, learning_rate * lr_decay)

        print("Epoch %d : Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        
        # Run the loop for this epoch in the training model
        train_perplexity = run_epoch(session, m, train_data, m.train_op,
                                   verbose=True)
        print("Epoch %d : Train Perplexity: %.3f" % (i + 1, train_perplexity))
        
        # Run the loop for this epoch in the validation model
        valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
        print("Epoch %d : Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
    
    # Run the loop in the testing model to see how effective was our training
    test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
    
    print("Test Perplexity: %.3f" % test_perplexity)


```

## APPLYING RNN/LSTM TO CHARACTER MODELLING

```python
import tensorflow as tf
import time
import codecs
import os
import collections
from six.moves import cPickle
import numpy as np

class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            data = f.read()
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        self.tensor = np.array(list(map(self.vocab.get, data)))
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

        # When the data (tensor) is too small, let's give them a better error message
        if self.num_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)


    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0

batch_size = 60 #minibatch size, i.e. size of dataset in each epoch
seq_length = 50 #RNN sequence length
num_epochs = 25 # you should change it to 50 if you want to see a relatively good results
learning_rate = 0.002
decay_rate = 0.97
rnn_size = 128 #size of RNN hidden state
num_layers = 2 #number of layers in the RNN

!wget -nv -O /resources/data/input.txt https://ibm.box.com/shared/static/a3f9e9mbpup09toq35ut7ke3l3lf03hg.txt 

data_loader = TextLoader('/resources/data/', batch_size, seq_length)
vocab_size = data_loader.vocab_size
data_loader.vocab_size

data_loader.num_batches
x,y = data_loader.next_batch()
x
x.shape  #batch_size =50, seq_length=50
y
data_loader.chars[0:5]
data_loader.vocab['t']
#cell= rnn_cell.BasicLSTMCell
cell = tf.contrib.rnn.BasicRNNCell(rnn_size)
# a two layer cell
stacked_cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
# hidden state size
stacked_cell.output_size
stacked_cell.state_size
input_data = tf.placeholder(tf.int32, [batch_size, seq_length])# a 60x50
targets = tf.placeholder(tf.int32, [batch_size, seq_length]) # a 60x50
initial_state = stacked_cell.zero_state(batch_size, tf.float32) #why batch_size ? 60x128
input_data
session = tf.Session()
feed_dict={input_data:x, targets:y}

session.run(input_data, feed_dict)
with tf.variable_scope('rnnlm',reuse=False):
    softmax_w = tf.get_variable("softmax_w", [rnn_size, vocab_size]) #128x65
    softmax_b = tf.get_variable("softmax_b", [vocab_size]) # 1x65)
    with tf.device("/cpu:0"):
        embedding = tf.get_variable("embedding", [vocab_size, rnn_size])  #65x128
        #input_data is a matrix of 60x50 and embedding is dictionary of 65x128 for all 65 characters
        # embedding_lookup goes to each row of input_data, and for each character in the row, finds the correspond vector in embedding
        # it creates a 60*50*[1*128] matrix
        # so, the first elemnt of em, is a matrix of 50x128, which each row of it is vector representing that character
        em = tf.nn.embedding_lookup(embedding, input_data) # em is 60x50x[1*128]
        # split: Splits a tensor into sub tensors.
        # syntax:  tf.split(split_dim, num_split, value, name='split')
        # it will split the 60x50x[1x128] matrix into 50 matrix of 60x[1*128]
        inputs = tf.split(em, seq_length, 1)
        # It will convert the list to 50 matrix of [60x128]
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

session.run(tf.global_variables_initializer())
session.run(embedding)
em = tf.nn.embedding_lookup(embedding, input_data)
em
emp = session.run(em,feed_dict={input_data:x})
emp.shape
emp[0]
inputs = tf.split(em, seq_length, 1)
inputs[0:5]
inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
inputs[0:5]
session.run(inputs[0],feed_dict={input_data:x})
cell.state_size
#outputs is 50x[60*128]
outputs, last_state = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, initial_state, stacked_cell, loop_function=None, scope='rnnlm')
outputs[0:5]

test = outputs[0]
test
session.run(tf.global_variables_initializer())
session.run(test,feed_dict={input_data:x})

output = tf.reshape(tf.concat( outputs,1), [-1, rnn_size])
output

logits = tf.matmul(output, softmax_w) + softmax_b
logits

probs = tf.nn.softmax(logits)
probs

session.run(tf.global_variables_initializer())
session.run(probs,feed_dict={input_data:x})

loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],
                [tf.reshape(targets, [-1])],
                [tf.ones([batch_size * seq_length])],
                vocab_size)

cost = tf.reduce_sum(loss) / batch_size / seq_length
cost

final_state = last_state
final_state
lr = tf.Variable(0.0, trainable=False)
grad_clip =5.
tvars = tf.trainable_variables()
session.run(tf.global_variables_initializer())
[v.name for v in tf.global_variables()]

grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), grad_clip)
grads

session.run(grads, feed_dict)[0]
optimizer = tf.train.AdamOptimizer(lr)
train_op = optimizer.apply_gradients(zip(grads, tvars))

import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq

import numpy as np

class LSTMModel():
    def __init__(self,sample=False):
        rnn_size = 128 # size of RNN hidden state vector
        batch_size = 60 # minibatch size, i.e. size of dataset in each epoch
        seq_length = 50 # RNN sequence length
        num_layers = 2 # number of layers in the RNN
        vocab_size = 65
        grad_clip = 5.
        if sample:
            print("sample mode")
            batch_size = 1
            seq_length = 1
        # The core of the model consists of an LSTM cell that processes one char at a time and computes probabilities of the possible continuations of the char. 
        basic_cell = tf.contrib.rnn.BasicRNNCell(rnn_size)
        # model.cell.state_size is (128, 128)
        self.stacked_cell = tf.contrib.rnn.MultiRNNCell([basic_cell] * num_layers)

        self.input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
        self.targets = tf.placeholder(tf.int32, [batch_size, seq_length])
        # Initial state of the LSTM memory.
        # The memory state of the network is initialized with a vector of zeros and gets updated after reading each char. 
        self.initial_state = stacked_cell.zero_state(batch_size, tf.float32) #why batch_size

        with tf.variable_scope('rnnlm_class1'):
            softmax_w = tf.get_variable("softmax_w", [rnn_size, vocab_size]) #128x65
            softmax_b = tf.get_variable("softmax_b", [vocab_size]) # 1x65
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [vocab_size, rnn_size])  #65x128
                inputs = tf.split(tf.nn.embedding_lookup(embedding, self.input_data), seq_length, 1)
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
                #inputs = tf.split(em, seq_length, 1)
                
                



        # The value of state is updated after processing each batch of chars.
        outputs, last_state = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, self.initial_state, self.stacked_cell, loop_function=None, scope='rnnlm_class1')
        output = tf.reshape(tf.concat(outputs,1), [-1, rnn_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([batch_size * seq_length])],
                vocab_size)
        self.cost = tf.reduce_sum(loss) / batch_size / seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        
    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1):
        state = sess.run(self.stacked_cell.zero_state(1, tf.float32))
        print state
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else: # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret

with tf.variable_scope("rnn"):
    model = LSTMModel()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
e=1
sess.run(tf.assign(model.lr, learning_rate * (decay_rate ** e)))
data_loader.reset_batch_pointer()
state = sess.run(model.initial_state)
state
x, y = data_loader.next_batch()
feed = {model.input_data: x, model.targets: y, model.initial_state:state}
train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
train_loss
state

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(num_epochs): # num_epochs is 5 for test, but should be higher
        sess.run(tf.assign(model.lr, learning_rate * (decay_rate ** e)))
        data_loader.reset_batch_pointer()
        state = sess.run(model.initial_state) # (2x[60x128])
        for b in range(data_loader.num_batches): #for each batch
            start = time.time()
            x, y = data_loader.next_batch()
            feed = {model.input_data: x, model.targets: y, model.initial_state:state}
            train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
            end = time.time()
        print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                .format(e * data_loader.num_batches + b, num_epochs * data_loader.num_batches, e, train_loss, end - start))
        #model.sample(sess, data_loader.chars , data_loader.vocab, num=200, prime='The ', sampling_type=1)

sess = tf.InteractiveSession()
with tf.variable_scope("sample_test"):
    sess.run(tf.global_variables_initializer())
    m = LSTMModel(sample=True)

prime='The '
num=200
sampling_type=1
vocab=data_loader.vocab
chars=data_loader.chars 

sess.run(m.initial_state)

#print state
sess.run(tf.global_variables_initializer())
state=sess.run(m.initial_state)
for char in prime[:-1]:
    x = np.zeros((1, 1))
    x[0, 0] = vocab[char]
    feed = {m.input_data: x, m.initial_state:state}
    [state] = sess.run([m.final_state], feed)

state

def weighted_pick(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    return(int(np.searchsorted(t, np.random.rand(1)*s)))

ret = prime
char = prime[-1]
for n in range(num):
    x = np.zeros((1, 1))
    x[0, 0] = vocab[char]
    feed = {m.input_data: x, m.initial_state:state}
    [probs, state] = sess.run([m.probs, m.final_state], feed)
    p = probs[0]

    if sampling_type == 0:
        sample = np.argmax(p)
    elif sampling_type == 2:
        if char == ' ':
            sample = weighted_pick(p)
        else:
            sample = np.argmax(p)
    else: # sampling_type == 1 default:
        sample = weighted_pick(p)

    pred = chars[sample]
    ret += pred
    char = pred

ret

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
state=sess.run(m.initial_state)
m.sample(sess, data_loader.chars , data_loader.vocab, num=200, prime='The ', sampling_type=1)

```

# Module 4 - Unsupervised Learning

## Learning Objectives

In this lesson you will learn about:

* The Applications of Unsupervised Learning
* Restricted Boltzmann Machine
* Training a Restricted Boltzman Machine

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
