Machine Learning Foundations: A Case Study Approach
by University of Washington

# Week 1: Welcome

## SFrame

# Week 2: Regression: Predicting House Prices

## Linear regression modeling

## Evaluating regression models

### Predicting house prices: IPython Notebook

Open the iPython Notebook used in this lesson to follow along
Loading & exploring house sale data
Splitting the data into training and test sets
Learning a simple regression model to predict house prices from house size
Evaluating error (RMSE) of the simple model
Visualizing predictions of simple model with Matplotlib
Inspecting the model coefficients learned
Exploring other features of the data
Learning a model to predict house prices from more features
Applying learned models to predict price of an average house
Applying learned models to predict price of two fancy houses

# Week 3: Classification: Analyzing Sentiment

## Classification modeling

Linear classifiers
Decision boundaries

## Evaluating classification models

Training and evaluating a classifier
What's a good accuracy?
False positives, false negatives, and confusion matrices
Learning curves
Class probabilities

## Analyzing sentiment: IPython Notebook

Open the iPython Notebook used in this lesson to follow along
Loading & exploring product review data
Creating the word count vector
Exploring the most popular product
Defining which reviews have positive or negative sentiment
Training a sentiment classifier
Evaluating a classifier & the ROC curve
Applying model to find most positive & negative reviews for a product
Exploring the most positive & negative aspects of a product

# Week 4: Clustering and Similarity: Retrieving Documents

## Algorithms for retrieval and measuring similarity of documents

Word count representation for measuring similarity
Prioritizing important words with tf-idf
Calculating tf-idf vectors
Retrieving similar documents using nearest neighbor search

## Clustering models and algorithms

k-means: A clustering algorithm

## Document retrieval: IPython Notebook

Open the iPython Notebook used in this lesson to follow along
Loading & exploring Wikipedia data
Exploring word counts
Computing & exploring TF-IDFs
Computing distances between Wikipedia articles
Building & exploring a nearest neighbors model for Wikipedia articles
Examples of document retrieval in action

# Week 5: Recommending Products

## Recommender systems

Building a recommender system via classification

## Co-occurrence matrices for collaborative filtering

Collaborative filtering: People who bought this also bought...
Effect of popular items
Normalizing co-occurrence matrices and leveraging purchase histories

## Matrix factorization

The matrix completion task
Recommendations from known user/item features
Predictions in matrix form
Discovering hidden structure by matrix factorization
Bringing it all together: Featurized matrix factorization

## Performance metrics for recommender systems
A performance metric for recommender systems
Optimal recommenders
Precision-recall curves

## Song recommender: IPython Notebook

Open the iPython Notebook used in this lesson to follow along
Loading and exploring song data
Creating & evaluating a popularity-based song recommender
Creating & evaluating a personalized song recommender
Using precision-recall to compare recommender models

# Week 6: Deep Learning: Searching for Images

## Neural networks: Learning very non-linear features

Learning very non-linear features with neural networks
Deep learning & deep features
Application of deep learning to computer vision
Deep learning performance
Demo of deep learning model on ImageNet data
Other examples of deep learning in computer vision
Challenges of deep learning
Deep Features


## Deep features for image classification: iPython Notebook

Open the iPython Notebook used in this lesson to follow along
Loading image data
Training & evaluating a classifier using raw image pixels
Training & evaluating a classifier using deep features

## Deep features for image retrieval: iPython Notebook

Open the iPython Notebook used in this lesson to follow along
Loading image data
Creating a nearest neighbors model for image retrieval
Querying the nearest neighbors model to retrieve images
Querying for the most similar images for car image
Displaying other example image retrievals with a Python lambda