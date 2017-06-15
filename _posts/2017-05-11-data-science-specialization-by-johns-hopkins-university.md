---
title: Data Science Specialization by Johns Hopkins University
layout: post
published: true
category: [Data Science]
---

<!--more-->

* TOC
{:toc}

# Course 1: The Data Scientist’s Toolbox

Week 1, 2 are already known.

## Week 3

**Types of Data Science Questions**, in approximate order of difficulty:

* Descriptive

    Goal: Describe a set of data.
    This method is usually applied to census data.
    It usually can't be generalized without additional statistical modeling.

* Exploratory

    Goal: Exploratory is to find relationship you didn't know about.
    * Exploratory models are good for discovering new connections.
    * They are also useful for defining future studies.
    * Exploratory analyses are usually not the final say.
    * Exploratory analyses alone should not be used for generalizing/predicting.
    * Correlation does not imply causation.

* Inferential

    Goal: Use a relatively small sample of data to say something about a bigger population.
    Inference is commonly the goal of statistical models.
    Inference involves both estimating the qualities you care about and the uncertainty of your estimating.
    Inference depends heavily both on the population and the sample scheme.

* Predictive

    Goal: To use the data on some objects to predict values for another object.
    If X predicts Y, it doesn't mean X causes Y.

* Causal

    Goal: To find what happens to a variable when you make another variable change.
    Usually randomized studies are required to identify causation.
    Causal relationships are usually identified as average effects, but may not apply to every individual.
    Causal models are usually the "gold standard" for data analysis.

* Mechanistic

    Goal: Understand the changes in variables that cause changes in other variables for individual objects.
    Incredibly hard except in simple situations.
    Usually modeled by a set of deterministic equations (physics/engineering science)
    Generally the random component is measurement error.
    If the equations are known but the parameters are not, the parameters can be inferred with data analysis.

Data is the second most important thing, the most important thing is the question.

**Experimental Design**

Confound in experimental design: Researchers found shoe size is correlated with literacy, but they didn't introduce variable "age", generally babies have small shoe size and lower literacy, here "age" is the confounder, if you only observe shoe size and literacy, you may get astray.

**Correlation is not causation.**

# Course 2: R Programming:

# Course 3: Getting and Cleaning Data:

## Week 1:

Packages for read data source:

* xls and xlsx

    ```r
    install.packages("xlsx")
    library(xlsx)
    data <- read.xls("./data.xlsx", sheetIndex=1, header=TRUE)
    ```
* xml

    ```r
    install.packages(XML)
    library(XML)
    fileUrl <- "http://www.w3schools.com/xml/simple.xml"
    doc <- xmlTreeParse(fileUrl, useInternal=TRUE)
    rootNode <- xmlRoot(doc)
    xmlName(rootNode)
    ```
* JSON

    ```r
    library(jsonlite)
    jsonData <- fromJSON("https://api.github.com/users/jtleek/repos")
    names(jsonData)
    ```

For data.table, in the square bracket, after the comma is the expression argument, instead of the column index. The expression arguments take the free variables as the attributes of the original data.table.

data.table can have keys, keys are used to select or join.

```r
DT <- data.table(x=rep(c("a","b","c"), each=100), y=rnorm(300))
setkey(DT, x)
DT["a"] # will return all rows x="a"

DT1 <- data.table(x=c("a","a","b","dt1"), y=1:4)
DT2 <- data.table(x=c("a","b","dt2"), z=5:7)
setkey(DT1,x);setkey(DT2,x)
merge(DT1,DT2)
#    x y z
# 1: a 1 5
# 2: a 2 5
# 3: b 3 6
```

# Course 4: Exploratory Data Analysis:

## Week 1

**Principles of Analytic Graphics**

1. Show comparisons
1. Show causality, mechanism, explanation
1. Show multivariate data
1. Integrate multiple models of evidence
1. Describe and document the evidence
1. Content is king

**Often used summarization and plot function**

```r
summary(pollution$pm25)
boxplot(pollution$pm25, col='blue')
abline(h=12)

hist(pollution$pm25, col='green', breaks=100)
abline(v=12, lwd=2)
abline(v=median(pollution$pm25), col='magenta', lwd=4)
rug(pollution$pm25)

barplot(table(pollution$region), col='wheat', main='Numbers of Counties in Each Region')

# multiple boxplots
boxplot(pm25~region, data=pollution, col='red') 

# multiple histgrams
par(mfrow=c(2,1), mar=c(4,4,2,1))
hist(subset(pollution, region=='east')$pm25, col='green')
hist(subset(pollution, region=='west')$pm25, col='green')

# scatter plot
with(pollution, plot(latitude, pm25, col=region))
abline(h=12, lwd=2, lty=2)

# multiple scatterplots
par(mfrow=c(1,2), mar=c(5,4,2,1))
with(subset(pollution, region=='west'), plot(latitude, pm25, main='West'))
with(subset(pollution, region=='east'), plot(latitude, pm25, main='East'))
```

**Three Plotting System**

1. Base Plotting System

    ```r
    library(datasets)
    data(cars)
    with(cars, plot(speed, dist))
    ```

1. The Lattice System

    ```r
    library(lattice)
    state <- data.frame(state.x77, region=state.region)
    xyplot(Life.Exp~Income | region, data=state, layout=c(4,1))
    ```
1. The ggplot2 System

    ```r
    library(ggplot2)
    data(mpg)
    qplot(displ, hwy, data=mpg)
    ```
**Many base plot functions share a set of parameters**

* pch: the plotting symbol (default is open circle)
* lty: the line type (default is solid line, can be dashed, dotted, etc.)
* lwd: the line width, specified as an integer multiple
* col: the plotting color, specified as an integer number, string, or hex code; the `colors()` function gives you a vector of colors by name
* xlab: character string for x-axis label
* ylab: character string for y-axis label

**Some important base graphics parameters**

The `par()` function is used to specify *global* graphics parameters that affect all plots in the R session.

* las: the orientation of the axis labels on the plot
* bg: the background color
* mar: the margin size
* oma: the outer margin size (default 0 for all sides)
* mfrow: the number of plots per row, column (plots are filled row-wise)
* mfcol: the number of plots per row, column (plots are filled column-wise)

# Course 5: Reproducible Research:

# Course 6: Statistical Inference:

# Course 7: Regression Models:

# Course 8: Practical Machine Learning:

# Course 9: Developing Data Products:

# Course 10: Data Science Capstone:

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>