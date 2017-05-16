---
title: R Cheat Sheet for Data Science
layout: post
published: true
category: [misc]
---

* TOC
{:toc}

This is not an exhaustive reference for R language, I just wrote this according to my recalling of DataCamp courses and Data Science Specialization by Johns Hopkins University on Coursera, but this may be suitable for most data analysts and data scientists.

# Data types - R Objects

There are 5 atomic data types (classes): 

* character
* numeric (real numbers)
* integer
* complex
* logical

Vector can only contain same class of data, list can contain different ones.

Most numbers in R are treated as numeric (double precision real numbers), if you want explicitly an integer, specify a "L" suffix.

Inf is infinity, $$1/0=Inf$$, $$1/Inf=0$$.

NaN is "Not a number", $$0/0=NaN$$, Both Inf and NaN are numeric.

**Mixing objects**, when objects of different class are mixed in a vector, coercion occurs so that every element in the vector is of the same class.

```r
c(1, "a") # character
c(1, TRUE) # numeric
c("a", TRUE) # character
```

# Data types - Vectors and Lists

Vector's elements are atomic, when printed, embraced with single square brackets, List's elements are recursive, when printed, embraced with double square brackets

# Data types - Matrices

Matrix can be constructed directly from vector.

```r
x <- 1:6
dim(x) <- c(2, 3)
x
# 1 3 5
# 2 4 6
```
Another way to create matrix is column-binding or row-binding: `cbind`, `rbind`.

# Data types - Factors

```r
factor(c("yes", "yes", "no", "yes"))
```

# Data types - Missing Values

`is.na()` is used to test if objects are NA

`is.nan()` for NaN

NA values also has class, like integer NA, character NA, etc.

NaN is also NA, but the converse is not true.

# Data types - Data Frames

```r
data.frame(foo=1:4, bar=c(T,T,F,F))
```

# Data Types - Names Attribute

All R objects can also have names

```r
x <- 1:3 # vector
names(x) <- c("foo", "bar", "baz")
x <- list(a=1, b=2, c=3) # list
m <- matrix(1:4, nrow=2, ncol=2) # matrix
dimnames(m) <- list(c("a", "b"), c("c", "d"))
```

# Reading Tabular Data

Commonly used functions:

Reading data

```r
read.table, read.csv
readLines
source # inverse of dump
dget # inverse of dput
load
unserialize
```

Writing data

```r
write.table
writeLines
dump
dput
save
serialize
```

# Reading Large Tables

Specifying parameter `colClasses` can make `read.table` much faster.

```r
initial <- read.table("datatable.txt", nrows=100)
classes <- sapply(initial, class)
tabAll <- read.table("datatable.txt", colClasses=classes)
```

# Connections: Interfaces to the Outside World

Often used connections: `file, gzfile, bzfile, url`.

# Subsetting - Basics

* [] returns element of the same class as original, can be used to extract more than one elements.
* [[]] is used to extract elements of list or data frame, and the returned objects will not necessarily be of the same class.
* $ is used to extract element of list or data frame by name, semantics are similar to [[]].

# Subsetting - Lists

```r
x <- list(foo=1:4, bar=0.6)
x[1]
# $foo
# [1] 1 2 3 4
x[[1]]
# [1] 1 2 3 4
x$bar
# [1] 0.6
x[["bar"]]
# [1] 0.6
x["bar"]
# $bar
# [1] 0.6
```

[] can take a vector, [[]] can take integer sequence


```r
x <- list(a=list(1,2,3), b=c(4,5,6))
x[[c(1,3)]]
# [1] 3
x[[1]][[3]]
# [1] 3
x[[c(2,1)]]
# [1] 4
```

# Subsetting - Matrices

```r
x <- matrix(1:6, 2, 3)
x[1, 2]
# [1] 3
x[2, 1]
# [1] 2
x[1,]
# [1] 1 3 5
x[,2]
# [1] 3 4
x[1, 2, drop=FALSE]
#      [,1]
# [1,]    3
x[1,, drop=FALSE]
#      [,1] [,2] [,3]
# [1,]    1    3    5
```

# Subsetting - Partial Matching

```r
x <- list(abc=1:5)
x$a
# [1] 1 2 3 4 5
x[["a"]]
# NULL
x[["a", exact=FALSE]]
# [1] 1 2 3 4 5
```
# Subsetting - Removing Missing Values

```r
x <- c(1,2,NA,4,NA,6)
bad <- is.na(x)
x[!bad]
# [1] 1 2 4 6

y <- c("a", "b", NA, "d", NA, "f")
good <- complete.cases(x, y)
x[good]
# [1] 1 2 4 6
y[good]
# [1] "a" "b" "d" "f"
```

# Vectorized Operations

for R matrix objects
```r
x * x # element-wise multiplication
x %*% x # true matrix multiplication
```

# Control Structures - Introduction

Common structures are:

* if, else
* for
* while
* repeat # execute an infinite loop
* break
* next # skip an iteration of a loop
* return

# Functions

You can mix positional matching with matching by name. When an argument is matched by name, it's taken from the arguments list and the remaining unnamed arguments are matched in the order of the definition of the function.

... argument indicates a variable numbers of arguments that are passed on to other function, ... is usually used to extend other function and you don't want to copy the entire argument list of the original function.

The ... argument is also used when the number of the arguments can't be known in advance, think of the function `paste, cat`.

# Scoping Rules - Symbol Binding

R will search symbols in `.GlobalEnv` first, if doesn't find the needed symbols, it will continues searching in packages loaded into workspace (print those packages by `search()`).

R has separated namespaces for functions and nonfunctions, so you can have a function and a variable both called the same name.

# Scoping Rules - R Scoping Rules

When a free variable in a function can't be found the function, R will search in parent environment, and down the sequence of parent environment till the variable is found (top environment is global environment or package namespace depending where the variable is defined). If still can't find the variable, search in the `search()` list, until all the .GlobalEnv and packages are searched. If still can't find, an error will occur.

# Scoping Rules - Optimization Example (OPTIONAL)

R uses lexical scoping instead of dynamic scoping.

```r
y <- 10

f <- function(x) {
    y <- 2
    y^2 + g(x)
}

g <- function(x) {
    x*y
}
```
f(3) = 34, g only finds its free variable in defining environment, not the calling environment. Scheme, Perl, Python and Common Lisp all support lexical scoping.

# Coding Standards

# Dates and Times


<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>