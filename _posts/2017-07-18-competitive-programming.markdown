---
title: Competitive Programming
layout: post
published: true
category: [Algorithm, Competitive Programming]
---

Competitive Programming can sharpen one's mind, sometimes helps with one's interview. This post is a note of Bjarki Ágúst Guðmundsson's [Competitive Programming course](https://algo.is/competitive-programming-course/).

<!--more-->

# Introduction

**Quickly classify problems**
1. Ad Hoc
2. Complete Search (Iterative/Recursive)
3. Divide & Conquer
4. Greedy (only the original ones)
5. Dynamic Programming (only the original ones)
6. Graph
7. Mathematics
8. String Processing
9. Computational Geometry
10. Some Harder Problems

**Rule of Thumb**
1. $$10^9$$ operations per second
1. $$2^{10} \approx 10^3$$.

**Algorithm Analysis**

| n            | Slowest Accepted Algorithm | Example                                                    |
| $$\le 10$$   | $$O(n!), O(n^6)$$          | Enumerating a permutation                                  |
| $$\le 15$$   | $$O(2^n\times n^2)$$       | DP TSP                                                     |
| $$\le 20$$   | $$O(2^n), O(n^5)$$         | DP + bitmask technique                                     |
| $$\le 50$$   | $$O(n^4)$$                 | DP with 3 dimensions + $$O(n)$$ loop, choosing $$_nC_k=4$$ |
| $$\le 10^2$$ | $$O(n^3)$$                 | Floyd Warshall's                                           |
| $$\le 10^3$$ | $$O(n^2)$$                 | Bubble/Selection/Insert sort                               |
| $$\le 10^5$$ | $$O(nlog_2n)$$             | Merge sort, building a segment tree                        |
| $$\le 10^6$$ | $$O(n), O(log_2n), O(1)$$  | Usually, contest problem have $$n<10^6$$ (to read input)     |




<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>