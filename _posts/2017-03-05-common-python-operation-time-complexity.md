---
title: Common Python Operation Time Complexity
layout: post
---

Python enable us to perform advanced operation in very expressive way, meanwhile covers many users' eyes from underlying implement details. If the performance of your application plays a critical role, please always keep in mind the time complexity of these common operations.

The following table is an important cheat sheet to memorize to keep your applications behave.
[Python official page](https://wiki.python.org/moin/TimeComplexity)

## List

|   Operation    | Average Case | Amortized Worst Case |
| -------------- | ------------ | -------------------- |
| Copy           | O(n)         | O(n)                 |
| Append         | O(1)         | O(1)                 |
| Insert         | O(n)         | O(n)                 |
| Get Item       | O(1)         | O(1)                 |
| Set Item       | O(1)         | O(1)                 |
| Delete Item    | O(n)         | O(n)                 |
| Iteration      | O(n)         | O(n)                 |
| Get Slice      | O(k)         | O(k)                 |
| Del Slice      | O(n)         | O(n)                 |
| Set Slice      | O(k+n)       | O(k+n)               |
| Extend         | O(k)         | O(k)                 |
| Sort           | O(n log n)   | O(n log n)           |
| Multiply       | O(nk)        | O(nk)                |
| x in s         | O(n)         |                      |
| min(s), max(s) | O(n)         |                      |
| Get Length     | O(1)         | O(1)                 |

## collections.deque

| Operation  | Average Case | Amortized Worst Case |
| ---------- | ------------ | -------------------- |
| Copy       | O(n)         | O(n)                 |
| append     | O(1)         | O(1)                 |
| appendleft | O(1)         | O(1)                 |
| pop        | O(1)         | O(1)                 |
| popleft    | O(1)         | O(1)                 |
| extend     | O(k)         | O(k)                 |
| extendleft | O(k)         | O(k)                 |
| rotate     | O(k)         | O(k)                 |
| remove     | O(n)         | O(n)                 |

## set

|             Operation             |                  Average case                 |     Worst Case     |                   notes                    |
| --------------------------------- | --------------------------------------------- | ------------------ | ------------------------------------------ |
| x in s                            | O(1)                                          | O(n)               |                                            |
| Union s                           | t                                             | O(len(s)+len(t))   |                                            |
| Intersection s&t                  | O(min(len(s), len(t))                         | O(len(s) * len(t)) | replace "min" with "max" if t is not a set |
| Multiple intersection s1&s2&..&sn | (n-1)*O(l) where l is max(len(s1),..,len(sn)) |                    |                                            |
| Difference s-t                    | O(len(s))                                     |                    |                                            |
| s.difference_                     | update(t)                                     | O(len(t))          |                                            |
| Symmetric Difference s^t          | O(len(s))                                     | O(len(s) * len(t)) |                                            |
| s.symmetric_difference_update(t)  | O(len(t))                                     | O(len(t) * len(s)) |                                            |

## dict

|  Operation  | Average Case | Amortized Worst Case |
| ----------- | ------------ | -------------------- |
| Copy        | O(n)         | O(n)                 |
| Get Item    | O(1)         | O(n)                 |
| Set Item    | O(1)         | O(n)                 |
| Delete Item | O(1)         | O(n)                 |
| Iteration   | O(n)         | O(n)                 |
