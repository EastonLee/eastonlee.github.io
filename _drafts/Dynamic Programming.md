# Can dynamic programming solve all the solvable problems? If yes, how so?
https://www.quora.com/Can-dynamic-programming-solve-all-the-solvable-problems-If-yes-how-so
No. Dynamic Programming is applicable when a problem satisfies the principle of optimality (see Bellman equation - Wikipedia).
>   “A problem is said to satisfy the Principle of Optimality if the subsolutions of an optimal solution of the problem are themselves optimal solutions for their subproblems.”

# GWU Dynamic Programming (This is very hard to understand)
http://www2.seas.gwu.edu/~ayoussef/cs6212/dynamicprog.html

## I. Perspective

Dynamic programming is an optimization technique.
### Greedy vs. Dynamic Programming :

* Both techniques are optimization techniques, and both build solutions from a collection of choices of individual elements.
* The greedy method computes its solution by making its choices in a serial forward fashion, never looking back or revising previous choices.
* Dynamic programming computes its solution bottom up by synthesizing them from smaller subsolutions, and by trying many possibilities and choices before it arrives at the optimal set of choices.
* There is no a priori litmus test by which one can tell if the Greedy method will lead to an optimal solution.
* By contrast, there is a litmus test for Dynamic Programming, called The Principle of Optimality

### Divide and Conquer vs. Dynamic Programming:

* Both techniques split their input into parts, find subsolutions to the parts, and synthesize larger solutions from smalled ones.
* Divide and Conquer splits its input at prespecified deterministic points (e.g., always in the middle)
* Dynamic Programming splits its input at every possible split points rather than at a pre-specified points. After trying all split points, it determines which split point is optimal.

## II. Principle of Optimality

Definition: A problem is said to satisfy the Principle of Optimality if the subsolutions of an optimal solution of the problem are themesleves optimal solutions for their subproblems.

Examples:

The shortest path problem satisfies the Principle of Optimality.

This is because if a,x1,x2,...,xn,b is a shortest path from node a to node b in a graph, then the portion of xi to xj on that path is a shortest path from xi to xj.

The longest path problem, on the other hand, does not satisfy the Principle of Optimality. Take for example the undirected graph G of nodes a, b, c, d, and e, and edges (a,b) (b,c) (c,d) (d,e) and (e,a). That is, G is a ring. The longest (noncyclic) path from a to d to a,b,c,d. The sub-path from b to c on that path is simply the edge b,c. But that is not the longest path from b to c. Rather, b,a,e,d,c is the longest path. Thus, the subpath on a longest path is not necessarily a longest path.

```
Formulate a greedy method for the Matrix Chain Problem, and prove by a counter example that it does not necessarily lead to an optimal solution.
```

## IV. First Application: The Matrix Chain Problem

## V. Second Application: The All-Pairs Shortest Path Problem

## VI. Third Application: Optimal Binary Search Trees

# What are the characteristics of the problems to be solvable via dynamic programming
https://www.researchgate.net/post/What_are_the_characteristics_of_the_problems_to_be_solvable_via_dynamic_programming

You can solve problem with dynamic programming with some approach, when you can find in the problem a recursive structure (Recursive formula). When this problem has a optimal structure as well, and of course with you recursive structure, it has cases bases.

it's very popular use two approach: backward approach and forward. These is a bottom-up (p.e: Fibonacci, coin problem, 0/1 knapsack) and top-bottom(tsp, factory problem, max-flow), respectively.

# Dynamic programming and memorization: bottom-up vs top-down approaches
https://stackoverflow.com/questions/6164629/dynamic-programming-and-memoization-bottom-up-vs-top-down-approaches

# Principle of Optimality

# Bellman equation

# Optimal Substructure

# Overlapping subproblems
https://en.wikipedia.org/wiki/Overlapping_subproblem

# Divide and conquer
