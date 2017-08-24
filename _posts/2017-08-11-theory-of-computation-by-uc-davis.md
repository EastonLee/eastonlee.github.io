---
title: "Theory of Computation by UC Davis"
layout: post
published: true
last_modified_at: 2017-08-11
#image: https://upload.wikimedia.org/wikipedia/commons/c/ce/Rivertree_thirds_md.gif
category: [Automata, Turing Machine, Regular Expression, Regular Language, Pushdown Automata, Intractable Problems, NP-completeness, Course Notes, Theory of Computation, UC Davis]
---

Automata itself is a very interesting thesis and Turing Machines which Automata leads to are breathtakingly beautiful.

<!--more-->

The Finite Automata part is taught by the Stanford University [here](https://lagunita.stanford.edu/courses/course-v1:ComputerScience+Automata+Fall2016/info) but is a little tedious, and the rest part is taught by UC Davis [here](http://web.cs.ucdavis.edu/~rogaway/classes/120/fall12/lectures.html) and videos can be found [here](https://www.youtube.com/playlist?list=PLslgisHe5tBM8UTCt1f66oMkpmjCblzkt), the latter UC Davis course is much easier to understand and a lot of fun.

# Introduction

When developing solutions to real problems, we often confront the limitations of what software can do.
* Undecidable things – no program whatever can do it.
* Intractable things – there are programs, but no fast programs.

# Finite Automata

**What is a Finite Automaton?**
* A formal system.
* Remembers only a finite amount of information.
* Information represented by its state.
* State changes in response to inputs.
* Rules that tell how the state changes in response to inputs are called transitions.

![Finite Automata Example](https://cdn.eastonlee.com/blog/2017-08-17-computer-science-automata-theory-by-stanford/Finite%20Automata%20Example.png)

**Acceptance of Inputs**
* Given a sequence of inputs (input string ), start in the start state and follow the transition from each symbol in turn.
* Input is accepted if you wind up in a final (accepting) state after all inputs have been read.

**Language of an Automaton** 
* The set of strings accepted by an automaton A is the language of A.
* Denoted L(A).
* Different sets of final states -> different languages.
* Example: As designed, L(Tennis) = strings that determine the winner.

**Nondeterminism**
* A nondeterministic finite automaton has the ability to be in several states at once.
* Transitions from a state on an input symbol can be to any set of states.
* Start in one start state.
* Accept if any sequence of choices leads to a final state.
* Intuitively: the NFA always "guesses right."

![Nondeterministic Finite Automata Example](https://cdn.eastonlee.com/blog/2017-08-17-computer-science-automata-theory-by-stanford/Nondeterministic%20Finite%20Automata%20Example.png)

**Equivalence of DFA’s, NFA’s**
* A DFA can be turned into an NFA that accepts the same language.
* If δD(q, a) = p, let the NFA have δN(q, a) = {p}.
* Then the NFA is always in a set containing exactly one state – the state the DFA is in after reading the same input.
* Surprisingly, for any NFA there is a DFA that accepts the same language.
* Proof is the subset construction.
* The number of states of the DFA can be exponential in the number of states of the NFA.
* Thus, NFA’s accept exactly the regular languages.

**Subset Construction**
* Given an NFA with states Q, inputs Σ, transition function $$δ_N$$, state state $$q_0$$, and final states F, construct equivalent DFA with:
* States $$2^Q$$ (Set of subsets of Q).
* Inputs Σ.
* Start state $${q_0}$$.
* Final states = all those with a member of F.
* The transition function δD is defined by: $$δ_D({q_1,...,q_k}, a)$$ is the union over all i = 1,...,k of $$δ_N(q_i, a)$$.

![Subset Construction Example](https://cdn.eastonlee.com/blog/2017-08-17-computer-science-automata-theory-by-stanford/Subset%20Construction%20Example.png)

<!-- 
**Turing-Machine Formalism**
A TM is described by:
1. A finite set of states (Q, typically).
2. An input alphabet (Σ, typically).
3. A tape alphabet (Γ, typically; contains Σ).
4. A transition function (δ, typically).
5. A start state ($$q_0$$, in Q, typically).
6. A blank symbol (B, in Γ-Σ, typically).
    
    All tape except for the input is blank initially.
7. A set of final states (F ⊆ Q, typically).

**Conventions**
* a, b, ... are input symbols.
* ..., X, Y, Z are tape symbols.
* ..., w, x, y, z are strings of input symbols.
* $$\alpha$$, $$\beta$$,... are strings of tape symbols.

**The Transition Function**
* Takes two arguments:

    1. A state, in Q.
    2. A tape symbol in Γ.
* δ(q, Z) is either undefined or a triple of the form (p, Y, D).
* p is a state.
* Y is the new tape symbol. 
* D is a direction, L or R. -->

 
# L2: Regular Languages and Non-Deterministic FSMs 

A DFA is an NFA, but not every NFA is a DFA.

Theorems:
* Every DFA has an equivalent NFA.
* Every NFA has an equivalent DFA.

**Regular Language**

A Language that is recognized by some DFA (DFSM) is called a regular language. A string set like this $$\left\{ A^nB^n where N \ge 0 \right\}$$ can't be recognized by a DFA, thus not a regular language.

Theorems:

* If A and B are regular, then $$A \cup B$$ is regular. Here is [proof](https://youtu.be/_TRUBByaJWg?list=PLslgisHe5tBM8UTCt1f66oMkpmjCblzkt)
* If A and B are regular, then $$A \cap B$$ is regular.
* If A and B are regular, then $$A \circ B$$ is regular. (Concatenation)

# L4: Regular Expression

A regular expression R is:

1. a for some $$a \in \Sigma$$
2. $$\epsilon$$, the empty string
3. $$\Phi$$, the empty set
4. $$(R_1 \cup R_2)$$ where $$R_1$$ and $$R_2$$ are regular expressions
5. $$(R_1 \circ R_2)$$ where $$R_1$$ and $$R_2$$ are regular expressions
6. $$(R_1^*)$$ where $$R_1$$ and $$R_2$$ are regular expressions

* $$R \circ \Phi \equiv \Phi$$ because empty set is equivalent to an NFA which doesn't accept any string.
* $$\Phi^* \equiv \left\{\epsilon \right\}$$ .
* $$R \cup \Phi \equiv R$$ .
* $$R \circ \epsilon \equiv R$$ .
* $$R \cup \epsilon \equiv R$$ if and only if R contains empty string.

Theorem: A language is regular if and only if it is described by some regular expressions.

![Examples of converting a regex to an NFA](https://cdn.eastonlee.com/blog/2017-08-17-computer-science-automata-theory-by-stanford/Examples%20of%20converting%20a%20regex%20to%20an%20NFA.png)

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>