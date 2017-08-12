---
title: "Computer Science: Automata Theory by Stanford"
layout: post
published: true
last_modified_at: 2017-08-11
#image: https://upload.wikimedia.org/wikipedia/commons/c/ce/Rivertree_thirds_md.gif
category: [Automata, Turing Machine, Regular Expression, Regular Language, Pushdown Automata, Intractable Problems, NP-completeness, Course Notes]
---

Automata itself is a very interesting thesis and Turing Machines which Automata leads to are breathtakingly beautiful.

<!--more-->

# Introduction

When developing solutions to real problems, we often confront the limitations of what software can do.
* Undecidable things – no program whatever can do it.
* Intractable things – there are programs, but no fast programs.

# Week 1: Finite Automata

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
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>