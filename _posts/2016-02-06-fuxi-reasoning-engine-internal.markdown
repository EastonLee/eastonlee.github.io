---
title: FuXi (reasoning engine) internal
layout: post
category: ComBrain
---

# What is FuXi?
[FuXi](https://github.com/RDFLib/FuXi/blob/master/docs/Tutorial.rst) is a multi-modal, logical reasoning system for the semantic web. Its primary capability is as a a SPARQL 1.1 RIF Core Entailment implementation. The results in the previous link to the SPARQL 1.1 test result show some of the semantics it supports.

FuXi has these features:

* Forward-chaining Simple Example
* Programmatic Equivalent
* Magic Set Method
* Backward-chaining inference
* SPARQL Entailment / Mediation over Remote Endpoints

[1](#1)
Pychinko is a python implementation of the classic Rete algorithm which provides the inferencing capabilities needed by an Expert System. Part of Pychinko works ontop of cwm / afon out of the box. However, it's Interpreter only relies on rdflib to formally represent the terms of an RDF statement.

FuXi only relies on Pychinko itself, the N3 deserializer for persisting N3 rules, and rdflib's Literal and UriRef to formally represent the corresponding terms of a Pychinko Fact. FuXi consists of 3 components (in addition to a 4RDF model for Versa queries):

## I. FtRdfReteReasoner

Uses Pychinko and N3RuleExtractor to reason over a scoped 4RDF model.

## II. N3RuleExtractor

Extracts Pychinko rules from a scoped model with statements deserialized from an N3 rule document

## III. 4RDF N3 Deserializer

The rule extractor reverses the reification of statements contained in formulae/contexts as performed by the N3 processor. It uses three Versa queries for this

Using the namespace mappings:
* n3r --> http://www.w3.org/2000/10/swap/reify#
* log --> http://www.w3.org/2000/10/swap/log#

Extract ancendent statements of logical implications

    distribute(
      all() |- log:implies -> *,
      '.',
      '. - n3r:statement -> *'
    )

Extract implied / consequent statements of logical implications

    distribute(
      all() - log:implies -> *,
      '.',
      '. - n3r:statement -> *'
    )

Extract the terms of an N3 reified statement

    distribute(
      <statement>,
      '. - n3r:subject -> *',
      '. - n3r:predicate -> *',
      '. - n3r:object -> *'
    )

The FuXi class provides methods for performing scoped Versa queries on a model extended via inference or on just the inferred statements:

For example, take the following fact document deserialized into a model:


    @prefix : <http://foo/bar#> .
    :chimezie :is :snoring .

Now consider the following rule:

    @prefix ex: <http://foo/bar#> .
   {?x ex:is ex:snoring} => {?x a ex:SleepingPerson} .

Below is a snapshot of Fuxi perforing the Versa query “type(ex:SleepingPerson)” on a model extended by inference using the above rule:

![](http://copia.ogbuji.net/files/FuXi/test-fuxi.png)


# What technique does FuXi use?
FuXi is based on Pychinko,  to match and fire the specified rules[r](http://copia.ogbuji.net/files/FuXi/README.txt).

Using FuXi, it takes all the facts from the current query context (which may or may not be scoped), the rules from the <ruleGraph> scope and invokes/executes the Rete reasoner.

# What is Pychinko?
[2](#2)Fuxi depends on Pychinko, rdflib, and CWM (just the 1.82 tarball is sufficient and suggested).
Pychinko's interpreter only uses CWM for it's N3 log/math/os function implementations
and rdflib for it's Literal and URIRef classes (for representing the correspondig terms of an
RDF statement for purposes of comparing them when matching rules).  See Yarden Katz's comments on 
the original FuXi http://copia.ogbuji.net/blog/2005-05-29/FuXi

The dependent libraries can be downloaded from:

 - http://www.mindswap.org/~katz/pychinko/downloads/pychinko-0.1.tar.gz
 - http://infomesh.net/2001/cwm/cwm1.82.tar.gz
 - http://rdflib.net/stable/

# What is SPARQL 1.1?

RDF is a directed, labeled graph data format for representing information in the Web. This specification defines the syntax and semantics of the SPARQL query language for RDF. SPARQL can be used to express queries across diverse data sources, whether the data is stored natively as RDF or viewed as RDF via middleware. SPARQL contains capabilities for querying required and optional graph patterns along with their conjunctions and disjunctions. SPARQL also supports extensible value testing and constraining queries by source RDF graph. The results of SPARQL queries can be results sets or RDF graphs.

# What is RIF?
The Rule Interchange Format (RIF) Working Group was chartered by the World Wide Web Consortium in 2005 to create a standard for exchanging rules among rule systems, in particular among Web rule engines. RIF focused on exchange rather than trying to develop a single one-fits-all rule language because, in contrast to other Semantic Web standards, such as * RDF, OWL, and SPARQL *, it was immediately clear that a single language would not satisfy the needs of many popular paradigms for using rules in knowledge representation and business modeling. But even rule exchange alone was recognized as a daunting task. Known rule systems fall into three broad categories: first-order, logic-programming, and action rules. These paradigms share little in the way of syntax and semantics. Moreover, there are large differences between systems even within the same paradigm.


# What is OWL 2?
[4](#4)

* OWL Full = Classical first order logic (FOL)
* OWL-DL = Description logic
* N3 rules ~= logic programming (LP) rules
* SWRL ~= DL + LP

[5](#5)

## Relationship to OWL 1
OWL 2 adds new functionality with respect to OWL 1. Some of the new features are syntactic sugar (e.g., disjoint union of classes) while others offer new expressivity, including:

* keys;
* property chains;
* richer datatypes, data ranges;
* qualified cardinality restrictions;
* asymmetric, reflexive, and disjoint properties; and
* enhanced annotation capabilities

OWL 2 also defines three new profiles [OWL 2 Profiles] and a new syntax [OWL 2 Manchester Syntax]. In addition, some of the restrictions applicable to OWL DL have been relaxed; as a result, the set of RDF Graphs that can be handled by Description Logics reasoners is slightly larger in OWL 2.

# What is RDF?
RDF 1.1 Semantics

W3C Recommendation 25 February 2014

http://www.w3.org/TR/2014/REC-rdf11-mt-20140225/

RDF Semantics

W3C Recommendation 10 February 2004

http://www.w3.org/TR/2004/REC-rdf-mt-20040210/

# What's New in RDF 1.1?
[What's New in RDF 1.1](http://www.w3.org/TR/rdf11-new/):

## Datasets

RDF 1.1 introduces the concept of RDF Datasets. An RDF Dataset is a collection of RDF Graphs. SPARQL 1.1 [SPARQL11-OVERVIEW](https://www.w3.org/TR/rdf11-new/#bib-SPARQL11-OVERVIEW) also defines the concept of an RDF Dataset, but the definition in RDF 1.1 differs slightly in that RDF 1.1 allows RDF Graphs to be identified using either an IRI or a blank node. More information is available in RDF 1.1 Concepts and Abstract Syntax.

## Datatypes

A table of RDF-compatible XSD datatypes has been added to RDF 1.1 Concepts and Abstract Syntax. Any XSD datatypes not represented in this table are incompatible with RDF. The following XSD 1.1 datatypes were added to the list of RDF-compatible datatypes:

    xsd:duration
    xsd:dayTimeDuration
    xsd:yearMonthDuration
    xsd:dateTimeStamp

Support for rdf:XMLLiteral support is now optional. Technically, support for any individual datatype is optional and therefore may not be present in a given implementation. RDF-conformant specifications may require specific datatype maps.

## New Serialization Formats

RDF 1.1 introduces a number of new serialization formats. RDF 1.1 Concepts and Abstract Syntax makes it clear that RDF/XML is no longer the only recommended serialization format; RDF itself should be considered to be the data model (the abstract syntax), not any particular serialization.
![](https://www.w3.org/TR/rdf11-new/serialization-formats.png)

# What is N3?
[3](#3)
N3 is short for Notation 3. Notation 3 provides the notion of a "formula", and syntax for its use in conjunction with RDF statements. There is not, to my knowledge, a clear consensus about what a formula actually is or denotes, beyond being a "set of statements". A "context" is described as being a relationship between a statement and the formula to which it belongs. (Interestingly, John McCarthy declined to offer a definition of context in his paper on formalizing context.)

However, it seems that this idea is intended to address the some problems like those introduced above, and maybe more. Example uses of formulae in N3 include things like these, where braces `"{...}"` are used to enclose a formula:

`{some-statements} a :FalseHood`

meaning that it is asserted that the conjunction of the statements in the formula is not true. This is an extreme case of some statements being unreliable.

`{some-statements} :implies {some-other-statements}`

meaning that in some circumstance where the first set of statements are all true, the second set of statements are also all true. This could be applied in any of the circumstances noted above.

    :Lois :accepts { 
    :Superman rdf:type :StrongPerson . 
    :ClarkKent rdf:type :WeakPerson . }

meaning that in the context of what Lois knows and believes, Superman is strong. But, not posessing knowledge that Clark Kent is the same person as Superman, she believes that he is weak. Someone in posession of the knowledge that they are the same person would have to conclude that Clark Kent is strong. This corresponds to the third case noted above.

The introduction of contexts in the description of Notation 3 seems to be very much associated with the introduction of primitives for expressing full first order logic. This note has a more modest goal, namely the scoping of asserted truths to some collection of statements (formula), and a consequent limitation on the conclusions that may be drawn from some rule of deduction (whose means of definition is not specified).

# Why Extended RDF graph
[3](#3)
The notion of an RDF graph is extended to include the concept of a formula node. A formula node may occur wherever any other kind of node can appear. Associated with a formula node is an RDF graph that is completely disjoint from all other graphs; i.e. has no nodes in common with any other graph. (It may contain the same labels as other RDF graphs; because this is, by definition, a separate graph, considerations of tidiness do not apply between the graph at a formula node and any other graph.)

This is intended to map the idea of `"{ N3-expression }"` that is used by N3 into an RDF graph upon which RDF semantics is defined.

So, for basic RDF, we have the following abstract syntax for a graph, following a style suggested by John McCarthy :

    graph(g)      = isSet(g) & eachMember(g,statement)
    statement(s)  = isTriple(s) &
                    nonLiteral(sub(s)) & 
                    isUriref(prop(s)) & 
                    anyNode(obj(s))
    nonLiteral(n) = isUriref(n) | isBlank(n)
    anyNode(n)    = nonLiteral(n) | isLiteral(n)

where isSet(s), isTriple(t), isUriref(n), isBlank(n), isLiteral(n) are primitive functions that recognize the elementary syntactic structures and components in an RDF graph, eachMember(s,f) is true if the function f yields true when applied to each member of the set s, and sub(t), prop(t) and obj(t) are selectors that return the subject, property and object respectively of a triple.

This is extended to account for formula nodes by extending the definition of nonLiteral(n):

```nonLiteral(n) = isUriref(n) | isBlank(n) | isFormula(n)```

where isFormula(n) is a new primitive function that recognizes a formula node.

# What is 4RDF
The [4Suite 4RDF](https://www.w3.org/2001/sw/wiki/4RDF) is an **outdated** open-source platform for XML and RDF processing implemented in Python with C extensions.

# References
<a name='1'>1 </a>[FuXi - Versa / N3 / Rete Expert System](http://copia.posthaven.com/fuxi-versa-n3-rete-expert-system)

<a name='2'>2 </a>[FuXi README](http://copia.ogbuji.net/files/FuXi/README.txt)

<a name='3'>3 </a>[Circumstance, provenance and partial knowledge, Limiting the scope of RDF assertions](http://www.ninebynine.org/RDFNotes/UsingContextsWithRDF.html)

<a name='4'>4 </a>[OWL, DL and Rules
](http://www.csee.umbc.edu/courses/graduate/691/spring14/01/notes/14_rules/14_owl_rules.ppt.pdf)

<a name='5'>5 </a>[OWL 2 Web Ontology Language Document Overview (Second Edition)](https://www.w3.org/TR/owl2-overview/)

<a name='6'>6 </a>[SPARQL Query Language for RDF](https://www.w3.org/TR/rdf-sparql-query/)

[Contexts, and Scopes, and Provenance, Oh My!](http://copia.posthaven.com/contexts-and-scopes-and-provenance-oh-my)

[Tutorial - A short description of using FuXi](https://github.com/RDFLib/FuXi/blob/master/docs/Tutorial.rst)