---
title: Full Stack Web Development by The Hong Kong University of Science and Technology
layout: post
published: true
category: [Full Stack, Web, HTML, CSS, JavaScript]
---

Finally a non-science course, purely technical one, I can't wait to make my hands dirty on web development. I think every developer whether he/she is front end or back end knows web development more or less, but to make your site/single page application professional and pretty, you have to go through some courses systematically.

Most concepts are straight forward and simple, so I will only take note of those which are tricky and exceptional.

* TOC
{:toc}

# Course 1: HTML, CSS and JavaScript

## Week 1: HTML & CSS

**[Void (empty) elements and self-closing start tags in HTML](http://www.456bereastreet.com/archive/201005/void_empty_elements_and_self-closing_start_tags_in_html/)**

The void elements can only have a start tag since they can't have any content. They must not have an end tags in HTML.
The void elements in HTML 4.01/XHTML 1.0 Strict include `area, base, br, col, hr, img, input, link, meta, param`. HTML 5 currently adds `command, keygen, source` to that list.

The opposite problem is when an element that can have content is empty. If it is not a void element, it must still have an end tag.

## Week 2: Introduction to JavaScript

**Data Types**

* Number
* String
* Boolean
* Other e.g. Object

**Global variable**: if you assign a variable without declaring it with `var` in any place whether in a function or not, the variable automatically become a global variable, and the whole script can access it.

## Week 3: Advanced JavaScript

There are 3 kinds of **for loops**:

1. for
2. for ... in

    like this:

    ```javascript
    var continents = ["Austrilia", "Africa", "Antarctica", "Eurasia", "America"];
    for ( var index in continents) {}
    ```

    remember that `index` is a number, not an object in the `continents` array.

    In other situation like this:

    ```javascript
    var onePerson = {initials: "DR", age: 40, job: "Professor"};
    for ( var property in onePerson ) {}
    ```

    The `property` gives the left hand side of the object `onePerson`.

3. for ... of 

    ```javascript
    var continents = ["Austrilia", "Africa", "Antarctica", "Eurasia", "America"];
    for ( var continent in continents) {}
    ```

    Here the `continent` is each element of the array `continents`, this syntax is equivalent to the Python `for ... in ...`.

**splice** can be used to remove or add or replace some element from or into an array.

* To remove elements at a position: `array.splice(startPosition, quantity)`, and it returns the removed elements.
* To add an element before a position: `array.splice(position, 0, element)`, and it returns empty array `[]`.
* To replace an element: `array.splice(position, quantity, elements)`, and it returns the removed elements.

**array functions**

* `forEach`
* `map`

**Whitespace Nodes**

There may be some text nodes which contains only whitespace, sometimes this is troublesome.

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>