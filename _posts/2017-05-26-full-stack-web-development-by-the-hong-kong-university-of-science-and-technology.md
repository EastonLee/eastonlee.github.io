---
title: Full Stack Web Development by The Hong Kong University of Science and Technology
layout: post
published: true
category: [Full Stack, Web, HTML, CSS, JavaScript, BootStrap, Angular, Node.js]
---

Finally a non-science course, purely technical one, I can't wait to make my hands dirty on web development. I think every developer whether he/she is front end or back end knows web development more or less, but to make your site/single page application professional and pretty, you have to go through some courses systematically.

Most concepts are straightforward and simple, so I will only take note of those which are tricky and exceptional.

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

# Course 2: Front-End Web UI Frameworks and Tools

## Week 1: Front-end Web UI Frameworks Overview: Bootstrap

The `container` class uses fixed width depending on the screen size, `container-fluid` uses the full width of screen.

These 3 lines of code must be contained in `head` section prior to enabling Bootstrap.

```html
    <meta charset="uft-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1">
```

**Media Queries** are the CSS technology to apply some styles based on the size of the **viewport** e.g.

```css
@media screen and (min-width:600px){
    /* CSS customized for desktop */
}
```

**Viewport**

```html
<meta name="viewport" content="width=device-width, initial-scale=1">
```

The `viewport` meta tag:

* Ensure the screen width is set to the device width and content is rendered with the width in mind.
* Designing the websites responsive to the size of the viewport (Bootstrap grid system).

If you don't specify classes for some screen size, for example, you only specify `col-sm-5` for an element, then on medium and large screen, Bootstrap will choose the specification for smaller screen, which is `col-sm-5` class in this case.

## Week 2: Bootstrap CSS Components; Week 3: Bootstrap Javascript Components; Week 4: Web Tools

I'm tired of Bootstrap's tedious class names, I won't put details here about them.

# Course 3: Front-End JavaScript Frameworks: AngularJS

## Week 1: Front-End Javascript Frameworks: AngularJS Overview

`ng-init` is used to 
* Evaluate an expression
* Initialize a JavaScript variable

## Week 2: Web tools: Grunt and Gulp

Two commonly used automation tools: Grunt and Gulp. The comparison between them:
* Grunt uses Configuration over Code
* Gulp uses Code over Configuration

**Dependency Injection** involves four roles:
* The service
* The client
* The interfaces
* The injector

**Dependency Annotation in Angualar**

* Inline array annotation

    ```javascript
    module.controller('MenuController', ['$scope', 'menuFactory', function($scope, menuFactory){
    }]);
    ```

* $inject property annotation

    ```javascript
    var MenuController = function($scope, menuFactory) {
    };
    MenuController.$inject = ['$scope', 'menuFactory'];
    module.controller('MenuController',  MenuController);
    ```

* Implicit annotation

    ```javascript
    module.controller('MenuController', function($scope, menuFactory) {
    }]);
    ```
**Angular Services**

* Substitudable objects wired together using DI
* Allow organizing and sharing code across an app
* Lazily instantiated
* Singletons

**Five functions that declare services**

* serveice()
* factory()
* provider()
* constant()
* value()

# Course 4: Server-side Development with NodeJS

## Week 1: Introduction to Server-side Development

**JavaScript** doesn't provide standard library, **CommanJS API** fills this gap by defining API for common application needs. It defines a module format, **Node** follows module specification.

* Each file is its own module.
* The *module* variable gives access to the current module definition in a file.
* The *module.exports* determines the export of the current module.
* The *require* function is used to import a module.

When you import **core modules or external modules**, do like this `require("module_name")`. When you are importing an external module, Node.js will look for the external module in `./node_modules/ or ../node_modules/ or ../../node_modules/ ...` until the module is found.

**Stateless server**: 

* Server side should not track the client side state:

    Every request is a new request from the client side.

* Client side should track its own state:

    * E.g. using cookies; client side database
    * Every request must include sufficient information so server side can serve up requested information
    * Client side MVC setup

## Week 2: Data, Data, Where art Thou Data?

**Four broad categories of NoSQL Databases**

* Document databases (e.g., MongoDB)
* Key-value databases (e.g., Redis)
* Column family databases (e.g., Cassadra)
* Graph databases (e.g., Neo4J)

## Week 3: Halt! Who goes there?

## Week 4: Backend as a Service (BaaS)