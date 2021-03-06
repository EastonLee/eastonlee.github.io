---
title: Full Stack Web Development by The Hong Kong University of Science and Technology
layout: post
published: true
last_modified_at: 2017-07-09
category: [Full Stack, Web, HTML, CSS, JavaScript, BootStrap, Angular, Node.js]
---

Finally a non-science course, purely a technical one, I can't wait to make my hands dirty on web development. I think every developer whether he/she is front end or back end knows web development more or less, but to make your site/single page application professional and pretty, you have to go through some courses systematically.

<!--more-->

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

Grunt and Gulp automate many things using tasks. Grunt is based on files while Gulp is a streaming build system.

## Week 3: Single Page Applications

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

## Week 4: Client-Server Communication and Angular Testing

# Course 4: Multiplatform Mobile App Development with Web Technologies

## week 1: Hybrid Mobile App Development Frameworks: An Introduction

## week 2: More Ionic CSS and JavaScript

## week 3: Deploying your App

## week 4: Accessing Native Capabilities of Devices: Cordova and ngCordova

# Course 5: Server-side Development with NodeJS

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

**Node.js callbacks**

A brief introduction of callback from [nodejitsu](https://docs.nodejitsu.com/articles/getting-started/control-flow/what-are-callbacks/):

> This works just fine and is very typical in other development environments. However, if fetchData takes a long time to load the data (maybe it is streaming it off the drive or the internet), then this causes the whole program to 'block' - otherwise known as sitting still and waiting - until it loads the data. Node.js, being an asynchronous platform, doesn't wait around for things like file I/O to finish - Node.js uses callbacks. A callback is a function called at the completion of a given task; this prevents any blocking, and allows other code to be run in the meantime.
> ```javascript
> function asyncOperation ( a, b, c, callback ) {
>   // ... lots of hard work ...
>   if ( /* an error occurs */ ) {
>     return callback(new Error("An error has occured"));
>   }
>   // ... more work ...
>   callback(null, d, e, f);
> }
> 
> asyncOperation ( params.., function ( err, returnValues.. ) {
>    //This code gets run after the async operation gets run
> });
> ```

## Week 2: MongoDB and Mongoose

**Four broad categories of NoSQL Databases**

* Document databases (e.g., MongoDB)
* Key-value databases (e.g., Redis)
* Column family databases (e.g., Cassadra)
* Graph databases (e.g., Neo4J)

**MongoDB**

With MongoDB, you can manipulate data with less explicit code.

Run `mongo` in terminal to enter MongoDB REPL.

```javascript
db
use conFusion
db
db.help()
db.dishes.insert({ name: "Uthapizza", description: "Test" });
db.dishes.find().pretty();
var id = new ObjectId();
id.getTimestamp();
```

**Mongoose**

Mongoose enables to use schema in MongoDB

```javascript
var mongoose = require('mongoose');
var Schema = mongoose.Schema;
require('mongoose-currency').loadType(mongoose);
var Currency = mongoose.Types.Currency;

// create a schema
var promotionSchema = new Schema({
    name: {
        type: String,
        required: true,
        unique: true
    },
    description: {
        type: String,
        required: true
    },
    image: {
        type: String,
        required: true
    },
    label: {
        type: String,
        required: true,
        default: ""
    },
    price: {
        type: Currency
    }
}, {
    timestamps: true
});

// the schema is useless so far
// we need to create a model using it
var Promotions = mongoose.model('Promotion', promotionSchema);

// make this available to our Node applications
module.exports = Promotions;
```

## Week 3: User Authentication

**Cookie + Session Authentication**

* Cookie set on the client side by the server
* Cookie is used as a storage for session ID that is used as an index into server-side storage of session information.

**Why Token-Based Authentication**

* Session authentication becomes a problem when we need stateless servers and scalability
* Mobile application platforms have a hard time handling cookies/sessions
* Sharing authentication with other applications not feasible
* Cross-origin resource sharing (CORS) problem
* Cross-site request forgery (CSRF)

**Token-based Authentication**

1. User requests access with their username and password
2. Server validates credentials
3. Server creates a signed token and sends it to the client
– Nothing stored on the server
4. All subsequent requests from the client should include the token
5. Server verifies the token and responds with data if validated

## Week 4: Backend as a Service (BaaS)

**MongoDB and Relations**
* NoSQL databases like MongoDB do not explicitly support relations like the SQL databases
* All documents are normally expected to be self- contained
* However you can store references to other documents within a document by using ObjectIds
* Mongoose does not have joins

**Mongoose Population works like Joining of Relational Database**

```javascript
//Modified Comment Schema
var commentSchema = new Schema({
        rating: { 
            type: Number, 
            min: 1, 
            max: 5, 
            required: true 
        },
        comment:  { 
            type: String, 
            required: true 
        },
        postedBy: {
            type: mongoose.Schema.Types.ObjectId,
            ref: 'User'
        }
    }, 
    {timestamps: true});

// Populating the Documents
Dishes.find({})        
.populate('comments.postedBy')        
.exec(function (err, dish) {
    if (err) throw err;
    res.json(dish);
});
```

**SSL/TLS Handshake**

![SSL/TLS Handshake by Prof. Jogesh	K.	Muppala](https://eastonlee.b0.upaiyun.com/blog/2017-05-26-full-stack-web-development-by-the-hong-kong-university-of-science-and-technology/SSL:TLS%20handshake.png)

**OAuth 2 Roles**

* Resource owner: You, the user that authorizes a client application to access their account
* Client Application: Application (website or app) that wants access to the resource server to obtain information about you
* Resource Server: Server hosting protected data (e.g., your personal information)
* Authorization Server: Server that issues an access token to the client application to request resource from the resource server

**OAuth 2 Tokens**
* Access token: allows access to user data by the client application
    * Has limited lifetime
    * Need to be kept confidential
    * Scope: parameter used to limit the rights of the access token
* Refresh token: Used to refresh an expired access token

![authorization code grand approach by Prof. Jogesh	K.	Muppala](https://eastonlee.b0.upaiyun.com/blog/2017-05-26-full-stack-web-development-by-the-hong-kong-university-of-science-and-technology/authorization%20code%20grand%20approach.png)