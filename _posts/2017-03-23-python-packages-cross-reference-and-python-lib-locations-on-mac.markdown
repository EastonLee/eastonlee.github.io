---
title: Python packages "Cross Reference" and Python Lib Locations on Mac
layout: post
published: true
category: [Python]
---

On Mac, you have many ways to install Python and its packages: pip, conda, brew, port, offical source or binary, etc. But none of them is perfect, sometimes you find you can only install a package through like conda or brew, but after installation, you start your port Python, but can't find that package at all.

<!--more-->

AFAIK it's very hard to install some Python packages via offical .dmg or .pkg files or pip, such as wxPython.

# Bring wxPython home

Let's start from a story. For example, if you need to install wxPython.

```bash
easton@MBP: ~ $ pip search wxpython
wxPython (2.9.1.1)                              - Cross platform GUI toolkit for Python

easton@MBP: ~ $ pip install wxPython
Collecting wxPython
Could not find a version that satisfies the requirement wxPython (from versions: )
No matching distribution found for wxPython
```

This is weird, you can search wxpython but you can't install it. Actually this is [intended](https://groups.google.com/forum/#!topic/wxpython-dev/hDegYu2uoY8), you can still install wxPython_Phoenix via pip by specifying url, by most times that's not what we want.

If you want to install via official binary, I can tell you that's another dead end. First you download binary installer from [official site](https://wxpython.org/download.php#osx), and try to install the carbon version or cocoa version, you will find they all fail! You are told "there was no software found to install" by the installer, that's most ridiculous thing I've ever seen.

OK, just let me tell you the saver: It's brew. Homebrew can handle this nicely, after your `brew install wxpython`, you can verify by `/usr/local/Cellar/python/2.7.13/bin/python -c "import wx;print wx"`. 

So far we have wxPython installed, but let's think what if you don't use `/usr/local/Cellar/python/2.7.13/bin/python`, instead you use some `/usr/bin/python`, `/System/Library/Frameworks/Python.framework/Versions/2.7/bin/python`, `/Users/username/miniconda2/bin/python`, `/opt/local/bin/python`, how can you use a package of Homebrew Python?

Hey easy, for example, you are using `/System/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7`, just append following line into your `/Library/Python/2.7/site-packages/others.pth`

```
/usr/local/lib/python2.7/site-packages/wx-3.0-osx_cocoa
```

or you add it to your `PYTHONPATH` environment variable. Both methods can add your wanted packages to you interpreter's `sys.path`.

# "Cross Reference" between Pythons

Here "Pythons" means same major version Python, doesn't cover situation like between Python2 and Python3.

Like the wxPython, you can add any package from Python A to search path of Python B by modifying Python B's .pth file. You just need to pay attention to the order of "reference", don't mess it up.

# Python Lib Locations on Mac

|                           Location                          |       Description       |
|-------------------------------------------------------------|-------------------------|
| /Library/Python/2.7/site-packages/                          | Mac built-in Python Lib |
| /Users/username/Library/Python/2.7/lib/python/site-packages | user scheme Lib         |
| /usr/local/lib/python2.7/site-packages                      | Homebrew Lib            |
| /opt/local/lib/python2.7/site-packages/                     | MacPorts Lib            |
| ~/[miniconda&#124;anaconda]/lib/python2.7/site-packages/    | Conda Lib               |

Now you don't need to hang on one package manager any more, you can choose suitable package manager to install suitable package, and then "borrow" the package in your preferred Python environment. Life becomes much easier, good luck!