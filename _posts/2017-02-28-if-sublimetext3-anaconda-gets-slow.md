---
title: If Sublimetext3 Anaconda Gets Slow
layout: post
category: tools
---

Recently I find Anaconda plugin for Sublime Text 3 has some performance issue. If your buffer is wider than like 20,000 characters, or higher than 5,000 lines, Anaconda will fail, the Autocomplete, Docstring and Goto_definition function will have several seconds' lag or will take forever before they can show up, sometimes Anaconda's jsonserver.py process even takes 100% CPU usage on one core and fails quiting when Sublime Text 3 main process is terminated.

<!--more-->

Update: This performance issue has been fixed by author of Anaconda.

To solve this problem, you have two options:

1. Try disabling jedi model built in Anaconda. The built-in jedi in Anaconda is a little old. You can find Anaconda's package in directory `$HOME/Library/Application Support/Sublime Text 3/Packages/anaconda`, in subfolder anaconda_lib, rename jedi folder to prevent it from being loaded. Then install new jedi into Python's site-packages using pip or similar Python package managers.

2. Reduce your file size if possible. Since Anaconda has performance problem, we can avoid that by controlling file size or split one file to multiple then merge them. 

If this problem still happens, consider disabling Anaconda, or switching to other plugins or Python editors.