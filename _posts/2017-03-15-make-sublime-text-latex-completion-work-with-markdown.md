---
title: Make Sublime Text LaTex Completion Work with Markdown
layout: post
category: tools
---

When I use Sublime Text to edit Markdown files, I appreciate the Syntax Highlight provided by *Markdown Light* plugin, but just like built-in and other Markdown plugins, they can't provide auto completion for embedded LaTex statements in Markdown files. But it's very common nowadays to mix various branches of Markdown with LaTex-like math equations, it will ease my pain if I can have LaTex auto completion when editing Markdown files.

<!--more-->

Here is the simple solution:

1 Install `LaTexBox` plugin, which gives us LaTex auto completion.

2 Modify `/Packages/LaTeXBox/auto_complete.py`, like this.

    # Line 18
    # Before
    def on_query_completions(self, view, prefix, locations):
        if not view.match_selector(locations[0], "text.tex.latex"):
            return None
    # After
    def on_query_completions(self, view, prefix, locations):
        if not ( view.match_selector(locations[0], "text.tex.latex") or view.match_selector(locations[0], "text.html.markdown")):
            return None
    # Line 38
    # Before
    if view.match_selector(locations[0], "meta.environment.math"):
        r = r + math_commands
    # After
    if view.match_selector(locations[0], "meta.environment.math") or \
        view.match_selector(locations[0], "text.html.markdown"):
        r = r + math_commands

Explanation:

These two modifications expand the processing scope from `tex` to `tex` and `markdown`, and guarantee LaTex syntax will also get auto completion in Markdown files.

Using similar method, you can mix auto completion of any multiple syntaxes. (Caveat: you can't mix Syntax Highlight in this way)