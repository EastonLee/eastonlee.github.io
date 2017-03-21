---
title: Interactively run Python projects in Sublime Text, using handy SublimeREPL
layout: post
category: tools
---

Sublime Text rocks, but can you run Python projects interactively when using Build System?

Sublime Text is super editor, it's fast, pretty and extensible, so I use it a lot to edit any possible text files, and develop Python projects. Vim, Emacs and PyCharm are also cool, but Vim is "evil", Emacs is too "guru", Pycharm is so slow, I decide to bury them and move to Sublime Text. 

In fact I've used Sublime Text for many years mainly as an ad-hoc and rich featured text editor, I never touched its plugins because I had powerful IDEs like PyCharm. But PyCharm is so bloated and takes too much resources including CPU, RAM and disk, let alone its start time. 

Sublime Text avoids all those weaknesses and is equally powerful, but it does have some flaws. When you want to run your Python scripts or a whole project, you may need Sublime's Build System. The Build System invokes external compiler, interpreter or project management system to run the build task, and displays the result in its result panel, however the result panel can only give running result but not take user input, which is ridiculous. You may wonder whether input is so important, my answer is yes, especially when it comes to pdb (The Python Debugger).

How to hack this ****? Well, you can try SublimeREPL. SublimeREPL provides REPL interpreter in Sublime Text view for many languages, so you can run current Python file interactively. But that's all it can give, every time you run a file a new tab is opened and code tab is overlaid, of course now you get REPL, but that's still not cool, I also want to see source code. Calm down, Sublime Text plugins can sort things out, if they can't, we write a new plugin.

Let's make things clear:

1. I want SublimeREPL to take place of built-in Build System to run Python files.

2. I want SublimeREPL to display in a new view, instead of the same view with source code, so I need a plugin that is good at Sublime Text layout to split one window into two groups.

    2.1 When only one Sublime Text group exists, create new one, then move REPL view into new group.

    2.2 When there are two or more groups, just move the REPL tab to the next one.

    2.3 When the new group contains only REPL tab, and you close that tab, the group should be destroyed too.

3. If built-in Build System is able to invoke SublimeREPL and layout plugin, that will be good. Otherwise we have to write our own plugin to invoke Build System then layout plugin.

Problem 1 is easy to solve. First, create a new Build System, configurate it like this, then save it with extension "sublime-build", then you can find this new Build System in top menu.

```json
    {
        "target": "run_existing_window_command", 
        "id": "repl_python_run",
        "selector": "source.python",
        "file": "config/Python/Main.sublime-menu",
    }
```

Then the new Build System will call REPL command.
![](http://7u2owr.com1.z0.glb.clouddn.com/repl_in_same_group.gif)
Problem 2, the vanilla Sublime Text layout is useful but far from handy, so you need to install enhanced layout plugin -- [Origami](https://github.com/SublimeText/Origami), BTW Origami calls group as pane, you just need to know they are the same thing. Look at what features Origami brings: 1) save and restore layouts, 2) shortcut to resize groups, 3) auto close empty group, 4) create new group if necessary. I love it, these features will make the entire Sublime Text layout more organizable, and problem 2 is no longer pain. Detailed config and code will be seen bellow.

Problem 3 is a little tricky, I tried to custom the Build System (above mentioned sublime-build file), but if I replace the target field with my own plugin command, how should I deal with the other fields like "id", "selector" and "file"? I read SublimeREPL's code, these fields are necessary but I'm not familiar with how Build System passes these to SublimeREPL, so a more plausible way is to write a new plugin, which we let call build command and layout command (most functions of Sublime Text and its plugins are implemented and encapsulated in commands, press CTRL+` and try them in console).

Now we need to write a Sublime Text plugin to chain all things up. That's not very hard, here are [some document](http://docs.sublimetext.info/en/latest/extensibility/plugins.html) to begin with. Some important concept includes *Conventions for Command Names, Window Commands and Text Commands*, notice the name you define a command class is different from the name you call, and the command file's name doesn't matter (a poor .py file), don't mess up. Below is my command file.

```python
import sublime
import sublime_plugin


class RunPythonReplCommand(sublime_plugin.TextCommand): 
    def run(self, edit):
        view = self.view
        view.window().run_command('build')
        # segment 1 and 2 are equivalent but segment 1 changes original view's focus
        # after moving repl to new group
        # segment 1
        # view.window().run_command('carry_file_to_pane', {"direction": "right"})
        # segment 2
        view.window().run_command('travel_to_pane', {"direction": "right"})
        view.window().run_command('travel_to_pane', {"direction": "left"})
        view.window().run_command('move_to_neighboring_group')
        view.window().run_command('zoom_pane', {"fraction": 0.5})
```

First we need to specify our newly created Build System in Sublime Text's top menu, then when we call `build` command, it will invoke SublimeREPL, a new tab will appear on the right hand side of source code view. The commands `travel_to_pane, travel_to_pane, zoom_pane` are provided by Origami, `move_to_neighboring_group` is built-in. After SublimeREPL tab appears, Origami will try to visit right side's group, if no group is there, Origami will create one, then Origami move focus back, then we move the SublimeREPL tab to the right group, and I specify the right group take 50% width, you can change as you need.

Finally we tame Sublime Text even better. Now you can bind the new command to a keyboard shortcut, and watch Sublime Text dance like a charm.
![](http://7u2owr.com1.z0.glb.clouddn.com/repl_in_new_group.gif)
