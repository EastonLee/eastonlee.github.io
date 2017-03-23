---
title: Emacs Configuring
layout: post
published: true
category: [tools]
---

* TOC
{:toc}

First, you should know Emacs' common keyboard shortcut.
Then you need to know Emacs Lisp.

# Change to a bright theme

(disable-theme 'spacemacs-dark)

# Make modeline foreground visible

(set-face-foreground 'powerline-active1 "green")
(set-face-foreground 'powerline-inactive1 "white")
https://gist.github.com/TheBB/367096660b203952c162

# Display time in modeline, because Emacs menu is too long, and overlap time area in Mac menu

(display-time-mode)
(setq display-time-mail-string "")
(setq display-time-format "%I:%M")