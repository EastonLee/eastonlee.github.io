---
title: Compiling TensorFlow on Local
layout: post
published: True
category: [TensorFlow, Machine Learning, tools]
---

Today I was trying [a RNN model](https://github.com/fchollet/keras/blob/master/examples/addition_rnn.py) using Keras and TensorFlow, and some warnings just came out:

```
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
```

It seems that TensorFlow installed via PIP doesn't enable some vectorization instructions. I thought these SSE/AVX/FMA instructions will make more use of the potential of my CPU, thus shortening my computation time, so I decided to spend some time compiling my own TensorFlow.

I followed [official document](https://www.tensorflow.org/install/install_sources), finally finished the time-consuming compiling after 40 minutes (pure compiling time).

Then I reran the same program, it turned out no significant speed up, what a shame. Maybe in my case these SSE/AVX/FMA instructions happened not to be used at all? 

Here is the [wheel file](https://drive.google.com/open?id=0Bz8u16o5sSTxSk9VbG5rMlUxODA), I compiled it against Python2.7.13 and TensorFlow-1.0.1. If you are using MacBook Pro 13-inch, Mid 2014, or Intel Core i5 4278U, or CPUs with same ISA, it can save your 40 minutes for compiling. If it's faster than your PIP installed TensorFlow, you can leave a comment here.

To install it:

    pip install tensorflow-1.0.1-cp27-cp27m-macosx_10_12_intel.whl

You may need root access.