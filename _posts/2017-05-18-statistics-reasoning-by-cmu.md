---
title: Statistics Reasoning by CMU
layout: post
published: true
category: [statistics, probability]
---

There are two version of this course in CMU, the other one is "Probability & Statistics", I prefer this version because the "probability" part in this version acts as a "bridge" to the inference section thus making easier to understand and grab the big picture.

* TOC
{:toc}

# UNIT 1

## The Big Picture

**Producing data** is choosing a sample and collecting data from it.

**Inference** is that we use what we've discovered about our sample to draw conclusion about our population.

![Big Picture of statistics](https://oli.cmu.edu/repository/webcontent/0dc371000ae1c686566a243ea7db95b7/_u1_intro_stats_online/webcontent/inference.png)

**Example**

At the end of April 2005, a poll was conducted (by ABC News and the Washington Post) for the purpose of learning the opinions of U.S. adults about the death penalty.

1. Producing Data: A (representative) sample of 1,082 U.S. adults was chosen, and each adult was asked whether he or she favored or opposed the death penalty.

2. Exploratory Data Analysis (EDA): The collected data were summarized, and it was found that 65% of the sampled adults favor the death penalty for persons convicted of murder.

3. and 4. Probability and Inference: Based on the sample result (of 65% favoring the death penalty) and our knowledge of probability, it was concluded (with 95% confidence) that the percentage of those who favor the death penalty in the population is within 3% of what was obtained in the sample (i.e., between 62% and 68%). The following figure summarizes the example:

![](https://oli.cmu.edu/repository/webcontent/0dc371000ae1c686566a243ea7db95b7/_u1_intro_stats_online/webcontent/big_picture_example.png)

# UNIT 2: Exploratory Data Analysis:

## Module 4: Examining Distributions

**Distribution** means:

1. What values the variable takes
2. How often the variable takes those values

**Categorical distribution** often uses **pie-chart and bar-chart**. **Quantitative distribution** often uses **histogram, stemplot, boxplot**.

When describing a distribution of quantitative variable, you need to include **shape (symmetry, or skewed left, skewed right, Peakedness/modality—the number of peaks/modes the distribution has), center, spread (range), outliers**.

There is an interesting kind of plot sililar to histogram, **stemplot**.To make a stemplot:

1. Separate each observation into a stem and a leaf.
1. Write the stems in a vertical column with the smallest at the top, and draw a vertical line at the right of this column.
1. Go through the data points, and write each leaf in the row to the right of its stem.
1. Rearrange the leaves in an increasing order.

![make a stemplot](https://oli.cmu.edu/repository/webcontent/0dc371000ae1c686566a243ea7db95b7/_u2_summarizing_data/_m1_examining_distributions/webcontent/eda_examining_distributions_best_actress_stemplot.jpg)

**Measures of center**

1. The three main numerical measures for the center of a distribution are the mode, mean ($$\bar{x}$$), and the median (M). The mode is the most frequently occurring value. The mean is the average value, while the median is the middle value.
1. The **mean** is very sensitive to outliers (because it factors in their magnitude), while the **median** is resistant to outliers.
1. The mean is an appropriate measure of center only for symmetric distributions with no outliers. In all other cases, the median should be used to describe the center of the distribution.

**Three most commonly used measures of spread**

* Range
* Inter-quartile range (IQR)
* Standard deviation

**The 1.5(IQR) Criterion for Outliers**

An observation is considered suspected outlier if it is below Q1-1.5IQR or above Q3+1.5IQR.

**Several possibilities about outliers**

1. Even though it is an extreme value, if an outlier can be understood to have been produced by essentially the same sort of physical or biological process as the rest of the data, and if such extreme values are expected to eventually occur again, then such an outlier indicates something important and interesting about the process you're investigating, and it should be kept in the data.
1. If an outlier can be explained to have been produced under fundamentally different conditions from the rest of the data (or by a fundamentally different process), such an outlier can be removed from the data if your goal is to investigate only the process that produced the rest of the data.
1. An outlier might indicate a mistake in the data (like a typo, or a measuring error), in which case it should be corrected if possible or else removed from the data before calculating summary statistics or making inferences from the data (and the reason for the mistake should be investigated).

**Choosing Numerical Summaries**

Use $$\bar{x}$$ (the mean) and the standard deviation as measures of center and spread only for reasonably symmetric distributions with no outliers.

Use the five-number summary (which gives the median, IQR and range) for all other cases.

## Module 5: Examining Relationships

**For possibilities for role-type classification**

1. Categorical explanatory and quantitative response
1. Categorical explanatory and categorical response
1. Quantitative explanatory and quantitative response
1. Quantitative explanatory and categorical response

![role-type classification table](https://oli.cmu.edu/repository/webcontent/0dc371000ae1c686566a243ea7db95b7/_u2_summarizing_data/_m2_examining_relationships/webcontent/relationships_overview1.gif)

# UNIT 3: Producing Data:

## Module 6: Sampling

## Module 7: Designing Studies

# UNIT 4: Probability:

## Module 8: Introduction (Probability)

## Module 9: Random Variables

## Module 10: Sampling Distributions

# UNIT 5: Inference:

## Module 11: Introduction (Inference)

## Module 12: Estimation

## Module 13: Hypothesis Testing

## Module 14: Inference for Relationships

## Module 15: Inference for Relationships Continued

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>