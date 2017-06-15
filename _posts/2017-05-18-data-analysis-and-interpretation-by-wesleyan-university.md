---
title: Data Analysis and Interpretation by Wesleyan University
layout: post
published: true
category: [Data Science]
---

<!--more-->

* TOC
{:toc}

# COURSE 1: Data Management and Visualization

# COURSE 2: Data Analysis Tools

**Central Limit Theorem**

As long as adequately large samples and an adequately large number of samples are used from a population, the distribution of the statistics will be normally distributed.

**Hypothesis Testing**

Definition: Assessing the evidence provided by the data, in favor of or against each hypothesis about the population.

Methods:
* ANOVA - Analysis of Variance
* X2 - Chi-Square of Independence


1. Specify the null($$h_0$$), and the alternate ($$h_a$$) hypothesis
2. Choose a sample
3. Assess the evidence
4. Draw conclusions

**p value**

Often noted as α, will be compared with "significance level of a test", usually taken for 0.05. If p-value < α (0.05), the data provides significant evidence against the null hypothesis ($$H_0$$), so we reject the null hypothesis and accept the alternate hypothesis ($$H_a$$).

p value is also known as "Type One Error Rate", means the number of times out of 100 we would be wrong if we reject the null hypothesis.

**Bivariate Statistical tools**

* ANOVA - Analysis of Variance
* X2 - Chi-Square of Independence
* r - Correlation Coefficient

How to choose a statistical test?

* C->Q: if you have categorical explanatory and quantitative response, choose ANOVA
* C->C: if you have categorical explanatory and response, choose X2
* Q->Q: if you have quantitative explanatory and response, choose Pearson Correlation
* Q->C: if you have categorical explanatory and quantitative response, you need to categorize your explanatory variable with only two levels then use the Chi-Square of Independence as your inferential test.

# COURSE 3: Regression Modeling in Practice

# COURSE 4: Machine Learning for Data Analysis

# COURSE 5: Data Analysis and Interpretation Capstone

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>