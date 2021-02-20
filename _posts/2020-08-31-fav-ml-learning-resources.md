---
layout: post
title: "Favourite learning resources for ML"
excerpt: "Books, MOOCs and other resources that I would highly recommend"
date: 2020-08-31
tags:
    - meta
comments: true
---


There's no dearth of free (or cheap) online resources for learning data science these days. In fact, with all the hype around AI in the last few years ("sexiest job" ?!?), I feel the only industry that has seen an AI boom is the MOOC industry! My own experience of learning ML, and broadly DS related topics, online has been mostly positive so far, though there have been the odd half baked courses designed to purely make some quick bucks. Here, I'm just listing down my all time favourite MOOCs / books that I thoroughly enjoyed :


- [R4DS by Hadley Wickham](https://r4ds.had.co.nz/) : Without trying to start any language war, let me just say that R is my first love! The tidyverse, (and tidymodels) set of packages share a common design philosophy that makes it very easy to work with any kind of data. The only problem is its speed, R is optimized for humans, not machines. And so, for deployment purposes, I have to end up using Python because either, A) the solution in R is too slow or B) other devs know Python, and they find it easier to integrate it with their existing application. Regardless, R for Data Science is one of the best books written in this field, especially as an introductory material.
    
    
- [fast.ai Course](https://course.fast.ai/) : People regularly tout Andrew Ng's ML & DL courses as the best courses on Coursera, I slightly disagree. While these courses are comprehensive in terms of theory, most programming assignments in these courses are simply "fill-in-the-blank" type tasks in Jupyter notebooks, or worse yet, Octave for the ML course. My favorite introductory DS course in Python is fast.ai's Practical DL for Coders by Jeremy Howard. The fast.ai library is roughly speaking, a wrapper around PyTorch, similar to what Keras does with Tensorflow. You may or may not choose to use fast.ai library in your actual work, but the top down approach used by Jeremy in this course is pretty unique. It's completely free and they've recently released the course with fastai v2 library and a free online [e-book](https://github.com/fastai/fastbook) as well.


- [Feature Engg & Selection book by Max Kuhn](http://www.feat.engineering/) - I'm still in the middle of this book (finished ~60%) but the practical lessons covered in this book can't be missed. We often get more excited about trying out different models for any ML problem, but it's the features that we are feeding to a model that affect the performance much more than out choice of model. After reading any chapter, I often end up thinking how I could've applied this technique to my current projects at work, which is just awesome. The book is completely free & uses R but the concepts can be easily applied in Python as well.

- [ML Engineer Nanodegree](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t) : This is the only paid learning resource I'm listing down here, & Udacity is fairly expensive compared to MOOC platforms in general. I took up this course because of a special COVID related discount on the website, and found it to be a pleasant surpise. The course projects (like [this](https://github.com/pritesh-shrivastava/sentiment-analysis-sagemaker) one) were pretty practical and thorough and taught me a lot of best practises around deploying ML projects. This is one of the older Nanodegree on Udacity that has now been broken down into smaller chunks.

- [Kaggle](https://www.kaggle.com/) : Lastly, Kaggle offers a great place to experiment with the latest techniques on a variety of datasets. The [Kaggle Learn](https://www.kaggle.com/learn/overview) courses are short & crisp tutorials on Jupyter notebooks. You get access to free CPUs & GPUs on the cloud with Kaggle kernels & the forums are pretty welcoming to newcomers. But the best way to learn from Kaggle is to compete in an ongoing competition. I plan to write a more detailed post about my learnings from Kaggle competitions in a separate post altogether. But Kaggle has definitely shaped my learning path significantly in the last couple of years.