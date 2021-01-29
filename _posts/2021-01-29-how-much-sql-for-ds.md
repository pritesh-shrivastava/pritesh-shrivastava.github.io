---
layout: post
title: "How much should one know in SQL for data science"
excerpt: "Most Tech firms include SQL problems in their 1st coding round interviews for data science roles"
date: 2021-01-29
tags:
  - interview
comments: true
---


Do data scientists need to be experts at SQL? 
Not quite. I've been in this field for almost 5 years now and I have rarely, if ever, written an SQL query that's longer than 3 lines. For our typical "small" datasets, ie, data that fits into your computer's memory, I've found `dplyr` to be much more efficient and practical. If you're working on Python, `pandas` does a pretty good job at data manipulation there as well. 
So for all kinds of filters, joins or aggregations, SQL is usually the 3rd best alternative.

When we talk about "big data" applications, `Apache Spark` has really made things pretty easy for us. Yes, you do have popular tools like `Google BigQuery` and `SparkSQL` that use SQL, you can perform the same queries in a more expressive manner using `sparklyr` or `pyspark` depending on your programming language. You can even code natively in `Scala` for Spark related applications for better flexibility and performance. 

Still, most data science interview questions at Tech firms usually involve atleast 1 SQL based problem in their initial coding round of interview. One reason for that could be to setup a minimum baseline. Yes, you can always have better tools to write your queries but you should be good enough to write that same query in good old SQL, even though it might look long and ugly and complicated.

Since I don't use a lot of advanced SQL in my day to day job, I've found it a good practise to revise certain topics that I've seen to be asked repeatedly in many tech interviews. The best resource I've found online is this free course on Kaggle - [Advanced SQL](https://www.kaggle.com/learn/advanced-sql). If you feel a bit rusty with SQL syntax, you can try their [introductory SQL](https://www.kaggle.com/learn/intro-to-sql) coure as well. These 2 courses have covered almost every SQL problem I've seen in an interview and they're fairly short and hands on. 

