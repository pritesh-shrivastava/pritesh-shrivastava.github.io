---
layout: post
title: "How much should one know in SQL for data science"
excerpt: "Most tech firm interviews include SQL problems for DS roles, so how should you prepare for them?"
date: 2021-01-29
tags:
  - interview
  - meta
comments: true
---


Do data scientists need to be experts at SQL? 
Not quite. I'd say that an intermediate level of SQL knowledge is good enough and I'll explain what this "intermediate" level is in more detail below.

For our typical "small" datasets, ie, data that fits into your computer's memory, I've found `dplyr` to be much more efficient and practical for data manipulation. If you're working on Python, `pandas` does a pretty good job at data manipulation there as well. So for all kinds of filters, joins or aggregations, SQL is usually the 3rd best alternative.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">import pandas as better_sql </p>
&mdash; Chris Albon (@chrisalbon) <a href="https://twitter.com/chrisalbon/status/1319349424145924096?ref_src=twsrc%5Etfw">October 22, 2020</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script> 

When we talk about "big data" applications, `Apache Spark` has really made things pretty easy now. You can run queries on distributed systems in a more expressive manner using `sparklyr` or `pyspark` depending on your programming language. You can even code natively in `Scala` for Spark related applications for better flexibility and performance. 

So the only area where SQL really seems to be irreplaceable is data extraction (and ofcourse, data definition), ie, working directly with the databases. And most data science interview questions at Tech firms usually involve atleast 1 SQL based problem in their initial coding round of interview to setup a minimum baseline. Yes, you can always have better tools to write your data manipulation queries but you should still be able to work with good old SQL to pull data from a real world database.

In addition to the introductory SQL clauses (WHERE, GROUP BY, ORDER BY, HAVING, etc.), I've seen that if you understand JOINs, UNIONs and [analytic functions](https://www.kaggle.com/alexisbcook/analytic-functions) in SQL, you should be good enough to solve most problems thrown at you during interviews or in your day-to-day job. Out of these topics, JOINs and analytic functions seem to be the most popular interview topics. Unlike aggregate functions like SUM or MAX which return a single value, analytic functions operate OVER sets of rows & can return sets of rows using PARTITION BY, ORDER BY or window frame clauses.

The best resource I've found online that covers these topics well is this course on Kaggle - [Advanced SQL](https://www.kaggle.com/learn/advanced-sql). If you feel a bit rusty with SQL syntax, you can try their [introductory SQL](https://www.kaggle.com/learn/intro-to-sql) coure as well. They're fairly short, hands on and most of all, free!

Again, this is based on my personal experience so far. If you feel I'm missing some important topics in the above list or have a different opinion, let me know in the comments below. I'll be talking about another common topic featuring in data science interviews at big tech firms, data structures and algorithms, soon.