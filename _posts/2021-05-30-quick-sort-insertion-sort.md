---
layout: post
title: "Structural and Generative Recursion"
excerpt: "Looking at the design recipes for 2 common sorting algorithms in Scheme"
date: 2021-05-30
tags:
    - lisp
    - functional programming
comments: true
---


Recursion is a central component in functional programming, and one of my favourity programming concepts. In this post, we are going to look at two distinct ways in which recursive functions can be designed. 

We'll using the examples of 2 very common sorting algorithms for numbers, ie, insertion sort and the quick sort. The code snippets used below have been adapted the free e-book [How to Design Programs](https://htdp.org/2020-5-6/Book/index.html#%28part._htdp2e%29) and translated to MIT Scheme (from a teaching language created in Racket). 


### Insertion Sort and Structural Recursion
This is the most common type of recursive functions that we see in functional programming languages. 
It follows a common template which is very similar to wishful thinking! It really intrigued me when I learnt about it back in high school and still intrigues me to this day. 

"Structural" recursion uses the underlying data "structures" naturally. Here, we only ever solve a very trivial case of the problem, and then, as if by magic, recursion solves the complete problem without us telling the computer how to do it!

Let's look at the `insertion-sort` function that takes a list of numbers as its argument. 

```scheme
; sorts a list l in increasing order
; List-of-numbers -> List-of-numbers
(define (insertion-sort l)
  (cond
    ((null? l) '())
    (else (insert (car l) (insertion-sort (cdr l))))
  )
)
```
We have not yet defined the `insert` function so far. This is "wishful thinking" !!
If a function `insert` existed that would insert a given no into a sorted list at the right position, that would solve our problem! And so we just used in our recursive definition of `insertion-sort`. Notice that all we have really just solved the trivial case of sorting an empty list. And then, we are relying on recursion to sort the entire list by just naturally traversing the data structure, ie, the list.

```scheme
; We now define the 'insert' function from our wish list, after which our problem is solved.
; Helper function that inserts a number n into a sorted list of numbers l at its correct position 
; Number List-of-numbers -> List-of-numbers
(define (insert n l)
  (cond
    ((null? l) (cons n '()))
    (else (if (<= n (first l))
              (cons n l)
              (cons (car l) (insert n (cdr l)))
          )
    )
  )
)

; testing the insert sort function with an example
(insertion-sort '(72 45 43 29 34))
```




    (29 34 43 45 72)



### Quick Sort and Generative Recursion
We see how structural recursion has a very natural design template. 
In generative recursion though, instead of following a simple template like we did with Insertion Sort, we need some kind of mathematical insight into the problem itself to come up with the algorithm design. For quick sort, it's ofcourse the "divide and conquer" strategy. 

```scheme
; [List-of Number] -> [List-of Number]
; produces a sorted version of list l, assume the numbers are all distinct 
(define (quick-sort l)
  (cond
    ((null? l) '())
    (else (let ((pivot (car l)))
               (append (quick-sort (smallers l pivot))
                       (list pivot)
                       (quick-sort (largers l pivot))
               )
          )
    )
  )
)
```

Here, again we used 2 functions `smallers` and `largers` that haven't been defined yet, using wishful thinking once again!

```scheme
; returns the sub list of numbers from a given list of numbers that are greater than the given number n
; [List-of Number] Number -> [List-of Number]
(define (largers alon n)
  (cond
    ((null? alon) '())
    (else (if (> (car alon) n)
              (cons (car alon) (largers (cdr alon) n))
              (largers (cdr alon) n))
    )
  )
)
 
; returns the sub list of numbers from a given list of numbers that are smaller than the given number n 
; [List-of Number] Number -> [List-of Number]
(define (smallers alon n)
  (cond
    ((null? alon) '())
    (else (if (< (car alon) n)
              (cons (car alon) (smallers (cdr alon) n))
              (smallers (cdr alon) n))
    )
  )
)
```

Now, our function `quick-sort` is complete, and we can test it similarly as above. Notice that the helper functions `largers` and `smallers` still follow the simple template of structural recursion only.

```scheme
; testing the insert sort function with an example
(quick-sort '(72 45 43 29 34))
```




    (29 34 43 45 72)


It's generally always easier to frame a recursive function as structural recursion, rather than to wait for an "AHA" moment to design a generative recursive function. However, sometimes the extra ingenuity can be worth it. For eg., the time completexity of quick sort is `O(n * log n)`, which is much better than our simple insertion sort function `O(n * n)`.


If you find the field of functional programming interesting, I would encourage you to check out the excellent books [SICP](https://sarabander.github.io/sicp/html/index.xhtml#SEC_Contents) or [HtDP](https://htdp.org/2020-5-6/Book/index.html#%28part._htdp2e%29). The [Youtube lectures](https://www.youtube.com/playlist?list=PLE18841CABEA24090) from MIT by the authors of the SICP are also a great learning resource. I've written a couple other blog posts on functional programming in Lisp dialects, which you might wanna check out [here](https://pritesh-shrivastava.github.io/tags/lisp).