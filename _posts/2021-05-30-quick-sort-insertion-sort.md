---
layout: post
title: "Structural and Generative Recursion"
excerpt: "Looking at the design recipes for 2 common sorting algorithms in Scheme"
date: 2021-05-30
tags:
    - meta
comments: true
---


The code snippets for this post have been adapted from ISL+, a teaching language in Racket to MIT Scheme, from the free e-book [How to Design Programs](https://htdp.org/2020-5-6/Book/index.html#%28part._htdp2e%29)

### Insertion Sort
This is what is called structural recursion. It follows a common template which uses "wishful thinking".
Here, we only solve a trivial case of the problem, rather than the complete problem itself. We then "wish" for the computer to solve the problem itlsef. 
If the trivial case is solved, recursion works it's magic on the entire list without us telling the computer how to do it!


```MIT Scheme
; List-of-numbers -> List-of-numbers
; sorts a list l in increasing order
(define (insertion-sort l)
  (cond
    ((null? l) '())
    (else (insert (car l) (insertion-sort (cdr l))))
  )
)

;; This is our wish list. We haven't defined insert function yet, but if we do it, then our problem is solved.

; Number List-of-numbers -> List-of-numbers
; Helper function that inserts n into the sorted list of numbers l 
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



### Quick Sort
This is an example of generative recursion. Instead of following a simple template like we did with Insertion Sort, here, you need some kind of mathematical insight to come up with such clever algorithm design. 


```MIT Scheme
; [List-of Number] -> [List-of Number]
; produces a sorted version of alon
; assume the numbers are all distinct 
(define (quick-sort alon)
  (cond
    ((null? alon) '())
    (else (let ((pivot (car alon)))
               (append (quick-sort (smallers alon pivot))
                       (list pivot)
                       (quick-sort (largers alon pivot))
               )
          )
    )
  )
)
 
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


; testing the insert sort function with an example
(quick-sort '(72 45 43 29 34))
```




    (29 34 43 45 72)



Notice that the helper functions `largers` and `smallers` still follow the simple template of structural recursion only.


```MIT Scheme

```
