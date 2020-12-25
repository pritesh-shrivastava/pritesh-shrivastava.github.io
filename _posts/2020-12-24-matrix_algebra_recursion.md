---
layout: single
classes: wide
title: "Matrix algebra operations using recursion"
excerpt: "Implementing basic matrix algebra operations in Scheme using a Jupyter notebook"
date: 2020-12-24
tags:
  - lisp
  - functional programming
  - sicp
  - scheme
comments: true
---


This post is inspired by a couple of [exercises](https://sarabander.github.io/sicp/html/2_002e2.xhtml#Exercise-2_002e37) from the classical book, [SICP](https://sarabander.github.io/sicp/html/index.xhtml). I found them pretty interesting as they were just using recursion & some common list operations to multiply matrices !! I also wanted to try out the Jupyter notebook kernel for [MIT Scheme](https://github.com/joeltg/mit-scheme-kernel).

#### Representation & Problem Statement

Suppose we represent vectors v = ( v<sub>i</sub> ) as sequences of numbers, and matrices m = ( m<sub>ij</sub> ) as sequences of vectors (the rows of the matrix). For example, the matrix

<div align=center>
    <table>
        <tr>
            <td>1</td>
            <td>2</td>
            <td>3</td>
            <td>4</td>
        </tr>
        <tr>
            <td>5</td>
            <td>6</td>
            <td>7</td>
            <td>8</td>
        </tr>
        <tr>
            <td>9</td>
            <td>10</td>
            <td>11</td>
            <td>12</td>
        </tr>
    </table>
</div>


is represented as the sequence `((1 2 3 4) (5 6 7 8) (9 10 11 12))`. With this representation, we can use sequence operations to concisely express the basic matrix and vector operations. 

We will look at the following 4 basic operations on matrices:
- `(dot-product v w)` returns the sum  &sum;<sub>i</sub> = v<sub>i</sub> w<sub>i</sub>  
- `(matrix-*-vector m v)` returns the vector `t` , where t<sub>i</sub> = &sum;<sub>j</sub> m<sub>ij</sub> v<sub>j</sub>  
- `(transpose m)` returns the matrix `n` , where n<sub>ij</sub> = m<sub>ji</sub> 
- `(matrix-*-matrix m n)` returns the matrix `p` , where p<sub>ij</sub> = &sum;<sub>k</sub> m<sub>ik</sub> n<sub>kj</sub> 

Dot product of 2 vectors in this notation can be done by using 2 higher order functions, `map` and `fold`, both of which are implemented using recursion.

Implementation of `map`:
```scheme
(define (map proc items)
  (if (null? items)
      nil
      (cons (proc (car items))
            (map proc (cdr items)))))
```


Implementation of `fold-right`:
```scheme
(define (fold-right op initial sequence) 
   (if (null? sequence) 
       initial 
       (op (car sequence) 
           (fold-right op initial (cdr sequence)))))
``` 

We can use either `fold-left` or `fold-right` for `dot-product`.


```scheme
;; Define dot product of 2 vectors of equal length
(define (dot-product v w)
  (fold-right + 0 (map * v w))
)

;; testing our function
(define vec1 (list 1 2 3) )
(define vec2 (list 1 1 1) )

(dot-product vec1 vec2)
```




    6



Calculating a dot product was really easy with a couple of higher order functions!
Let's work with matrices now. We will now right a function to multiply a matrix and a vector:


```scheme
(define (matrix-*-vector m v)
  (map (lambda (m-row)(dot-product m-row v) ) 
       m)
)

;; testing the function
(define mat1 (list (list 1 0 0) (list 0 1 0) (list 0 0 1)))
(matrix-*-vector mat1 vec1)
```




    (1 2 3)



Let's look at transpose now! For this, we will need to implement a helper function, `accumulate-n`, which is similar to `fold` except that it takes as its third argument a sequence of sequences, which are all assumed to have the same number of elements. 


```scheme
;; Defining helper functions for transpose to 
;; apply the operation op to combine all the first elements of the sequences,
;; all the second elements of the sequences, and so on, 
;; and returns a sequence of the results. 
(define (accumulate-n op init seqs)
  (if (null? (car seqs))
      '()
      (cons (fold-right op init (map car seqs))
            (accumulate-n op init (map cdr seqs))
      )
  )
)

(define (transpose mat)
  (accumulate-n cons '() mat)
)

;; testing transpose
(define mat2 (list (list 1 2 3) (list 4 5 6) (list 7 8 9))) 
(transpose mat2)
```




    ((1 4 7) (2 5 8) (3 6 9))



Now, let's use this transpose function to do matrix multiplication:


```scheme
; Matrix multiplication
(define (matrix-*-matrix m n)
  (let ((n-cols (transpose n)))
    (map (lambda (m-row)(matrix-*-vector n-cols m-row)) 
         m)
  )
)

;; For testing
(matrix-*-matrix mat2 mat1)
```




    ((1 2 3) (4 5 6) (7 8 9))



We can even write recursive procedures to create some special kinds of matrices, for eg, an identity matrix. Here is a recursive procudure to create an identity matrix of length `n`:


```scheme
;; Create an identity matrix of length n
; N -> [List-of [List-of N]]
(define (identityM n)
    (letrec   ;; Documentation for letrec : https://groups.csail.mit.edu/mac/ftpdir/scheme-7.4/doc-html/scheme_3.html
        (
          ;; N -> [List-of [List-of N]]
          (generate-matrix (lambda (row)
            (cond
             ((= row 0) '())
             (else (cons (generate-row row n)
                         (generate-matrix (- row 1)))))))

          ;; N N -> [List-of N]
          (generate-row (lambda (row col)
            ;; col goes from column n to 0
            (cond
             ((= col 0) '())
             (else (cons (if (= row col) 1 0)
                         (generate-row row (- col 1)))))))
        )
        (generate-matrix n)
    )
)

(identityM 3)
```




    ((1 0 0) (0 1 0) (0 0 1))



If you found these functions interesting, I'de definitely encourage to go read SICP. I wrote about why I'm reading SICP [here](https://pritesh-shrivastava.github.io/blog/2020/08/30/sicp-so-far).