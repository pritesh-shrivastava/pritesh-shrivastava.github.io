---
layout: post
title: "Fun with Haskell"
date: 2020-09-13
comments: true
---


My interest in Haskell arose while studying functional programming in [Lisp](https://pritesh-shrivastava.github.io/blog/2020/08/30/sicp-so-far). While Lisp is not used as widely today, languages like Haskell, Scala & Clojure are fairly prominent. Haskell seemed to be the most unique & interesting to me as it is a pure & lazy FP language. I also really enjoyed some [quirky](https://youtu.be/SqWDAo1Jnyc) yet [intriguing](https://www.youtube.com/watch?v=re96UgMk6GQ) videos by Simon Peyton Jones on Haskell & its history. So I wanted to give it a try & started reading this funky book called "Learn You a Haskell" which is[freely available online](http://learnyouahaskell.com/). I'm roughly halfway into the book & wanted to talk about some cool Haskell features I've learned so far.

## Haskell reads like math

Haskell's list comprehensions tend to remind me a lot of my high school math books. For eg., to take a list of numbers from 1 to 10 & multiply them by 2, you simply write :  
    `[x*2 | x <- [1..10]]` 

The ease with which you can express mathematical problems in Haskell so succintly just blows me away sometimes! Here's a one liner to find the no of right triangles where each side has an integer length less than or equal to 10 :  
    `rightTriangles = [(a,b,c) | c <- [1..10], b <- [1..10], a <- [1..10], a^2 + b^2 == c^2]`  

Here's another one line function to calculate the length of a list :  
    `length xs = sum [1 | _ <- xs]`

The `_` symbol means we don't care about the value of the specific list element

## Haskell has type inference

While a function like `factorial n = product [1..n]` will work perfectly fine in Haskell, we can add typeclasses for its arguments which make the function more readable :  

    factorial :: Integer -> Integer  
    factorial n = product [1..n]



## Recursion feels so natural

Haskell is built for recursion. Here's a simple function to reverse a list, using the `++` list append operator. `[]` refers to the empty list which is the base case when doing recursion on lists :  
```
reverse [] = []  
reverse (x:xs) = reverse' xs ++ [x]
```

Let's take a loot at the popular quicksort algorithm to sort lists in Haskell :  
```
quicksort [] = []  
quicksort (x:xs) =   
    smallerSorted ++ [x] ++ biggerSorted
    where   smallerSorted = quicksort [a | a <- xs, a <= x]  
            biggerSorted = quicksort [a | a <- xs, a > x]
```

## Pattern Matching rocks

We can rewrite the factorial function above in a recursive style as well, starting with the base case first, and then writing the inductive case :  
```
factorial 0 = 1  
factorial n = n * factorial (n - 1)
```

This feels more natural, and more math-y! The `length` function from above can also be re-written like this :  
```
length :: [a] -> Int
length [] = 0  
length (_:xs) = 1 + length' xs 
```

Here, `a`(pronounced as alpha) is a polymorphic data type, ie, it can have any type, but the function `length'` will always return a value of type `Int`. 

## Build using Higher order functions

Haskell comes with built-in support for a lot of high level functions like map, filter, fold (foldl, foldr), zip, etc.


To find the largest number under 100,000 that's divisible by 3829, we can make use of Haskell's laziness as :  
```
largestDivisible = head (filter p [100000,99999..])  
    where p x = x `mod` 3829 == 0
```
Although we pass an infinite list, due to Haskell's laziness, evalutation will stop once we find the `head` ie first element of the list that satisfies the filter.

Here's another example using map, to find the sum of all odd squares that are smaller than 10,000 :  
`sumOddSq = sum (takeWhile (<10000) (map (^2) [1, 3..]) )`


This was a quick summary of some interesting programming ideas that I've seen in Haskell so far. Although I know I've barely scratched the surface here. I'm yet to study more advanced concepts like monads, typeclasses, monoids & dealing with laziness. I'll try & share my learnings of these topics as I go along. 

If you're interested in functional programming, I'll highly recommend the [Programming Languages](https://www.coursera.org/learn/programming-languages) MOOC by Prof Dan Grossman. While the course uses Standard ML to teach statically typed functional programming (Part A), most concepts were taught in such a generalized manner that I could easily apply them to Haskell as well !!