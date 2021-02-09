---
layout: post
title: "How matrices transform space"
excerpt: "An intuitive way to look at matrix vector multiplication"
date: 2021-02-09
tags:
comments: true
---


We'll use some common Python libraries to look at matrix-vector multiplication in a new light.
```python
import numpy as np
%matplotlib inline
from matplotlib import pyplot
```

We'll also need a helper script to plot gridlines in a 2-D space.
```python
## Helper script for plotting  gridlines and vectors
## Source : https://github.com/engineersCode/EngComp4_landlinear
from urllib.request import urlretrieve
URL = 'https://go.gwu.edu/engcomp4plot'  
urlretrieve(URL, 'plot_helper.py')
```




    ('plot_helper.py', <http.client.HTTPMessage at 0x7fc7a1c3f0d0>)



Importing functions (`plot_vector`, `plot_linear_transformation`, `plot_linear_transformations`) from the helper script

```python
from plot_helper import *
```

## Matrices are objects that operate on vectors

When we mulitply a matrix with an n-dimensional vector, it essentially transforms the vector in n-dimensional space!
This wonderful video by 3 Blue 1 Brown expplains this concept through beautiful visualizations - [YouTube link](https://youtu.be/kYB8IZa5AuE)

We can take a 2x2 matrix A to consider how it will transform the 2-D space.


```python
A = np.array([[3, 2],
              [-2, 1]])
```

Let's just start with the basis vectors, `i_hat` and `j_hat` in 2-D coordinate system, and see how matrix multiplication transforms these 2 basis vectors.

```python
i_hat = np.array([1, 0])
j_hat = np.array([0, 1])
```

How does the matrix A transform `i_hat` via multiplication ?

```python
A @ i_hat
```




    array([ 3, -2])

This is just the 1st column of matrix A.

How does the matrix A transform `j_hat`?

```python
A @ j_hat
```




    array([2, 1])

Similary, multiplication of A with `j_hat` just gives the second column of matrix A.

How the matrix A transforms the 2-D space?
 
Here's a screenshot of the above 3B1B video that shows how we can understand this 2-D transformation by simply looking at the columns of the matrix A. 

![ss](/assets/images/matrices-as-linear-transformations-of-space_files/Screenshot_3b1b.png)


```python
plot_linear_transformation(A)
```


![png](/assets/images/matrices-as-linear-transformations-of-space_files/matrices-as-linear-transformations-of-space_13_0.png)



```python
M = np.array([[1,2], [2,1]])
plot_linear_transformation(M)
```


![png](/assets/images/matrices-as-linear-transformations-of-space_files/matrices-as-linear-transformations-of-space_14_0.png)



```python
N = np.array([[1,2],[-3,2]])
plot_linear_transformation(N)
```


![png](/assets/images/matrices-as-linear-transformations-of-space_files/matrices-as-linear-transformations-of-space_15_0.png)



```python
rotation = np.array([[0,-1], [1,0]])
plot_linear_transformation(rotation)
```


![png](/assets/images/matrices-as-linear-transformations-of-space_files/matrices-as-linear-transformations-of-space_16_0.png)



```python
shear = np.array([[1,1], [0,1]])
plot_linear_transformation(shear)
```


![png](/assets/images/matrices-as-linear-transformations-of-space_files/matrices-as-linear-transformations-of-space_17_0.png)



```python
scale = np.array([[2,0], [0,0.5]])
plot_linear_transformation(scale)
```


![png](/assets/images/matrices-as-linear-transformations-of-space_files/matrices-as-linear-transformations-of-space_18_0.png)



```python
plot_linear_transformation(shear@rotation)
```


![png](/assets/images/matrices-as-linear-transformations-of-space_files/matrices-as-linear-transformations-of-space_19_0.png)



```python
plot_linear_transformations(rotation, shear)
```


![png](/assets/images/matrices-as-linear-transformations-of-space_files/matrices-as-linear-transformations-of-space_20_0.png)



```python
plot_linear_transformations(shear, rotation)
```


![png](/assets/images/matrices-as-linear-transformations-of-space_files/matrices-as-linear-transformations-of-space_21_0.png)



```python
M = np.array([[1,2], [2,1]])
M_inv = np.linalg.inv(M)
plot_linear_transformations(M, M_inv)
```


![png](/assets/images/matrices-as-linear-transformations-of-space_files/matrices-as-linear-transformations-of-space_22_0.png)


Degenerate


```python
D = np.array([[-2,-1], [1,0.5]])
plot_linear_transformation(D)
```


![png](/assets/images/matrices-as-linear-transformations-of-space_files/matrices-as-linear-transformations-of-space_24_0.png)


## Applications in Image Processing and Computer Vision


```python

```

**PS:** If you find this concept of matrix multiplication exciting, I would urge you to check out the YouTube playlist, [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab), from 3Blue1Brown. There is also a free MOOC available from the George Washington University [here](https://openedx.seas.gwu.edu/courses/course-v1:GW+EngComp4+2019/about) and paid one on Coursera available [here](https://www.coursera.org/learn/linear-algebra-machine-learning) 



This blog post is written on a Jupyter notebook hosted on Kaggle [here](https://www.kaggle.com/priteshshrivastava/matrices-as-linear-transformations-of-space)
