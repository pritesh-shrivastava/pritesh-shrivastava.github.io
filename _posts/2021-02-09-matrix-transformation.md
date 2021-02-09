---
layout: post
title: "How matrices transform space"
excerpt: "An intuitive way to look at matrix vector multiplication"
date: 2021-02-09
tags:
comments: true
---

## Matrices are objects that operate on vectors

When we mulitply a matrix with an n-dimensional vector, it essentially transforms the vector in n-dimensional space!
This wonderful video by 3 Blue 1 Brown expplains this concept through beautiful visualizations - [YouTube link](https://youtu.be/kYB8IZa5AuE)

We can take a 2x2 matrix A to consider how it will transform the 2-D space using Python.

```python
import numpy as np
%matplotlib inline
from matplotlib import pyplot
```

We'll also need a helper script to plot gridlines in a 2-D space.
```python
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

Now, let's take an example of a 2x2 matrix A.

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

So the columns a matrix, in fact, just give us the location of where our original basis vectors will land. 
Here's a screenshot of the above 3B1B video that shows how we can understand this 2-D transformation by simply looking at the columns of the matrix A. 

![ss](/assets/images/matrices-as-linear-transformations-of-space_files/Screenshot_3b1b.png)

Using our helper script, we can better understand this transformation by looking at the gridlines of our 2-D space before and after transformation.
```python
plot_linear_transformation(A)
```


![png](/assets/images/matrices-as-linear-transformations-of-space_files/matrices-as-linear-transformations-of-space_13_0.png)


Let's look at another example using another 2x2 matrix M and see how M transforms the 2-D space!

```python
M = np.array([[1,2], [2,1]])
plot_linear_transformation(M)
```


![png](/assets/images/matrices-as-linear-transformations-of-space_files/matrices-as-linear-transformations-of-space_14_0.png)

This also reminds us why "linear algebra" is called "linear", because A) the origin does not move and B) the gridlines remain parallel straight lines after transformation!

We can now start looking at matrices as not just a collection of numbers in row / column format, but as objects that we can use to transform space the way we want it. And we just need to look at the columns a matrix to understnad where the original basis vectors will land after the transformation!

Here's another example with the matrix N. 
```python
N = np.array([[1,2],[-3,2]])
plot_linear_transformation(N)
```


![png](/assets/images/matrices-as-linear-transformations-of-space_files/matrices-as-linear-transformations-of-space_15_0.png)


Now, let's look at the some special kinds of matrices. We can rotate the 2-D space by 90 degrees counter clockwise by just rotating the original basis vectors in that direction.
```python
rotation = np.array([[0,-1], [1,0]])
plot_linear_transformation(rotation)
```


![png](/assets/images/matrices-as-linear-transformations-of-space_files/matrices-as-linear-transformations-of-space_16_0.png)


We can even shear the 2-D space by designing our transformation matrix accordingly.
```python
shear = np.array([[1,1], [0,1]])
plot_linear_transformation(shear)
```


![png](/assets/images/matrices-as-linear-transformations-of-space_files/matrices-as-linear-transformations-of-space_17_0.png)


We can scale the X-axis by 2x and Y-axis by 0.5x using the below matrix.
```python
scale = np.array([[2,0], [0,0.5]])
plot_linear_transformation(scale)
```


![png](/assets/images/matrices-as-linear-transformations-of-space_files/matrices-as-linear-transformations-of-space_18_0.png)


Interestingly, we can compose multiple transformations of 2-D space by mulitplying our transformation matrices together.
So, applying the above shear and rotation transformations will be the same when done sequentially OR just using the product of the 2 matrices as one single transformation.

```python
plot_linear_transformation(shear@rotation)
```


![png](/assets/images/matrices-as-linear-transformations-of-space_files/matrices-as-linear-transformations-of-space_19_0.png)



```python
plot_linear_transformations(rotation, shear)
```


![png](/assets/images/matrices-as-linear-transformations-of-space_files/matrices-as-linear-transformations-of-space_20_0.png)


However, the order of these transformations is important as matrix multiplication is not commutative!
```python
plot_linear_transformations(shear, rotation)
```


![png](/assets/images/matrices-as-linear-transformations-of-space_files/matrices-as-linear-transformations-of-space_21_0.png)


This concept of space transformation also gives a new meaning to a matrix inverse. The inverse of a matrix simply reverses the transformation of space to its original state.
```python
M = np.array([[1,2], [2,1]])
M_inv = np.linalg.inv(M)
plot_linear_transformations(M, M_inv)
```


![png](/assets/images/matrices-as-linear-transformations-of-space_files/matrices-as-linear-transformations-of-space_22_0.png)


Degenerate matrices will reduce the dimension of the space. For eg, the below matrix D reduces the 2-D space into a 1-D line!

```python
D = np.array([[-2,-1], [1,0.5]])
plot_linear_transformation(D)
```


![png](/assets/images/matrices-as-linear-transformations-of-space_files/matrices-as-linear-transformations-of-space_24_0.png)


## Applications in Image Processing and Computer Vision

So where can we use this concept of matrix vector multiplication other than in thinking about abstract spaces ?
One very important application of this concept can be see in image processing applications. We can consider an image to be a collection of vectors. Let's consider grayscale images for simplicity, then a grayscale image basically is just a collection of vectors in 2-D space (location of grayscale pixels can be considered a 2-D vector). And we can multiply each pixel vector with a given matrix to transform the entire image!

```python

```

**PS:** If you find this concept of matrix multiplication exciting, I would urge you to check out the YouTube playlist, [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab), from 3Blue1Brown. There is also a free MOOC available from the George Washington University [here](https://openedx.seas.gwu.edu/courses/course-v1:GW+EngComp4+2019/about) and paid one on Coursera available [here](https://www.coursera.org/learn/linear-algebra-machine-learning) 



This blog post is written on a Jupyter notebook hosted on Kaggle [here](https://www.kaggle.com/priteshshrivastava/matrices-as-linear-transformations-of-space)
