```python
from PIL import Image 
import requests

## Sample image URL
url = 'https://images.pexels.com/photos/2850833/pexels-photo-2850833.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940'
```


```python
# Open image 
im = Image.open(requests.get(url, stream=True).raw)
#im = Image.open("../input/lineartransformationimages/pexels-cristina-andrea-alvarez-cruz-2850833.jpg")
plt.imshow(im)
```




    <matplotlib.image.AxesImage at 0x7fa0e20d8910>




![png](linear-transformations-of-images-with-matrices_files/linear-transformations-of-images-with-matrices_2_1.png)



```python
## Grayscale
im = im.convert('LA')
im
```




![png](linear-transformations-of-images-with-matrices_files/linear-transformations-of-images-with-matrices_3_0.png)




```python
b, h = im.size
b, h
```




    (1300, 1300)




```python
def linear_transform(trans_mat, b_new = b, h_new = h):
    '''
    Effectively mulitplying each pixel vector by the transformation matrix
    PIL uses a tuple of 1st 2 rows of the inverse matrix
    '''
    Tinv = np.linalg.inv(trans_mat)
    Tinvtuple = (Tinv[0,0],Tinv[0,1], Tinv[0,2], Tinv[1,0],Tinv[1,1],Tinv[1,2])
    return im.transform((int(b_new), int(h_new)), Image.AFFINE, Tinvtuple, resample=Image.BILINEAR) 
```

Scaling the image to half the size


```python
T = np.matrix([[1/2, 0, 0],
               [0, 1/2, 0],
               [0, 0, 1]])

trans = linear_transform(T, b/2, h/2)
trans
```




![png](linear-transformations-of-images-with-matrices_files/linear-transformations-of-images-with-matrices_7_0.png)




```python
T = np.matrix([[1/4, 0, 0],
               [0, 1/2, 0],
               [0, 0, 1]])

trans = linear_transform(T, b/4, h/2)
trans
```




![png](linear-transformations-of-images-with-matrices_files/linear-transformations-of-images-with-matrices_8_0.png)



Rotation by 45 degree counter clockwise


```python
mat_rotate = (1/ np.sqrt(2)) * \
    np.matrix([[1, -1, 0],
               [1, 1, 0],
               [0, 0, np.sqrt(2)]])

trans = linear_transform(mat_rotate)
trans
```




![png](linear-transformations-of-images-with-matrices_files/linear-transformations-of-images-with-matrices_10_0.png)




```python

T = mat_rotate @ np.matrix(
    [[1/4, 0, 0],
     [0, 1/4, 0], 
     [0, 0, 1]])

linear_transform(T, b/4, h/4)
```




![png](linear-transformations-of-images-with-matrices_files/linear-transformations-of-images-with-matrices_11_0.png)




```python
T = np.matrix(
    [[0, -1, 0],
     [1, 0, 0], 
     [0, 0, 1]]) @ np.matrix(
    [[1, 0, -b],
     [0, 1, h],
     [0, 0, 1]])

linear_transform(T, b, h)
```




![png](linear-transformations-of-images-with-matrices_files/linear-transformations-of-images-with-matrices_12_0.png)




```python

```
