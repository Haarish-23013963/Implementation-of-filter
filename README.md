# Implementation-of-filter
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.
## Software Required:
Anaconda - Python 3.7
## Algorithm:
### Step1
</br>
Import the required libraries.
</br> 

### Step2
</br>
Convert the image from BGR to RGB.
</br> 

### Step3
</br>
Apply the required filters for the image separately.
</br> 

### Step4
</br>
Plot the original and filtered image by using matplotlib.pyplot.
</br> 

### Step5
</br>
End the program.
</br> 

## Program
### Developed By : HAARISH V
### Register Number :212223230067
</br>

## 1. Smoothing Filters

### i) Using Averaging Filter
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Read the input image
image = cv2.imread('LUFFY.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Define a Kernel
kernel = np.ones((5,5), dtype = np.float32) / 5**2

print (kernel)

image = cv2.imread('LUFFY.jpeg')
dst = cv2.filter2D(image, ddepth = -1, kernel = kernel)

plt.figure(figsize = (18, 6))
plt.subplot(121); plt.imshow(image [:, :, ::-1]);
plt.title('Input Image')
plt.subplot(122); plt.imshow(average_filter[:, :, ::-1]);
plt.title('Output Image ( Average Filter)')

```
### ii) Using Weighted Averaging Filter
```
kernel = np.array([[1,2,1],
                   [2,4,2],
                   [1,2,1]])/16
weighted_average_filter = cv2.filter2D(image, -1, kernel)
# Display the images.
plt.figure(figsize = (18, 6))
plt.subplot(121);plt.subplot(121); plt.imshow(image [:, :, ::-1]);
plt.title('Input Image')
plt.subplot(122);plt.imshow(weighted_average_filter[:, :, ::-1]);
plt.title('Output Image(weighted_average_filter)');plt.show()

```
### iii) Using Gaussian Filter
```
# Apply Gaussian blur.
gaussian_filter = cv2.GaussianBlur(image, (29,29), 0, 0)
# Display the images.

plt.figure(figsize = (18, 6))
plt.subplot(121); plt.imshow(image [:, :, ::-1]); plt.title('Input Image')
plt.subplot(122); plt.imshow(gaussian_filter[:, :, ::-1]); plt.title('Output Image ( Gaussian Filter)')
```
### iv)Using Median Filter
```


median_filter = cv2.medianBlur(image, 19)
# Display the images.

plt.figure(figsize = (18, 6))
plt.subplot(121); plt.imshow(image [:, :, ::-1]); plt.title('Input Image')
plt.subplot(122); plt.imshow(median_filter[:, :, ::-1]); plt.title('Output Image ( Median_filter)')
```

## 2. Sharpening Filters
### i) Using Laplacian Linear Kernal
```
# i) Using Laplacian Kernel (Manual Kernel)
laplacian_kernel = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])
sharpened_laplacian_kernel = cv2.filter2D(image, -1, kernel = laplacian_kernel)
# Display the images.

plt.figure(figsize = (18, 6))
plt.subplot(121); plt.imshow(image [:, :, ::-1]); plt.title('Input Image')
plt.subplot(122); plt.imshow(sharpened_laplacian_kernel[:, :, ::-1]); plt.title('Output Image ( Laplacian_filter)')

```
### ii) Using Laplacian Operator
```
# ii) Using Laplacian Operator (OpenCV built-in)
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
laplacian_operator = cv2.Laplacian(gray_image, cv2.CV_64F)
laplacian_operator = np.uint8(np.absolute(laplacian_operator))
# Display the images.

plt.figure(figsize = (18, 6))
plt.subplot(131); plt.imshow(image [:, :, ::-1]); plt.title('Input Image')
plt.subplot(132); plt.imshow(gray_image, cmap='gray'); plt.title('Gray_image')
plt.subplot(133); plt.imshow(laplacian_operator,cmap='gray'); plt.title('Output Image ( Laplacian_filter)')

```

## OUTPUT:
## 1. Smoothing Filters


### i) Using Averaging Filter
<img width="1380" height="610" alt="image" src="https://github.com/user-attachments/assets/e1eda3dd-0cc6-4aed-8bb5-d45ffbd86b95" />



### ii)Using Weighted Averaging Filter

<img width="1379" height="576" alt="image" src="https://github.com/user-attachments/assets/4ba04a8d-f7fd-4eff-8859-51226ea58324" />


### iii)Using Gaussian Filter

<img width="1371" height="609" alt="image" src="https://github.com/user-attachments/assets/fc31b4dd-0b09-44c9-bab8-d854e0cbabf3" />


### iv) Using Median Filter

<img width="1373" height="604" alt="image" src="https://github.com/user-attachments/assets/bf76ab96-1a06-4027-aa30-aa1e2c47d20d" />

## 2. Sharpening Filters


### i) Using Laplacian Kernal
<img width="1366" height="599" alt="image" src="https://github.com/user-attachments/assets/970e84de-84f5-45f8-a2fa-4d7335153147" />


### ii) Using Laplacian Operator
<img width="1368" height="492" alt="image" src="https://github.com/user-attachments/assets/5aa90123-7a63-4d01-877e-4a16815a7408" />


## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
