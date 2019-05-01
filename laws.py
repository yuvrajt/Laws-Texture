import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import signal as sg

def norm(ar):
    """ Normalize IConvolved image"""
    return 255.*np.absolute(ar)/np.max(ar)

#read image and convert it to gray
I = cv2.imread('pebbles.jpg')
I2 = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
gray2 = np.copy(gray.astype(np.float64))
(rows, cols) = gray.shape[:2]

#Create space for 16 convolutions for each kernel
conv_maps = np.zeros((rows, cols,16),np.float64)

#create an array of Laws filter vectors
filter_vectors = np.array([[1, 4, 6,  4, 1],
                            [-1, -2, 0, 2, 1],
                            [-1, 0, 2, 0, 1],
                            [1, -4, 6, -4, 1]])

#Perform matrix multiplication of vectors to get 16 kernels
filters = list()
for ii in range(4):
    for jj in range(4):
        filters.append(np.matmul(filter_vectors[ii][:].reshape(5,1),filter_vectors[jj][:].reshape(1,5)))

#Preprocess the image
smooth_kernel = (1/25)*np.ones((5,5))
gray_smooth = sg.convolve(gray2 ,smooth_kernel,"same")
gray_processed = np.abs(gray2 - gray_smooth)

#Convolve the Laws kernels
for ii in range(len(filters)):
    conv_maps[:, :, ii] = sg.convolve(gray_processed,filters[ii],'same')

#Create the 9 texture maps
texture_maps = list()
texture_maps.append(norm((conv_maps[:, :, 1]+conv_maps[:, :, 4])//2))
texture_maps.append(norm((conv_maps[:, :, 3]+conv_maps[:, :, 12])//2))
texture_maps.append(norm(conv_maps[:, :, 10]))
texture_maps.append(norm(conv_maps[:, :, 15]))
texture_maps.append(norm((conv_maps[:, :, 2]+conv_maps[:, :, 8])//2))
texture_maps.append(norm(conv_maps[:, :, 5]))
texture_maps.append(norm((conv_maps[:, :, 7]+conv_maps[:, :, 13])//2))
texture_maps.append(norm((conv_maps[:, :, 11]+conv_maps[:, :, 14])//2))

#plot the results
plt.figure('Image pre-processing')
plt.subplot(221)
plt.imshow(I2)
plt.subplot(222)
plt.imshow(gray,'gray')
plt.subplot(223)
plt.imshow(gray_smooth.astype(np.uint8),'gray')
plt.subplot(224)
plt.imshow(gray_processed.astype(np.uint8),'gray')

plt.figure('Texture Maps 1')
plt.subplot(221)
plt.imshow(texture_maps[0],'gray')
plt.subplot(222)
plt.imshow(texture_maps[1],'gray')
plt.subplot(223)
plt.imshow(texture_maps[2],'gray')
plt.subplot(224)
plt.imshow(texture_maps[3],'gray')


plt.figure('Texture Maps 2')
plt.subplot(221)
plt.imshow(texture_maps[4],'gray')
plt.subplot(222)
plt.imshow(texture_maps[5],'gray')
plt.subplot(223)
plt.imshow(texture_maps[6],'gray')
plt.subplot(224)
plt.imshow(texture_maps[7],'gray')
plt.show()
