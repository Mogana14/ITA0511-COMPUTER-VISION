import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("C:/Users/mohan/Downloads/bird-2295436_640.jpg")

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian blur to the grayscale image (low-pass filter)
blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

# Calculate the high-boost mask
mask = gray_img - blurred_img

# Choose a scaling factor (e.g., alpha = 2 for a simple example)
alpha = 2

# Apply the high-boost filter
sharpened_img = gray_img + alpha * mask

# Clip the values to be in the valid range [0, 255]
sharpened_img = np.clip(sharpened_img, 0, 255).astype(np.uint8)

# Display the original and sharpened images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(gray_img, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(sharpened_img, cmap='gray')
plt.title('Sharpened Image')

plt.show()
