import cv2
import numpy as np

# Load the image
a = cv2.imread("C:/Users/mohan/Downloads/bird-2295436_640.jpg", cv2.IMREAD_GRAYSCALE)

# Laplacian kernel
laplacian_kernel = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]], dtype=np.float32)

# Gradient kernel
gradient_kernel = np.array([[-1, -1, -1],[-1,  8, -1],[-1, -1, -1]], dtype=np.float32)

# Perform convolution with Laplacian kernel
a1 = cv2.filter2D(a, -1, laplacian_kernel)

# Perform convolution with Gradient kernel
a3 = cv2.filter2D(a, -1, gradient_kernel)

# Display the original, Laplacian, and Gradient images
cv2.imshow("Original",a)
cv2.imshow("Laplacian Sharpened Image",a1)
cv2.imshow("Gradient Sharpened Image",a3 )

cv2.waitKey(0)
cv2.destroyAllWindows()
