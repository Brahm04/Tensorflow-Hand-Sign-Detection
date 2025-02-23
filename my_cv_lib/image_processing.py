import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter

def convert_to_grayscale(image):
    """Convert an RGB image to grayscale using OpenCV"""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def resize_image(image, new_width, new_height):
    """Resize an image using OpenCV's Bicubic interpolation"""
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

def crop_image(image, x_start, y_start, width, height):
    """Crop an image to a specified region."""
    y_end = min(y_start + height, image.shape[0])
    x_end = min(x_start + width, image.shape[1])
    return image[y_start:y_end, x_start:x_end]

def sobel_edge_detection(image):
    """Apply Sobel edge detection, auto-converting to grayscale if needed."""
    if len(image.shape) == 3:
        image = convert_to_grayscale(image)
    
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0,  0,  0], [1,  2,  1]])

    grad_x = convolve2d(image, sobel_x, mode="same", boundary="symm")
    grad_y = convolve2d(image, sobel_y, mode="same", boundary="symm")

    sobel_image = np.sqrt(grad_x**2 + grad_y**2)
    sobel_image = (sobel_image / np.max(sobel_image)) * 255  # Normalize

    return sobel_image.astype(np.uint8)

def canny_edge_detection(image, low_threshold=50, high_threshold=150):
    """Apply OpenCV's built-in Canny edge detection."""
    if len(image.shape) == 3:
        image = convert_to_grayscale(image)
    return cv2.Canny(image, low_threshold, high_threshold)

def flip_image(image, axis):
    """Flip an image using OpenCV."""
    if axis not in [0, 1]:
        raise ValueError("Axis must be 0 (vertical) or 1 (horizontal).")
    return cv2.flip(image, axis)

def rotate_image(image, angle):
    """Rotate an image by any angle using OpenCV."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))

def perspective_transform(image, src_points, dst_points):
    """Apply a perspective transformation using OpenCV."""
    matrix = cv2.getPerspectiveTransform(np.float32(src_points), np.float32(dst_points))
    return cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))
