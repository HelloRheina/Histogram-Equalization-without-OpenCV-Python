import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2

def read_image(file_path):
    img = Image.open(r"C:\Users\Rheina Trudy\Documents\UNI\SEMESTER 5\Multimedia\Experiments\images\1.jpg").convert('L')  # Convert to grayscale
    return np.array(img)

def calculate_histogram(image):
    histogram = np.zeros(256)
    for pixel_value in range(256):
        mask = (image == pixel_value)
        histogram[pixel_value] = np.sum(mask)
    return histogram

def normalize_histogram(histogram, total_pixels):
    return histogram / total_pixels

def cumulative_distribution_function(normalized_histogram):
    return np.cumsum(normalized_histogram)

def histogram_equalization(image, cdf):
    return np.interp(image, np.arange(256), cdf * 255).astype(np.uint8)

def plot_cumulative_histogram(cdf):
    plt.plot(range(256), cdf, color='g')
    plt.grid(True)
    plt.ylabel('Cumulative Distribution Function (CDF)')
    plt.xlabel('Level of intensity')
    plt.title('Histogram Cumulative')
    plt.show()

def plot_histogram_and_cdf(image, cdf):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf_normalized = cdf * hist.max() / cdf.max()

    plt.plot(cdf_normalized, color='b')
    plt.hist(image.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('CDF', 'Histogram'), loc='upper left')
    plt.show()

def plot_original_histogram(image):
    plt.figure(figsize=(12, 6))

    # Original Image
    plt.subplot(231)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')

    # Original Histogram
    plt.subplot(232)
    hist = calculate_histogram(image)
    plt.bar(range(256), hist, color='r')
    plt.grid(True)
    plt.ylabel('Pixels with same intensity')
    plt.xlabel('Level of intensity')
    plt.title('Original Histogram')

    plt.show()

def plot_normalized_histogram(histogram):
    plt.bar(range(256), histogram, color='b')
    plt.grid(True)
    plt.ylabel('Pixels with same intensity (normalized)')
    plt.xlabel('Level of intensity')
    plt.title('Normalized Histogram')
    plt.show()

def show_comparison(original_img, custom_img, built_in_img, original_hist, custom_hist, built_in_hist):
    plt.figure(figsize=(15, 8))

    # Original Image + Original Histogram
    plt.subplot(331)
    plt.imshow(original_img, cmap='gray')
    plt.title('Original Image')
    plt.subplot(334)
    plt.hist(original_img.flatten(), 256, [0, 256], color='r')
    plt.grid(True)
    plt.ylabel('Pixels with same intensity')
    plt.xlabel('Level of intensity')
    plt.title('Original Histogram')

    # Enhanced Image + Enhanced Histogram (Written by me)
    plt.subplot(332)
    plt.imshow(custom_img, cmap='gray')
    plt.title('Enhanced Image (Written by me)')
    plt.subplot(335)
    plt.hist(custom_img.flatten(), 256, [0, 256], color='r')
    plt.grid(True)
    plt.ylabel('Pixels with same intensity')
    plt.xlabel('Level of intensity')
    plt.title('Equalized Histogram (Written by me)')

    # Enhanced Image + Enhanced Histogram (Built-in using OpenCV)
    plt.subplot(333)
    plt.imshow(built_in_img, cmap='gray')
    plt.title('Enhanced Image (Built-in using OpenCV)')
    plt.subplot(336)
    plt.hist(built_in_img.flatten(), 256, [0, 256], color='r')
    plt.grid(True)
    plt.ylabel('Pixels with same intensity')
    plt.xlabel('Level of intensity')
    plt.title('Equalized Histogram (Built-in using OpenCV)')

    # Comparison: Enhanced Image (Written by me) vs Enhanced Image (Built-in using OpenCV)
    plt.subplot(337)
    plt.imshow(custom_img, cmap='gray')
    plt.title('Enhanced Image (Written by me)')
    plt.subplot(338)
    plt.imshow(built_in_img, cmap='gray')
    plt.title('Enhanced Image (Built-in using OpenCV)')

    # Histogram: Enhanced Image (Written by me) vs Enhanced Image (Built-in using OpenCV)
    plt.subplot(339)
    plt.hist(custom_img.flatten(), 256, [0, 256], color='r', alpha=0.5, label='Written by me')
    plt.hist(built_in_img.flatten(), 256, [0, 256], color='b', alpha=0.5, label='Built-in OpenCV')
    plt.grid(True)
    plt.ylabel('Pixels with same intensity')
    plt.xlabel('Level of intensity')
    plt.title('Comparison: Equalized Histogram')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Using custom function to read the image
img = read_image("path/to/your/image.jpg")
plot_original_histogram(img)
hist = calculate_histogram(img)

# Normalize
total_pixels = img.shape[0] * img.shape[1]
norm_hist = normalize_histogram(hist, total_pixels)

# Histogram and CDF
plot_histogram_and_cdf(img, norm_hist)

# Normalized histogram
plot_normalized_histogram(norm_hist)

# CDF calculation
cdf = cumulative_distribution_function(norm_hist)

# Cumulative histogram
plot_cumulative_histogram(cdf)

# Equalization
img_equalized = histogram_equalization(img, cdf)

# Enhanced image + histogram (written by me)
plt.subplot(121)
plt.imshow(img_equalized, cmap='gray')
plt.title('Enhanced Image (Written by me)')
plt.subplot(122)
plt.hist(img_equalized.flatten(), 256, [0, 256], color='r')
plt.grid(True)
plt.ylabel('Pixels with same intensity')
plt.xlabel('Level of intensity')
plt.title('Equalized Histogram (Written by me)')
plt.show()

# Equalization using OpenCV built-in function
img_histeq = cv2.equalizeHist(img)

# Image + histogram (Built-in using OpenCV)
plt.subplot(121)
plt.imshow(img_histeq, cmap='gray')
plt.title('Enhanced Image (Built-in using OpenCV)')
plt.subplot(122)
plt.hist(img_histeq.flatten(), 256, [0, 256], color='r')
plt.grid(True)
plt.ylabel('Pixels with same intensity')
plt.xlabel('Level of Intensity')
plt.title('Equalized Histogram (Built-in using OpenCV)')
plt.show()

# Compare
show_comparison(img, img_equalized, img_histeq, hist, norm_hist, cv2.calcHist([img_histeq], [0],
                                                                                None, [256], [0, 256]))
