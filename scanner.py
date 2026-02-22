# =====================================================
# Name: Yashwant
# Roll No: 2301010329
# Course: Image Processing & Computer Vision
# Unit: Image Acquisition & Enhancement
# Assignment Title: Smart Document Scanner & Quality Analysis
# Date: 09-Feb-2026
# =====================================================

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------
# Welcome Message
# -------------------------------
def welcome():
    print("=" * 60)
    print(" SMART DOCUMENT SCANNER & QUALITY ANALYSIS SYSTEM ")
    print("=" * 60)
    print("Simulating real-world document digitization\n")


# -------------------------------
# Create Output Folder
# -------------------------------
def create_output_folder():
    if not os.path.exists("outputs"):
        os.makedirs("outputs")


# -------------------------------
# Load Image
# -------------------------------
def load_image(path):

    image = cv2.imread(path)

    if image is None:
        print("Error loading:", path)
        return None, None

    image = cv2.resize(image, (512, 512))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image, gray


# -------------------------------
# Sampling
# -------------------------------
def sampling(gray):

    high = cv2.resize(gray, (512, 512))

    medium = cv2.resize(gray, (256, 256))
    medium = cv2.resize(medium, (512, 512))

    low = cv2.resize(gray, (128, 128))
    low = cv2.resize(low, (512, 512))

    return high, medium, low


# -------------------------------
# Quantization
# -------------------------------
def quantize(image, levels):

    step = 256 // levels
    quantized = (image // step) * step

    return quantized.astype(np.uint8)


# -------------------------------
# Observations
# -------------------------------
def print_observations():

    print("\n========= QUALITY ANALYSIS =========")

    print("\nText Clarity:")
    print("- High resolution retains edge sharpness.")
    print("- Low resolution loses fine characters.")

    print("\nReadability:")
    print("- 8-bit image maintains smooth gradients.")
    print("- 4-bit shows banding.")
    print("- 2-bit severely degrades quality.")

    print("\nOCR Suitability:")
    print("- Best: 512x512 + 8-bit.")
    print("- Worst: 128x128 + 2-bit.")

    print("====================================")


# -------------------------------
# Comparison Display
# -------------------------------
def show_and_save(original, gray,
                  high, medium, low,
                  q8, q4, q2,
                  name):

    fig, ax = plt.subplots(3, 3, figsize=(12, 12))

    ax[0,0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax[0,0].set_title("Original")

    ax[0,1].imshow(gray, cmap="gray")
    ax[0,1].set_title("Grayscale")

    ax[0,2].imshow(high, cmap="gray")
    ax[0,2].set_title("512x512")

    ax[1,0].imshow(medium, cmap="gray")
    ax[1,0].set_title("256x256")

    ax[1,1].imshow(low, cmap="gray")
    ax[1,1].set_title("128x128")

    ax[1,2].imshow(q8, cmap="gray")
    ax[1,2].set_title("8-bit")

    ax[2,0].imshow(q4, cmap="gray")
    ax[2,0].set_title("4-bit")

    ax[2,1].imshow(q2, cmap="gray")
    ax[2,1].set_title("2-bit")

    ax[2,2].axis("off")

    for i in range(3):
        for j in range(3):
            ax[i,j].axis("off")

    plt.suptitle("Document Quality Comparison")

    filename = f"outputs/{name}_comparison.png"
    plt.savefig(filename)
    plt.show()

    print("Saved:", filename)


# -------------------------------
# Main
# -------------------------------
def main():

    welcome()
    create_output_folder()

    image_folder = "images"

    if not os.path.exists(image_folder):
        print("Images folder not found!")
        return

    files = os.listdir(image_folder)

    for file in files:

        print("\nProcessing:", file)

        path = os.path.join(image_folder, file)
        original, gray = load_image(path)

        if original is None:
            continue

        high, medium, low = sampling(gray)

        q8 = quantize(gray, 256)
        q4 = quantize(gray, 16)
        q2 = quantize(gray, 4)

        name = file.split(".")[0]

        cv2.imwrite(f"outputs/{name}_8bit.png", q8)
        cv2.imwrite(f"outputs/{name}_4bit.png", q4)
        cv2.imwrite(f"outputs/{name}_2bit.png", q2)

        show_and_save(original, gray,
                      high, medium, low,
                      q8, q4, q2,
                      name)

    print_observations()
    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()