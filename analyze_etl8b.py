import os
from collections import Counter
from etl8b_parser import read_record_etl8b
from PIL import Image
import numpy as np

def analyze_etl8b(file_path):
    ## Check if the file exists
    if(not os.path.exists(file_path)):
        print(f"Error: File '{file_path}' does not exist.")
        return

    ## Initialize counters and data structures
    total_images = 0
    char_counter = Counter()
    image_sizes = set()
    total_pixels = 0
    black_pixel_counts = []
    white_pixel_counts = []

    ## Analyze the file
    for char, img in read_record_etl8b(file_path):
        total_images += 1
        char_counter[char] += 1
        
        ## Image size
        image_sizes.add(img.size)
        
        ## Convert image to numpy array for analysis
        img_array = np.array(img)
        total_pixels += img_array.size
        black_pixel_counts.append(np.sum(img_array == 0))
        white_pixel_counts.append(np.sum(img_array == 255))

        ## Print progress every 1000 images
        if(total_images % 1000 == 0):
            print(f"Processed {total_images} images...")

    ## Calculate statistics
    unique_chars = len(char_counter)
    most_common_chars = char_counter.most_common(10)
    avg_black_pixels = np.mean(black_pixel_counts)
    avg_white_pixels = np.mean(white_pixel_counts)
    avg_black_percentage = (avg_black_pixels / total_pixels) * 100
    avg_white_percentage = (avg_white_pixels / total_pixels) * 100

    ## Print analysis results
    print("\nETL8B File Analysis Results:")
    print(f"File path: {file_path}")
    print(f"Total images: {total_images}")
    print(f"Unique characters: {unique_chars}")
    print(f"Image sizes: {image_sizes}")
    print("\nTop 10 most common characters:")
    for char, count in most_common_chars:
        print(f"  {char}: {count} occurrences")
    print(f"\nAverage black pixels per image: {avg_black_pixels:.2f} ({avg_black_percentage:.2f}%)")
    print(f"Average white pixels per image: {avg_white_pixels:.2f} ({avg_white_percentage:.2f}%)")

if(__name__ == "__main__"):
    etl8b_file_path = "ETL8B/ETL8B/ETL8B2C1"
    analyze_etl8b(etl8b_file_path)
