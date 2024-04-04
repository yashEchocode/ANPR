import asyncio
from deeplearning import object_detection
import os
import csv
import re

# Directory containing images
IMAGES_DIRECTORY = "static/newImg"

# Directory to save CSV file
CSV_FILE_PATH = "./results.csv"

# Modify your code to call the object_detection function asynchronously
async def process_image(image_path, filename):
    result = await object_detection(image_path, filename)
    return result

async def process_images_in_directory(directory):
    results = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            result = await process_image(image_path, filename)
            print("Result: " + result)
            results.append([filename, result])
    return results

async def main():
    print("Running")
    results = await process_images_in_directory(IMAGES_DIRECTORY)
    # Write results to CSV file
    with open(CSV_FILE_PATH, mode='w', newline='') as csv_file:
      writer = csv.writer(csv_file)
      writer.writerow(['Filename', 'Result'])  # Header row
      writer.writerows(results)

# Call main function
asyncio.run(main())
