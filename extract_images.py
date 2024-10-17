import os
from etl8b_parser import read_record_etl8b

def extract_images(file_path, output_dir):
    ## Create the output directory if it doesn't exist
    if(not os.path.exists(output_dir)):
        os.makedirs(output_dir)

    ## Counter for naming the image files
    counter = 0

    ## Iterate through the dataset
    for char, img in read_record_etl8b(file_path):
        filename = f"{counter:06d}_{char}.png"
        file_path = os.path.join(output_dir, filename)

        img.save(file_path)

        counter += 1

        ## Print progress every 1000 images
        if(counter % 1000 == 0):
            print(f"Processed {counter} images...")

    print(f"Extraction complete. Total images extracted: {counter}")

if(__name__ == "__main__"):
    etl8b_file_path = "ETL8B/ETL8B/ETL8B2C1"
    output_directory = "extracted_images"

    extract_images(etl8b_file_path, output_directory)
