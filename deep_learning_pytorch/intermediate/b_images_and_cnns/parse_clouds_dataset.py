# Simply a quick script to change the directory structure to 
# how the data is on kaggle https://www.kaggle.com/competitions/cloud-type-classification2/data
# to how the DataCamp wants the structure to be

import os
import shutil
import pandas as pd

# --- Configuration ---
# Define the names of your directories and files.
# These are assumed to be in the same directory as the script (root dir).
IMAGES_DIR_NAME = 'images'
IMAGES_DIR_NAME = 'images'
OUTPUT_TRAIN_DIR_NAME = 'clouds_train'
OUTPUT_TEST_DIR_NAME = 'clouds_test'
TRAIN_CSV_NAME = 'train.csv'
TEST_CSV_NAME = 'test.csv'
ID_COLUMN_NAME = 'id'  # The name of the column in your CSVs that holds the image filenames
LABEL_COLUMN_NAME = 'label'

# --- Get Directory ---
# This is the directory where the script is located and where 'images', 'train.csv', etc., are expected.
BASE_DIR = "" # Add your dir here

# --- Construct Full Paths ---
# Create absolute paths for all directories and files.
test_images_dir_path = os.path.join(BASE_DIR, IMAGES_DIR_NAME, "test")
train_images_dir_path = os.path.join(BASE_DIR, IMAGES_DIR_NAME, "train")
output_train_dir_path = os.path.join(BASE_DIR, OUTPUT_TRAIN_DIR_NAME)
output_test_dir_path = os.path.join(BASE_DIR, OUTPUT_TEST_DIR_NAME)
train_csv_file_path = os.path.join(BASE_DIR, TRAIN_CSV_NAME)
test_csv_file_path = os.path.join(BASE_DIR, TEST_CSV_NAME)


def organize_images(csv_file_path, source_images_dir, target_output_dir, id_column, label_column):
    """
    Reads a CSV file and copies images listed in the specified id_column
    from the source_images_dir to the target_output_dir.

    Args:
        csv_file_path (str): The full path to the CSV file (e.g., train.csv or test.csv).
        source_images_dir (str): The full path to the directory containing the original images.
        target_output_dir (str): The full path to the directory where images should be copied.
        id_column (str): The name of the column in the CSV that contains the image filenames.
        label_column (str): The name of the column in the CSV that contains the labels.
    """
    print(f"\nProcessing CSV file: {csv_file_path}")
    print(f"Source images from: {source_images_dir}")
    print(f"Copying images to: {target_output_dir}")

    # 1. Create the target output directory if it doesn't already exist.
    # os.makedirs will create any necessary parent directories as well.
    # exist_ok=True means it won't raise an error if the directory already exists.
    try:
        os.makedirs(target_output_dir, exist_ok=True)
        print(f"Ensured target directory exists: {target_output_dir}")
    except OSError as e:
        print(f"Error: Could not create directory {target_output_dir}. Reason: {e}")
        return # Stop processing for this CSV if directory creation fails

    # 2. Check if the source images directory exists.
    if not os.path.isdir(source_images_dir):
        print(f"Error: Source images directory '{source_images_dir}' not found. Please check the path.")
        return

    # 3. Read the CSV file using pandas.
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file_path}' not found. Please ensure it's in the correct location.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file '{csv_file_path}' is empty.")
        return
    except Exception as e:
        print(f"Error: Could not read CSV file '{csv_file_path}'. Reason: {e}")
        return

    # 4. Check if the specified ID & label columns exists in the CSV.
    if id_column not in df.columns:
        print(f"Error: Column '{id_column}' not found in '{csv_file_path}'. Available columns: {df.columns.tolist()}")
        return
    
    if label_column not in df.columns:
        print(f"Error: Column '{label_column}' not found in '{csv_file_path}'. Available columns: {df.columns.tolist()}")
        return

    # 5. Iterate through each row in the CSV.
    images_copied_count = 0
    images_not_found_count = 0

    for index, row in df.iterrows():
        # Get the image filename from the specified ID column.
        # Convert to string in case pandas interprets it as a number.
        image_filename = str(row[id_column])
        image_label = str(row[id_column])


        # Construct the full path to the source image.
        source_image_path = os.path.join(source_images_dir, image_filename)
        # Construct the full path to where the image will be copied.
        destination_image_path = os.path.join(target_output_dir, image_label, image_filename)
        destination_dir = os.path.dirname(destination_image_path)
        os.makedirs(destination_dir, exist_ok=True)

        # Check if the source image actually exists.
        if os.path.isfile(source_image_path):
            try:
                # Copy the image. shutil.copy2 also attempts to copy metadata.
                shutil.copy2(source_image_path, destination_image_path)
                # print(f"  Copied: {image_filename} to {target_output_dir}") # Uncomment for verbose output
                images_copied_count += 1
            except Exception as e:
                print(f"  Error copying {image_filename}: {e}")
        else:
            print(f"  Warning: Image '{image_filename}' not found in '{source_images_dir}'. Skipping.")
            images_not_found_count += 1
            
    print(f"Finished processing for {csv_file_path}.")
    print(f"  Images successfully copied: {images_copied_count}")
    print(f"  Images listed in CSV but not found in source directory: {images_not_found_count}")


def main():
    """
    Main function to orchestrate the creation of directories and copying of images.
    """
    print("Starting image organization process...")

    # --- Process Test Images ---
    # This will create 'clouds_test' and copy images based on 'test.csv'.
    organize_images(
        csv_file_path=test_csv_file_path,
        source_images_dir=test_images_dir_path,
        target_output_dir=output_test_dir_path,
        id_column=ID_COLUMN_NAME,
        label_column=LABEL_COLUMN_NAME
    )

    # --- Process Training Images ---
    # This will create 'clouds_train' and copy images based on 'train.csv'.
    organize_images(
        csv_file_path=train_csv_file_path,
        source_images_dir=train_images_dir_path,
        target_output_dir=output_train_dir_path,
        id_column=ID_COLUMN_NAME,
        label_column=LABEL_COLUMN_NAME
    )

    print("\nImage organization process completed!")


if __name__ == "__main__":
    # This standard Python construct ensures that main() is called only when
    # the script is executed directly (e.g., python your_script_name.py),
    # and not when it's imported as a module into another script.
    main()