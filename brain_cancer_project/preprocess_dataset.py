import numpy as np
import cv2
import os
import imutils
from tqdm import tqdm 


CURRENT_SCRIPT_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_TRAINING_DATA_INPUT_DIR = os.path.join(CURRENT_SCRIPT_BASE_DIR, 'brain_cancer') 
RAW_TESTING_DATA_INPUT_DIR = os.path.join(CURRENT_SCRIPT_BASE_DIR, 'testing')  


CLEANED_TRAINING_OUTPUT_DIR = os.path.join(CURRENT_SCRIPT_BASE_DIR, 'brain_cancer_cleaned')
CLEANED_TESTING_OUTPUT_DIR = os.path.join(CURRENT_SCRIPT_BASE_DIR, 'testing_cleaned')


try:
    from . import config as project_config 
except ImportError:
    import config as project_config

TARGET_IMG_WIDTH_SCRIPT = project_config.IMG_WIDTH  # 224X224
TARGET_IMG_HEIGHT_SCRIPT = project_config.IMG_HEIGHT 
TARGET_IMG_SIZE_SCRIPT = (TARGET_IMG_WIDTH_SCRIPT, TARGET_IMG_HEIGHT_SCRIPT)



def crop_img(img_bgr): 
    
   
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0) 

    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1] 
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if not cnts:
        return None 

    c = max(cnts, key=cv2.contourArea)

    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    h_orig, w_orig = img_bgr.shape[:2]
    x1, y1 = max(0, extLeft[0]), max(0, extTop[1])
    x2, y2 = min(w_orig, extRight[0]), min(h_orig, extBot[1])

    if y1 >= y2 or x1 >= x2:
        return None 
            
    new_img_cropped = img_bgr[y1:y2, x1:x2].copy()
    
    return new_img_cropped


def process_directory(source_input_dir, target_output_dir, dir_type="Data"): 
    print(f"\nProcessing {dir_type} from: {source_input_dir}")
    print(f"Saving cleaned data to: {target_output_dir}")

    if not os.path.exists(source_input_dir):
        print(f"ERROR: Source input directory {source_input_dir} not found!")
        return

    class_dirs = [d for d in os.listdir(source_input_dir) if os.path.isdir(os.path.join(source_input_dir, d))]
    
    total_images_processed = 0
    total_images_failed = 0

    for class_dir in class_dirs:
        source_class_path = os.path.join(source_input_dir, class_dir)
        target_class_path = os.path.join(target_output_dir, class_dir) 
        
        if not os.path.exists(target_class_path):
            os.makedirs(target_class_path)
            print(f"Created directory: {target_class_path}")

        image_files = [f for f in os.listdir(source_class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
        
        print(f"  Processing class: {class_dir} ({len(image_files)} images)")
        for img_name in tqdm(image_files, desc=f"    {class_dir}"):
            img_path = os.path.join(source_class_path, img_name)
            save_img_path = os.path.join(target_class_path, img_name)

            try:
                image_bgr = cv2.imread(img_path)
                if image_bgr is None:
                    total_images_failed += 1
                    continue
                
                cropped_image_bgr = crop_img(image_bgr)
                
                if cropped_image_bgr is None or cropped_image_bgr.size == 0:
                    total_images_failed += 1
                    continue

                resized_image_bgr = cv2.resize(cropped_image_bgr, TARGET_IMG_SIZE_SCRIPT) 
                
                cv2.imwrite(save_img_path, resized_image_bgr)
                total_images_processed += 1
            except Exception as e:
                total_images_failed += 1
    
    print(f"\n{dir_type} processing complete.")
    print(f"Total images successfully processed: {total_images_processed}")
    print(f"Total images failed to process: {total_images_failed}")


if __name__ == "__main__":
    print("Starting dataset pre-processing (cropping and resizing)...")
    print(f"Reading RAW training data from: {RAW_TRAINING_DATA_INPUT_DIR}")
    print(f"Reading RAW testing data from: {RAW_TESTING_DATA_INPUT_DIR}")
    print(f"Outputting CLEANED training data to: {CLEANED_TRAINING_OUTPUT_DIR}")
    print(f"Outputting CLEANED testing data to: {CLEANED_TESTING_OUTPUT_DIR}")
    print(f"Target image size for cleaned images: {TARGET_IMG_SIZE_SCRIPT[0]}x{TARGET_IMG_SIZE_SCRIPT[1]}")
    
   
    process_directory(RAW_TRAINING_DATA_INPUT_DIR, CLEANED_TRAINING_OUTPUT_DIR, "Training Data")
    
   
    process_directory(RAW_TESTING_DATA_INPUT_DIR, CLEANED_TESTING_OUTPUT_DIR, "Testing Data")
    
    print("\nPre-processing finished.")
    print("IMPORTANT: Ensure your project's config.py has DATASET_DIR and TEST_DATA_DIR")
    print(f"pointing to '{CLEANED_TRAINING_OUTPUT_DIR}' and '{CLEANED_TESTING_OUTPUT_DIR}' respectively for the main application.")
