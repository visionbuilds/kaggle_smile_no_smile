import os
import cv2

def crop_and_save_images(img_dir, save_to, crop_size=148):
    """
    Crop the central square of each image in img_dir and save to save_to.

    Parameters:
        img_dir (str): Path to the folder containing JPEG images.
        save_to (str): Path to the folder to save cropped images.
        crop_size (int): The side length of the square crop (default is 148 pixels).
    """
    # Create the output directory if it doesn't exist
    os.makedirs(save_to, exist_ok=True)

    # Process each image in the input directory
    for img_name in os.listdir(img_dir):
        if img_name.endswith(".jpg") or img_name.endswith(".jpeg"):
            img_path = os.path.join(img_dir, img_name)

            # Load the image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Could not load image: {img_name}")
                continue

            # Get image dimensions
            height, width = image.shape[:2]

            # Calculate cropping coordinates
            start_x = (width - crop_size) // 2
            start_y = (height - crop_size) // 2
            end_x = start_x + crop_size
            end_y = start_y + crop_size

            # Crop the central square
            cropped_image = image[start_y:end_y, start_x:end_x]

            # Save the cropped image
            save_path = os.path.join(save_to, img_name)
            cv2.imwrite(save_path, cropped_image)
            print(f"Cropped and saved: {save_path}")

# Example usage
img_dir = r"A:\pycharm_projects\kaggle_smile_no_smile\datasets\Celeba\img_align_celeba"  # Replace with the path to your folder containing JPEG images
save_to = r"A:\pycharm_projects\kaggle_smile_no_smile\datasets\Celeba\cropped"    # Replace with the path to the folder to save cropped images

crop_and_save_images(img_dir, save_to)

