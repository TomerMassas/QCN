import os
import glob
from PIL import Image


def resize_images(source_dir, target_dir, width, height):
    """
    Resize all images from source_dir and save to target_dir.

    Args:
        source_dir (str): Directory containing original images
        target_dir (str): Directory to save resized images
        width (int): Target width
        height (int): Target height
    """
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Get all image files
    image_files = glob.glob(os.path.join(source_dir, '*.jpg')) + \
                  glob.glob(os.path.join(source_dir, '*.png')) + \
                  glob.glob(os.path.join(source_dir, '*.jpeg'))

    for image_path in image_files:
        # Get the filename
        filename = os.path.basename(image_path)

        # Open the image
        with Image.open(image_path) as img:
            # Resize the image
            resized_img = img.resize((width, height), Image.LANCZOS)

            # Save the resized image
            resized_img.save(os.path.join(target_dir, filename))

        print(f"Processed: {filename}")


# Example usage for SPAQ images (384x384)
resize_images(r"C:\Users\TomerMassas\Documents\GitHub\QCN\dataset_test\segment_034_original_frames",
              r"C:\Users\TomerMassas\Documents\GitHub\QCN\dataset_test\segment_034_test",
              384, 384)

# Example usage for KonIQ10K images (512x384)
# resize_images("path_to_original_KonIQ10K_images", "512x384", 512, 384)