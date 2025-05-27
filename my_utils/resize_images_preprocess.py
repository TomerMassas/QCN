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

    print(f"Processed: {source_dir}")


if __name__ == "__main__":

    video_name = "CM_C+R_It Get's Better 4K"
    frames_seg_paths = os.listdir(fr"C:\Users\TomerMassas\Desktop\Video project\video scene detection\tests\{video_name}\frames\frames of segments")
    for seg_n in frames_seg_paths:

        # # Example usage for SPAQ images (384x384)
        # resize_images(fr"C:\Users\TomerMassas\Desktop\Video project\video scene detection\tests\Film\frames\segment_{seg_num}",
        #               fr"C:\Users\TomerMassas\Documents\GitHub\QCN\dataset_test\SPAQ\segment_{seg_num}_test",
        #               384, 384)



        # Example usage for KonIQ10K images (512x384)
        resize_images(fr"C:\Users\TomerMassas\Desktop\Video project\video scene detection\tests\{video_name}\frames\frames of segments\{seg_n}",
                      fr"C:\Users\TomerMassas\Documents\GitHub\QCN\dataset_test\KonIQ10K\{video_name}\{seg_n}_test",
                      512, 384)