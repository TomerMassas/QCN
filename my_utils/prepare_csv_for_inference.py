import os
import csv
import argparse

def create_mos_csv(image_folder, output_folder, output_filename):
    # Get all image filenames
    image_names = [f for f in os.listdir(image_folder)
                   if os.path.isfile(os.path.join(image_folder, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]

    # Prepare CSV path
    os.makedirs(output_folder, exist_ok=True)
    csv_path = os.path.join(output_folder, output_filename)

    # Write to CSV
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_name", "MOS"])
        for image_name in image_names:
            writer.writerow([image_name, 0])

    print(f"CSV file saved to: {csv_path}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Create a MOS CSV from a folder of images.")
    # parser.add_argument("image_folder", type=str, help="Path to folder containing images")
    # parser.add_argument("output_folder", type=str, help="Path to folder where CSV should be saved")
    #
    # args = parser.parse_args()
    # create_mos_csv(args.image_folder, args.output_folder)
    type_eval =  'test' #'test' 'ref'
    dataset_name = 'KonIQ10K'  # 'SPAQ'  or 'KonIQ10K'
    video_name = "CM_C+R_It Get's Better 4K"
    for seg_num in os.listdir(fr"C:\Users\TomerMassas\Documents\GitHub\QCN\dataset_test\KonIQ10K\{video_name}"):
        seg_num = seg_num.split('_')[1]  # Extract segment number from folder name
        image_folder = fr'C:\Users\TomerMassas\Documents\GitHub\QCN\dataset_test\{dataset_name}\{video_name}\segment_{seg_num}_{type_eval}'
        output_folder = fr'C:\Users\TomerMassas\Documents\GitHub\QCN\datasplit\pictime\{video_name}'
        output_filename = f"segment_{seg_num}_{type_eval}.csv"
        create_mos_csv(image_folder, output_folder, output_filename)

