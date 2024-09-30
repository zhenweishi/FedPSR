import os
import json
import numpy as np
import sys
sys.path.append('lib/')
from monai.transforms import Compose, LoadImaged, AddChanneld, Orientationd, Spacingd, ScaleIntensityRanged, \
    CropForegroundd, ToTensord
from monai.data import ImageWriter
from glob import glob
from utils import set_seed, dist_setup, get_conf
#import lib.trainers as trainers

def save_transformed_data(args, img_path, label_path, save_dir):
    # Get a list of image and label files
    img_files = sorted(glob(os.path.join(img_path, "img*.nii.gz")))
    label_files = sorted(glob(os.path.join(label_path, "label*.nii.gz")))

    # Define your transform
    val_transform = Compose([
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"],
                 pixdim=(args.space_x, args.space_y, args.space_z),
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"],
                             a_min=args.a_min,
                             a_max=args.a_max,
                             b_min=args.b_min,
                             b_max=args.b_max,
                             clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ])

    transformed_data = []
    writer = ImageWriter(output_dir=save_dir, output_ext='.nii.gz')

    # Loop through the image and label files, apply the transform, and save the results
    for img_file, label_file in zip(img_files, label_files):
        print("img_file",img_file)
        print("label_file",label_file)
        data_dict = {"image": img_file, "label": label_file}
        transformed = val_transform(data_dict)
        save_img_file = os.path.join(save_dir, os.path.basename(img_file))
        print(save_img_file)
        save_label_file = os.path.join(save_dir, os.path.basename(label_file))
        print(save_label_file)
        writer.write(transformed["image"].numpy(), file_name=save_img_file)
        writer.write(transformed["label"].numpy(), file_name=save_label_file)
        print("yydssss")
        transformed_data.append({"image": save_img_file, "label": save_label_file})

    # Save json
    json_file = os.path.join(save_dir, "transformed_data.json")
    with open(json_file, 'w') as f:
        json.dump(transformed_data, f)

    print(f"Transformed data saved to {save_dir} and file paths saved to {json_file}")


# You would need to replace 'args' with actual argument values.
args = get_conf()
save_transformed_data(args, "/media/cz/disk14/CODE/MAE/DATA/RawData/Training/img",
                      "/media/cz/disk14/CODE/MAE/DATA/RawData/Training/label",
                      "/media/cz/disk14/CODE/MAE/DATA/RawData/Training/pre")