from re import L
import numpy as np
from monai import transforms
from torch import scalar_tensor, zero_

class ConvertToMultiChannelBasedOnBratsClassesd(transforms.MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(np.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                np.logical_or(
                    np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = np.concatenate(result, axis=0).astype(np.float32)
        return d


def get_val_transforms(args):
    if args.dataset == 'lung':
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.AddChanneld(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(args.space_x, args.space_y, args.space_z),
                                    mode=("bilinear", "nearest")),
                transforms.ScaleIntensityRanged(keys=["image"],
                                                a_min=args.a_min,
                                                a_max=args.a_max,
                                                b_min=args.b_min,
                                                b_max=args.b_max,
                                                clip=True),
                # transforms.ScaleIntensityd(keys=["image"]),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    return val_transform


def get_metric_val_transforms(args):
    if args.dataset == 'lung':
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.AddChanneld(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(args.space_x, args.space_y, args.space_z),
                                    mode=("bilinear", "nearest")),
                transforms.ScaleIntensityRanged(keys=["image"],
                                                a_min=args.a_min,
                                                a_max=args.a_max,
                                                b_min=args.b_min,
                                                b_max=args.b_max,
                                                clip=True),
                # transforms.ScaleIntensityd(keys=["image"]),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    return val_transform


def get_scratch_labeled_transforms(args):
    if args.dataset == 'lung':
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.AddChanneld(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.ScaleIntensityRanged(keys=["image"],
                                                a_min=args.a_min,
                                                a_max=args.a_max,
                                                b_min=args.b_min,
                                                b_max=args.b_max,
                                                clip=True),
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    #spatial_size=(48, 48, 48),
                    pos=1,
                    neg=1,
                    num_samples=args.num_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                transforms.RandFlipd(keys=["image", "label"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=2),
                transforms.RandRotate90d(
                    keys=["image", "label"],
                    prob=args.RandRotate90d_prob,
                    max_k=3,
                ),
                transforms.RandScaleIntensityd(keys="image",
                                            factors=0.1,
                                            prob=args.RandScaleIntensityd_prob),
                transforms.RandShiftIntensityd(keys="image",
                                            offsets=0.1,
                                            prob=args.RandShiftIntensityd_prob),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    return train_transform

def get_scratch_unlabeled_transforms(args):
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.AddChanneld(keys=["image"]),
            transforms.Orientationd(keys=["image"], axcodes="RAS"),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max,
                clip=True
            ),
            transforms.SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            transforms.RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=args.num_samples,
                random_center=True,
                random_size=False,
            ),
            transforms.RandFlipd(keys=["image"],
                                 prob=args.RandFlipd_prob,
                                 allow_missing_keys=True,
                                 spatial_axis=0),
            transforms.RandFlipd(keys=["image"],
                                 prob=args.RandFlipd_prob,
                                 allow_missing_keys=True,
                                 spatial_axis=1),
            transforms.RandFlipd(keys=["image"],
                                 prob=args.RandFlipd_prob,
                                 allow_missing_keys=True,
                                 spatial_axis=2),
            transforms.RandRotate90d(
                keys=["image"],
                allow_missing_keys=True,
                prob=args.RandRotate90d_prob,
                max_k=3,
            ),
            transforms.RandScaleIntensityd(keys="image",
                                           factors=0.1,
                                           prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image",
                                           offsets=0.1,
                                           prob=args.RandShiftIntensityd_prob),
            transforms.ToTensord(keys=["image"]),
        ]
    )
    return train_transform

def get_scratch_train_transforms(args):
    if args.dataset == 'lung':
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.AddChanneld(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.ScaleIntensityRanged(keys=["image"],
                                                a_min=args.a_min,
                                                a_max=args.a_max,
                                                b_min=args.b_min,
                                                b_max=args.b_max,
                                                clip=True),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=1,
                    neg=1,
                    num_samples=args.num_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                transforms.RandFlipd(keys=["image", "label"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=2),
                transforms.RandRotate90d(
                    keys=["image", "label"],
                    prob=args.RandRotate90d_prob,
                    max_k=3,
                ),
                transforms.RandScaleIntensityd(keys="image",
                                            factors=0.1,
                                            prob=args.RandScaleIntensityd_prob),
                transforms.RandShiftIntensityd(keys="image",
                                            offsets=0.1,
                                            prob=args.RandShiftIntensityd_prob),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    return train_transform


def get_mae_radom_transforms(args):
    if args.dataset == 'lung':
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image"]),
                transforms.AddChanneld(keys=["image"]),
                transforms.Orientationd(keys=["image"], axcodes="RAS"),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max,
                    clip=True
                ),
                transforms.SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
                transforms.CropForegroundd(keys=["image"], source_key="image",
                                           k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
                transforms.RandSpatialCropSamplesd(
                    keys=["image"],
                    roi_size=[args.roi_x, args.roi_y, args.roi_z],
                    num_samples=args.num_samples,
                    random_center=True,
                    random_size=False,
                ),
                transforms.RandFlipd(keys=["image"],
                                     prob=args.RandFlipd_prob,
                                     spatial_axis=0),
                transforms.RandFlipd(keys=["image"],
                                     prob=args.RandFlipd_prob,
                                     spatial_axis=1),
                transforms.RandFlipd(keys=["image"],
                                     prob=args.RandFlipd_prob,
                                     spatial_axis=2),
                transforms.ToTensord(keys=["image"]),
            ]
        )
    return train_transform


def get_mae_pretrain_transforms(args):
    if args.dataset == 'lung':
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image"]),
                transforms.AddChanneld(keys=["image"]),
                transforms.Orientationd(keys=["image"],
                                        axcodes="RAS"),
                transforms.ScaleIntensityRanged(keys=["image"],
                                                a_min=args.a_min,
                                                a_max=args.a_max,
                                                b_min=args.b_min,
                                                b_max=args.b_max,
                                                clip=True),
                transforms.RandFlipd(keys=["image"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=0),
                transforms.RandFlipd(keys=["image"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=1),
                transforms.RandFlipd(keys=["image"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=2),
                transforms.ToTensord(keys=["image"]),
            ]
        )
    return train_transform

def get_mae_radom_transforms(args):
    if args.dataset == 'lung':
        train_transform = transforms.Compose(
            [
                    transforms.LoadImaged(keys=["image"]),
                    transforms.AddChanneld(keys=["image"]),
                    transforms.Orientationd(keys=["image"], axcodes="RAS"),
                    transforms.ScaleIntensityRanged(
                        keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max,
                        clip=True
                    ),
                    transforms.CropForegroundd(keys=["image"], source_key="image",
                                    k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
                    transforms.RandSpatialCropSamplesd(
                        keys=["image"],
                        roi_size=[args.roi_x, args.roi_y, args.roi_z],
                        num_samples=args.num_samples,
                        random_center=True,
                        random_size=False,
                    ),
                    transforms.RandFlipd(keys=["image"],
                                         prob=args.RandFlipd_prob,
                                         spatial_axis=0),
                    transforms.RandFlipd(keys=["image"],
                                         prob=args.RandFlipd_prob,
                                         spatial_axis=1),
                    transforms.RandFlipd(keys=["image"],
                                         prob=args.RandFlipd_prob,
                                         spatial_axis=2),
                    transforms.ToTensord(keys=["image"]),
                ]
        )
    return train_transform


def get_post_transforms(args):
    if args.dataset == 'lung':
        if args.test:
            post_pred = transforms.Compose([transforms.EnsureType(),
                                            # Resize(scale_params=(args.space_x, args.space_y, args.space_z)),
                                            transforms.AsDiscrete(argmax=True, to_onehot=args.num_classes)])
            post_label = transforms.Compose([transforms.EnsureType(),
                                            # Resize(scale_params=(args.space_x, args.space_y, args.space_z)),
                                            transforms.AsDiscrete(to_onehot=args.num_classes)])
        else:
            post_pred = transforms.Compose([transforms.EnsureType(),
                                            transforms.AsDiscrete(argmax=True, to_onehot=args.num_classes)])
            post_label = transforms.Compose([transforms.EnsureType(),
                                            transforms.AsDiscrete(to_onehot=args.num_classes)])
    return post_pred, post_label