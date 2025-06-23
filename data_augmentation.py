import numpy as np
from torchvision import transforms


def expand_to_3_channels(image):
    """将灰度图像扩展到3通道"""
    if image.size(0) != 3:
        return image.repeat(3, 1, 1)
    return image


def get_data_transform_2D_vit(patch_size):
    img_trans = {
        'train': transforms.Compose([
            # transforms.Resize(patch_size),
            # transforms.RandomCrop(224),
            # transforms.RandomRotation(5, expand=False, center=None),  # 旋转
            transforms.RandomHorizontalFlip(p=0.3),
            # transforms.RandomAffine(degrees=(10), translate=(0.1, 0.1), scale=(0.8, 1.2), shear=None),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # 随机调整亮度、对比度、饱和度和色调
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.3),  # 高斯模糊
            transforms.RandomAdjustSharpness(sharpness_factor=1.2, p=0.2),  # 增强超声图像边缘
            transforms.Resize(patch_size),
            # transforms.Lambda(lambda x: custom_resize_or_pad(x, patch_size)),
            transforms.ToTensor(),
            transforms.Lambda(expand_to_3_channels),  # 扩展到3通道
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize(patch_size),
            transforms.ToTensor(),
            transforms.Lambda(expand_to_3_channels),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize(patch_size),
            transforms.ToTensor(),
            transforms.Lambda(expand_to_3_channels),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }
    return img_trans


