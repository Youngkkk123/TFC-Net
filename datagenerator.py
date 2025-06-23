from torchvision import transforms
from dataAugmentation import ImageEqualSize, Square_Generated
import torch
import numpy as np
from torch.utils.data import Dataset
import os
import PIL
from PIL import Image


class Datasets_Bmode_patient_dir(Dataset):
    '''加载超声图像和标签的数据集类'''
    def __init__(self, data_list, transform=None, device='cuda:0'):
        self.data_list = data_list
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # 提取图像路径和标签
        patient_path, label = self.data_list[index]
        patient_path = patient_path.replace(r'D:\project\Thyroid_classify\FTCdata', r'D:\Work\Thyroid_classify\FTC_train_data')
        patient_path = patient_path.replace(r'D:\FTC_data\验证集合并', r'D:\Work\Thyroid_classify\FTC_train_data\external_test_new\验证集一')
        # patient_path = patient_path.replace(r'D:\project\Thyroid_classify\test1n2', r'D:\Work\Thyroid_classify\FTC_train_data\external_test_new')      # 外部
        # patient_path = patient_path.replace(r'D:\Work\Thyroid_classify\FTC_train_data\external_test_new', r'D:\Work\Thyroid_classify\FTC_train_data\external_test_new')

        label = int(label)  # 确保标签是整数类型

        data = {
            'path': patient_path,
            'label': label
        }
        return data


class Datasets_Bmode_roi_patient_allimg(Dataset):
    def __init__(self, data_list, transform=None, device='cuda:0'):
        self.data_list = data_list
        self.transform = transform
        self.device = device
        self.image_paths = []
        self.labels = []  # 显式声明 labels 属性

        # 预先生成所有图像路径和标签
        for patient_folder_path, label in self.data_list:
            # patient_folder_path = patient_folder_path.replace('D:\\Work\\Thyroid_classify\\', '/home/yangkai/project/Thyroid_classify/')
            # patient_folder_path = patient_folder_path.replace('\\', '/')

            patient_folder_path = patient_folder_path.replace(r'D:\project\Thyroid_classify\TrainDataset_1_ljz',r'D:\Work\Thyroid_classify\FTC_train_data')
            patient_folder_path = patient_folder_path.replace(r'D:\project\Thyroid_classify\FTCdata',r'D:\Work\Thyroid_classify\FTC_train_data')
            patient_folder_path = patient_folder_path.replace(r'D:\project\Thyroid_classify\test1n2',r'D:\Work\Thyroid_classify\FTC_train_data\external_test')

            label = int(label)
            image_paths = [os.path.join(patient_folder_path, f)
                           for f in os.listdir(patient_folder_path)
                           if os.path.isfile(os.path.join(patient_folder_path, f))
                           and f.lower().endswith(('.png', '.jpg', '.jpeg', 'bmp'))]
            if not image_paths:
                raise ValueError(f"No images found in folder {patient_folder_path}")
            self.image_paths.extend(image_paths)
            self.labels.extend([label] * len(image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label = self.labels[index]

        img = Image.open(img_path).convert('L')
        img = Square_Generated(img)
        if self.transform:
            img = self.transform(img)

        return {
            'image': img,
            'label': label,
            'path': img_path
        }

