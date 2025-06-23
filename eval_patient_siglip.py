import os
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from Loss import FocalLoss
from data_augmentation import get_data_transform_2D_vit
from utils_index import calcAUC, calcACCSENSPE
from sklearn.metrics import *
from datagenerator import Datasets_Bmode_patient_dir
import random
from PIL import Image
from utils import extract_patient_identifier
from transformers import AutoModel, AutoProcessor
from peft import LoraConfig, get_peft_model  # 导入 peft 库

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
os.environ['PYTHONHASHSEED'] = str(0)
from torch.backends import cudnn

cudnn.benchmark = False
cudnn.deterministic = True


def Square_Generated(image,fill_style:int=0,map_color:tuple=0): # 创建一个函数用来产生所需要的正方形图片转化
    w, h = image.size  # 得到图片的大小
    new_image = Image.new(image.mode, size=(max(w, h), max(w, h)),color=map_color)  # 创建新的一个图片，大小取长款中最长的一边，color决定了图片中填充的颜色
    if fill_style == 0:
        point = int(abs(w - h)) // 2
        box = (point,0) if w < h else (0,point)
    elif fill_style == 1:
        length = int(abs(w - h))  # 一侧需要填充的长度
        box = (length, 0) if w < h else (0, length)  # 放在box中
    else:
        box = (0,0)
    new_image.paste(image, box)  # 将原来的image填充到new_image上，box为填充的起点坐标
    return new_image


def val_patient_process(model, datasetloader, transform, processor, device):
    with torch.no_grad():
        model.eval()

        total_loss = 0
        patient_truth_list = []  # 存储每个患者的真实标签
        patient_probability_list = []  # 存储每个患者的预测概率

        # 定义两个类别的文本
        class_texts = [
            "An image of thyroid adenoma.",
            "An image of thyroid follicular carcinoma."
        ]

        # 预先处理文本，获取文本嵌入
        text_inputs = processor(
            text=class_texts,
            return_tensors="pt",
            padding="max_length"
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        # 分离文本处理和图像处理
        text_outputs = model.get_text_features(**text_inputs)
        text_embeds = F.normalize(text_outputs, p=2, dim=1)  # 确保L2归一化

        # 定义focalloss
        criterion = FocalLoss(gamma=2, alpha=0.75)

        # 处理每个患者
        for j, datas in enumerate(datasetloader):
            patient_path = datas['path'][0]
            label = datas['label'].item()

            # 获取患者文件夹下的所有图像路径
            image_paths = [os.path.join(patient_path, f) for f in os.listdir(patient_path)
                           if os.path.isfile(os.path.join(patient_path, f)) and
                           f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

            if not image_paths:
                print(f"No images found in folder {patient_path}")
                continue

            # 存储当前患者的所有图像预测概率
            patient_image_probs = []

            for img_path in image_paths:
                # 加载图像
                img = Image.open(img_path).convert('L')
                img = Square_Generated(img)
                if transform:
                    img = transform(img)

                image_inputs = processor(images=img, return_tensors="pt").to(device)

                image_embeds = model.get_image_features(**image_inputs)
                image_embeds = F.normalize(image_embeds, p=2, dim=1)

                # 计算图像与文本之间的余弦相似度
                cos_similarity = (
                        torch.matmul(text_embeds.detach().clone(),
                                     image_embeds.t().to(text_embeds.device)) * model.base_model.logit_scale.exp()
                        + model.base_model.logit_bias
                ).T

                # 计算预测概率
                probability = torch.softmax(cos_similarity, dim=1)[:, 1].item()
                # print(probability)
                patient_image_probs.append(probability)

                # 计算交叉熵损失
                label_tensor = torch.tensor([label], dtype=torch.long).to(device)
                loss = criterion(cos_similarity, label_tensor)

                total_loss += loss.item()

            patient_probability = max(patient_image_probs)

            patient_truth_list.append(label)
            patient_probability_list.append(patient_probability)

        avg_auc, threshold = calcAUC(patient_truth_list, patient_probability_list)
        acc, sen, spe = calcACCSENSPE(patient_truth_list, patient_probability_list, threshold)
        avg_loss = total_loss / len(patient_truth_list) if patient_truth_list else 0

    return avg_loss, avg_auc, acc, sen, spe, threshold, patient_truth_list, patient_probability_list


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    foldnum = 5
    patchsize = (224, 224)
    batchsize = 1
    lr = 1e-4

    for train_seed in [68]:
        seed = train_seed
        data_path = ''
        model_load_fold = ''
        if os.path.exists(model_load_fold) is False:
            raise ValueError('{} is not exist.'.format(model_load_fold))

        save_path = ''

        # 初始化所有数据集的汇总列表
        all_test_truth_list = []
        all_test_probability_list = []
        all_val_truth_list = []
        all_val_probability_list = []
        all_train_truth_list = []
        all_train_probability_list = []

        test_save_path = os.path.join(save_path)
        os.makedirs(test_save_path, exist_ok=True)

        model_load_path = os.path.join(model_load_fold, 'Best_model.pt')

        model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device)
        processor = AutoModel.from_pretrained("google/siglip-base-patch16-224", do_rescale=False)

        config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=['q_proj', 'k_proj', 'v_proj'],
            bias="none"
        )

        model.vision_model = get_peft_model(model.vision_model, config)
        model.text_model = get_peft_model(model.text_model, config)

        model_checkpoint = torch.load(model_load_path).state_dict()
        model.load_state_dict(model_checkpoint, strict=True)
        model = model.to(device)

        img_trans = get_data_transform_2D_vit(patchsize)

        test_path = os.path.join(data_path, "test.npy")
        test_list1 = np.load(test_path, allow_pickle=True)

        valid_path = os.path.join(data_path, "valid.npy")
        valid_list1 = np.load(valid_path, allow_pickle=True)

        test_list = np.concatenate((valid_list1, test_list1))
        valid_list = test_list

        train_path_neg = os.path.join(data_path, "train_neg.npy")
        train_path_pos = os.path.join(data_path, "train_pos.npy")
        train_list_neg = np.load(train_path_neg, allow_pickle=True)
        train_list_pos = np.load(train_path_pos, allow_pickle=True)
        train_list = np.concatenate((train_list_neg, train_list_pos))

        datasetTest = Datasets_Bmode_patient_dir(test_list, device=device)
        testloader = DataLoader(dataset=datasetTest, batch_size=1, shuffle=False)

        datasetVal = Datasets_Bmode_patient_dir(valid_list, device=device)
        valloader = DataLoader(dataset=datasetVal, batch_size=1, shuffle=False)

        datasetTrain = Datasets_Bmode_patient_dir(train_list, device=device)
        trainloader = DataLoader(dataset=datasetTrain, batch_size=1, shuffle=False)

        epoch_start = time.time()
        # 测试集验证
        print("----------------------> test result <----------------------------")
        avg_test_loss, avg_test_auc, test_acc, test_sen, test_spe, test_threshold, test_truth_list, test_probability_list = val_patient_process(
            model, testloader, img_trans['val'], processor, device)
        test_model_preds = {
            'identifier': [extract_patient_identifier(item) for item in test_list[:, 0]],
            'label': test_truth_list,
            'pro': test_probability_list
        }
        data = pd.DataFrame(test_model_preds)
        data.to_excel(os.path.join('{}/test_pre.xlsx'.format(test_save_path)), sheet_name='sheet1')

        test_info = "Test: Loss: {:.4f}, AUC: {:.4f}, ACC: {:.4f}, SEN: {:.4f}, SPE: {:.4f}, Threshold: {:.4f}".format(
            avg_test_loss, avg_test_auc, test_acc, test_sen, test_spe, test_threshold
        )
        with open(os.path.join('{}/test_result.txt'.format(test_save_path)), 'w') as f:
            f.write(test_info)
        print(test_info)

        plt.figure(figsize=(8, 6), dpi=300)
        fpr, tpr, thresholds = roc_curve(test_truth_list, test_probability_list, pos_label=1)
        plt.plot(fpr, tpr, lw=2, label='(AUC = {:.3f})'.format(avg_test_auc))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('1-Specificity', fontsize=13)
        plt.ylabel('Sensitivity', fontsize=13)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2, left=0.2)
        plt.legend(fontsize=12, loc='lower right')
        plt.savefig(os.path.join(test_save_path, 'test_auc.png'))
        plt.close()

        print("*" * 100)

        # 验证集验证
        print("----------------------> valid result <----------------------------")
        avg_valid_loss, avg_valid_auc, valid_acc, valid_sen, valid_spe, val_threshold, val_truth_list, val_probability_list = val_patient_process(
            model, valloader, img_trans['val'], processor, device)

        val_model_preds = {
            'identifier': [extract_patient_identifier(item) for item in valid_list[:, 0]],
            'label': val_truth_list,
            'pro': val_probability_list
        }
        data = pd.DataFrame(val_model_preds)
        data.to_excel(os.path.join('{}/valid_pre.xlsx'.format(test_save_path)), sheet_name='sheet1')

        val_info = "Val: Loss: {:.4f}, AUC: {:.4f}, ACC: {:.4f}, SEN: {:.4f}, SPE: {:.4f}, Threshold: {:.4f}".format(
            avg_valid_loss, avg_valid_auc, valid_acc, valid_sen, valid_spe, val_threshold
        )
        with open(os.path.join('{}/val_result.txt'.format(test_save_path)), 'w') as f:
            f.write(val_info)
        print(val_info)

        plt.figure(figsize=(8, 6), dpi=300)
        fpr, tpr, thresholds = roc_curve(val_truth_list, val_probability_list, pos_label=1)
        plt.plot(fpr, tpr, lw=2, label='(AUC = {:.3f})'.format(avg_valid_auc))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('1-Specificity', fontsize=13)
        plt.ylabel('Sensitivity', fontsize=13)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2, left=0.2)
        plt.legend(fontsize=12, loc='lower right')
        plt.savefig(os.path.join(test_save_path, 'val_auc.png'))
        plt.close()

        print("*" * 100)

        # 训练集验证
        print("----------------------> train result <----------------------------")
        avg_train_loss, avg_train_auc, train_acc, train_sen, train_spe, train_threshold, train_truth_list, train_probability_list = val_patient_process(
            model, trainloader, img_trans['val'], processor, device)

        train_model_preds = {
            'identifier': [extract_patient_identifier(item) for item in train_list[:, 0]],
            'label': train_truth_list,
            'pro': train_probability_list
        }
        data = pd.DataFrame(train_model_preds)
        data.to_excel(os.path.join('{}/train_pre.xlsx'.format(test_save_path)), sheet_name='sheet1')

        train_info = "Train: Loss: {:.4f}, AUC: {:.4f}, ACC: {:.4f}, SEN: {:.4f}, SPE: {:.4f}, Threshold: {:.4f}".format(
            avg_train_loss, avg_train_auc, train_acc, train_sen, train_spe, train_threshold
        )
        with open(os.path.join('{}/train_result.txt'.format(test_save_path)), 'w') as f:
            f.write(train_info)
        print(train_info)

        plt.figure(figsize=(8, 6), dpi=300)
        fpr, tpr, thresholds = roc_curve(train_truth_list, train_probability_list, pos_label=1)
        plt.plot(fpr, tpr, lw=2, label='(AUC = {:.3f})'.format(avg_train_auc))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('1-Specificity', fontsize=13)
        plt.ylabel('Sensitivity', fontsize=13)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2, left=0.2)
        plt.legend(fontsize=12, loc='lower right')
        plt.savefig(os.path.join(test_save_path, 'train_auc.png'))
        plt.close()

        epoch_end = time.time()
        print("all time: {:.4f} s".format(epoch_end - epoch_start))

        # 添加当前折的测试集、验证集和训练集的真实标签和预测概率到汇总列表中
        all_test_truth_list.extend(test_truth_list)
        all_test_probability_list.extend(test_probability_list)
        all_val_truth_list.extend(val_truth_list)
        all_val_probability_list.extend(val_probability_list)
        all_train_truth_list.extend(train_truth_list)
        all_train_probability_list.extend(train_probability_list)
