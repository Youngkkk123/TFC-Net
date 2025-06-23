import os
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
from datagenerator import Datasets_Bmode_roi_patient_allimg
from utils_index import calcAUC, calcACCSENSPE
from data_augmentation import get_data_transform_2D_vit
from Loss import FocalLoss
import random
from transformers import AutoModel
from peft import LoraConfig, get_peft_model


def train_process(model, optimizer, datasetloader, processor, device):
    model.train()
    optimizer.zero_grad()

    total_loss = 0
    truth_list = []
    probability_list = []

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
    with torch.no_grad():
        text_outputs = model.get_text_features(**text_inputs)
        text_embeds = F.normalize(text_outputs, p=2, dim=1)  # 确保L2归一化

    criterion = FocalLoss(gamma=2, alpha=0.75)

    for j, datas in enumerate(datasetloader):
        images = datas['image'].to(device)
        labels = datas['label'].to(device)

        # 处理图像，获取图像嵌入
        image_inputs = processor(images=images, return_tensors="pt").to(device)
        image_embeds = model.get_image_features(**image_inputs)
        image_embeds = F.normalize(image_embeds, p=2, dim=1)  # 确保L2归一化

        # cosine similarity as logits
        cos_similarity = (
            torch.matmul(text_embeds.detach().clone(), image_embeds.t().to(text_embeds.device)) * model.base_model.logit_scale.exp()
            + model.base_model.logit_bias
        ).T

        loss = criterion(cos_similarity, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item() * images.size(0)

        truth_list.extend(labels.cpu().detach().numpy())
        # 只在计算概率时应用softmax
        probabilities = torch.softmax(cos_similarity, dim=1)
        probability = probabilities[:, 1]
        probability_list.extend(probability.cpu().detach().numpy())

    avg_auc, threshold = calcAUC(truth_list, probability_list)
    acc, sen, spe = calcACCSENSPE(truth_list, probability_list, threshold)
    avg_loss = total_loss / len(datasetloader.dataset)

    return model, avg_loss, avg_auc, acc, sen, spe, threshold


def val_process(model, datasetloader, processor, device):
    with torch.no_grad():
        model.eval()

        total_loss = 0
        truth_list = []
        probability_list = []

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

        criterion = FocalLoss(gamma=2, alpha=0.75)

        for j, datas in enumerate(datasetloader):
            images = datas['image'].to(device)
            labels = datas['label'].to(device)

            # 处理图像，获取图像嵌入
            image_inputs = processor(images=images, return_tensors="pt").to(device)
            image_embeds = model.get_image_features(**image_inputs)
            image_embeds = F.normalize(image_embeds, p=2, dim=1)  # 确保L2归一化

            # 计算图像与文本之间的余弦相似度矩阵 [batch_size, 2]
            cos_similarity = (
                    torch.matmul(text_embeds.detach().clone(),
                                 image_embeds.t().to(text_embeds.device)) * model.base_model.logit_scale.exp()
                    + model.base_model.logit_bias
            ).T

            # 应用softmax
            probabilities = torch.softmax(cos_similarity, dim=1)

            # 计算交叉熵损失
            loss = criterion(probabilities, labels)

            total_loss += loss.item() * images.size(0)

            truth_list.extend(labels.cpu().detach().numpy())
            probability = probabilities[:, 1]
            probability_list.extend(probability.cpu().detach().numpy())

        avg_auc, threshold = calcAUC(truth_list, probability_list)
        acc, sen, spe = calcACCSENSPE(truth_list, probability_list, threshold)
        avg_loss = total_loss / len(datasetloader.dataset)

    return model, avg_loss, avg_auc, acc, sen, spe, threshold


def custom_collate_fn(batch):
    images = []
    labels = []
    paths = []
    for item in batch:
        images.append(item['image'])
        labels.append(item['label'])
        paths.append(item['path'])
    return {
        'image': torch.stack(images),
        'label': torch.tensor(labels),
        'path': paths
    }


class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, datasets, batch_size):
        self.datasets = datasets
        self.batch_size = batch_size
        self.pos_indices = []
        self.neg_indices = []

        # 遍历所有数据集，获取正样本和负样本的索引
        start_index = 0
        for dataset in self.datasets:
            labels = dataset.labels
            for i, label in enumerate(labels):
                if label == 1:
                    self.pos_indices.append(start_index + i)
                else:
                    self.neg_indices.append(start_index + i)
            start_index += len(labels)

        self.num_batches = min(len(self.pos_indices), len(self.neg_indices)) // (self.batch_size // 2)

    def __iter__(self):
        for _ in range(self.num_batches):
            pos_batch = random.sample(self.pos_indices, self.batch_size // 2)
            neg_batch = random.sample(self.neg_indices, self.batch_size // 2)
            yield pos_batch + neg_batch

    def __len__(self):
        return self.num_batches


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    foldnum = 5
    epoch = 100
    patchsize = (224, 224)
    batchsize = 8
    lr = 1e-4

    for seed in [68]:
        print("\n**********{}***********\n".format(seed))
        seed_num = seed
        torch.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)
        random.seed(seed_num)
        np.random.seed(seed_num)
        os.environ['PYTHONHASHSEED'] = str(seed_num)
        from torch.backends import cudnn
        cudnn.benchmark = False
        cudnn.deterministic = True

        data_path = ""
        model_save_path = ""
        os.makedirs(model_save_path, exist_ok=True)

        model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device)
        processor = AutoModel.from_pretrained("google/siglip-base-patch16-224", do_rescale=False)

        config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=['q_proj', 'k_proj', 'v_proj'],
            bias="none"
        )

        # 对图像编码器应用LoRA
        model.vision_model = get_peft_model(model.vision_model, config)
        # 对文本编码器应用LoRA
        model.text_model = get_peft_model(model.text_model, config)

        model = model.to(device)
        # print(model)

        model_save = model_save_path
        os.makedirs(model_save, exist_ok=True)

        train_path_pos = os.path.join(data_path, "train_pos.npy")
        train_path_neg = os.path.join(data_path, "train_neg.npy")
        val_path = os.path.join(data_path, "valid.npy")

        train_list_pos = np.load(train_path_pos, allow_pickle=True)
        train_list_neg = np.load(train_path_neg, allow_pickle=True)
        val_list = np.load(val_path, allow_pickle=True)

        print('train_pos list length:', len(train_list_pos))
        print('train_neg list length:', len(train_list_neg))
        print('valid list length:', len(val_list))

        imgTrans = get_data_transform_2D_vit(patchsize)

        # 创建数据集
        datasetTrain_pos = Datasets_Bmode_roi_patient_allimg(train_list_pos, transform=imgTrans['train'], device=device)
        datasetTrain_neg = Datasets_Bmode_roi_patient_allimg(train_list_neg, transform=imgTrans['train'], device=device)
        datasetVal = Datasets_Bmode_roi_patient_allimg(val_list, transform=imgTrans['val'], device=device)

        # 创建 BalancedBatchSampler
        balanced_sampler = BalancedBatchSampler([datasetTrain_pos, datasetTrain_neg], batch_size=batchsize)

        # 创建 DataLoader
        trainloader = DataLoader(dataset=ConcatDataset([datasetTrain_pos, datasetTrain_neg]), batch_sampler=balanced_sampler, collate_fn=custom_collate_fn)
        valloader = DataLoader(dataset=datasetVal, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100, eta_min=0.000001)

        history = []

        best_avg_valid_loss = 9999
        best_avg_valid_auc = 0
        best_acc = 0
        best_sen = 0
        best_spe = 0
        best_epoch = 0
        best_threshold = 0
        for epoch_i in range(epoch):
            epoch_start = time.time()
            model, avg_train_loss, avg_train_auc, train_acc, train_sen, train_spe, train_threshold = train_process(
                model, optimizer, trainloader, processor, device)
            model, avg_valid_loss, avg_valid_auc, valid_acc, valid_sen, valid_spe, val_threshold = val_process(model,
                                                                                                               valloader,
                                                                                                               processor,
                                                                                                               device)
            epoch_end = time.time()

            history.append([avg_train_loss, avg_valid_loss, avg_train_auc, avg_valid_auc])

            # 修改保存模型的条件
            if (epoch_i + 1) > 5 and best_avg_valid_loss >= avg_valid_loss:
                best_avg_valid_loss = avg_valid_loss
                best_avg_valid_auc = avg_valid_auc
                best_acc = valid_acc
                best_sen = valid_sen
                best_spe = valid_spe
                best_epoch = epoch_i + 1
                best_threshold = val_threshold
                torch.save(model, model_save + '/' + "Best_model.pt")
                train_info = 'Epoch: {:03d}, Training: Loss: {:.4f}, AUC: {:.4f}, ACC: {:.4f}, SEN: {:.4f}, SPE: {:.4f}, threshold: {:.4f} \nValidation: Loss: {:.4f}, AUC: {:.4f}, ACC: {:.4f}, SEN: {:.4f}, SPE: {:.4f}, threshold: {:.4f}, Time: {:.4f}s, \nlearning rate: {:.7f}'.format(
                    epoch_i + 1, avg_train_loss, avg_train_auc, train_acc, train_sen, train_spe, train_threshold,
                    avg_valid_loss, avg_valid_auc, valid_acc, valid_sen, valid_spe, best_threshold,
                    epoch_end - epoch_start, optimizer.param_groups[0]['lr']
                )
                with open(model_save + '/' + 'train_metrics.txt', 'w') as f:
                    f.write(train_info)

            print(
                "Epoch: {:03d}, Training: Loss: {:.4f}, AUC: {:.4f}, ACC: {:.4f}, SEN: {:.4f}, SPE: {:.4f} \n\t\t\tValidation: Loss: {:.4f}, AUC: {:.4f}, ACC: {:.4f}, SEN: {:.4f}, SPE: {:.4f}, Time: {:.4f}s, \n\t\t\tlearning rate: {:.7f}".format(
                    epoch_i + 1, avg_train_loss, avg_train_auc, train_acc, train_sen, train_spe,
                    avg_valid_loss, avg_valid_auc, valid_acc, valid_sen, valid_spe,
                    epoch_end - epoch_start, optimizer.param_groups[0]['lr']
                ))
            print(
                "Best Epoch: {:03d}, Validation: Loss: {:.4f}, AUC: {:.4f}, ACC: {:.4f}, SEN: {:.4f}, SPE: {:.4f}".format(
                    best_epoch, best_avg_valid_loss, best_avg_valid_auc, best_acc, best_sen, best_spe
                ))

        torch.save(history, model_save + '/history.pt')

        history = np.array(history)
        fig1 = plt.figure()
        plt.plot(history[:, 0])
        plt.plot(history[:, 1])
        plt.legend(['Tr Loss', 'Val Loss'])
        plt.xlabel('Epoch Number')
        plt.ylabel('Loss')
        fig1.savefig(model_save + '/loss_curve.png')

        fig2 = plt.figure()
        plt.plot(history[:, 2])
        plt.plot(history[:, 3])
        plt.legend(['Tr AUC', 'Val AUC'])
        plt.xlabel('Epoch Number')
        plt.ylabel('AUC')
        fig2.savefig(model_save + '/AUC_curve.png')

    txr_path = os.path.join(model_save_path, 'train_param.txt')  # 保存训练参数和模型
    with open(txr_path, 'w') as writer:
        writer.write('lr:{}\n'.format(lr))
        writer.write('epochs:{}\n'.format(epoch))
        writer.write('batch_size:{}\n'.format(batchsize))
        writer.write('lora_r:{}\n'.format(config.r))
        writer.write('lora_alpha:{}\n'.format(config.lora_alpha))
        writer.write('lora_dropout:{}\n'.format(config.lora_dropout))
        writer.write('optimizer:{}\n'.format(optimizer))
        writer.write('loss_func: Focalloss\n')
        writer.write('model:{}\n'.format(model))
        writer.close()
