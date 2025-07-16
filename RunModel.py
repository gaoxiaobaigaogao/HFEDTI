import os
import random

import sys  # 导入sys模块

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from config import hyperparameter
from model import HFEDTI
from utils.DataPrepare import get_kfold_data, shuffle_dataset
from utils.DataSetsFunction import CustomDataSet, collate_fn
from utils.EarlyStoping import EarlyStopping
from LossFunction import CELoss, FocalLoss
from utils.TestModel import test_model
from utils.ShowResult import show_result
import warnings
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score

warnings.filterwarnings("ignore", category=FutureWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_features(model, data_loader, save_dir):
    model.eval()
    drug_features = []
    target_features = []
    all_features = []
    all_labels = []

    with torch.no_grad():
        for drug, protein, label in data_loader:
            drug, protein, label = drug.cuda(), protein.cuda(), label.cuda()
            # forward
            drugembed = model.drug_embed(drug).permute(0, 2, 1)
            proteinembed = model.protein_embed(protein).permute(0, 2, 1)

            drugConv = model.Drug_CNNs(drugembed).permute(0, 2, 1)
            proteinConv = model.Protein_CNNs(proteinembed).permute(0, 2, 1)

            drug_lstm = model.drug_bilstm(drugConv)
            protein_lstm = model.protein_bilstm(proteinConv)

            drug_att, protein_att = model.deep_inter_attention(drug_lstm, protein_lstm)

            drugConv = drugConv * 0.5 + drug_att * 0.5
            proteinConv = proteinConv * 0.5 + protein_att * 0.5

            drugConv = drugConv.permute(0, 2, 1)
            proteinConv = proteinConv.permute(0, 2, 1)

            drugConv = model.Drug_max_pool(drugConv).squeeze(2)
            proteinConv = model.Protein_max_pool(proteinConv).squeeze(2)

            pair = torch.cat([drugConv, proteinConv], dim=1)

            all_features.append(pair.cpu().numpy())
            drug_features.append(drugConv.cpu().numpy())
            target_features.append(proteinConv.cpu().numpy())
            all_labels.append(label.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, 'feature.npy'), all_features)
    np.save(os.path.join(save_dir, 'label.npy'), all_labels)

    print('特征和标签已保存:', save_dir)

def run_model(SEED, DATASET, MODEL, K_Fold, LOSS):
    '''set random seed'''
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    '''init hyperparameters'''
    hp = hyperparameter()

    '''load dataset from text file'''
    assert DATASET in ["DrugBank", "Human","Celegans", "Unseen drug", "Unseen protein"]
    print("Train in " + DATASET)
    print("load data")
    dir_input = ('./DataSets/{}.txt'.format(DATASET))
    with open(dir_input, "r") as f:
        data_list = f.read().strip().split('\n')
    print("load finished")

    '''set loss function weight'''
    weight_loss = None

    '''shuffle data'''
    print("data shuffle")
    data_list = shuffle_dataset(data_list, SEED)

    '''split dataset to train&validation set and test set'''
    split_pos = len(data_list) - int(len(data_list) * 0.2)
    train_data_list = data_list[0:split_pos]
    test_data_list = data_list[split_pos:-1]
    print('Number of Train&Val set: {}'.format(len(train_data_list)))
    print('Number of Test set: {}'.format(len(test_data_list)))

    '''metrics'''
    Accuracy_List_stable, AUC_List_stable, AUPR_List_stable, Recall_List_stable, Precision_List_stable = [], [], [], [], []

    for i_fold in range(K_Fold):
        print('*' * 25, 'No.', i_fold + 1, '-fold', '*' * 25)

        train_dataset, valid_dataset = get_kfold_data(i_fold, train_data_list, k=K_Fold)
        train_dataset = CustomDataSet(train_dataset)
        valid_dataset = CustomDataSet(valid_dataset)
        test_dataset = CustomDataSet(test_data_list)
        train_size = len(train_dataset)

        train_dataset_loader = DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, num_workers=0,
                                          collate_fn=collate_fn, drop_last=True)
        valid_dataset_loader = DataLoader(valid_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,
                                          collate_fn=collate_fn, drop_last=True)
        test_dataset_loader = DataLoader(test_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,
                                         collate_fn=collate_fn, drop_last=True)

        """ create model"""
        model = MODEL(hp).to(DEVICE)

        """Initialize weights"""
        weight_p, bias_p = [], []
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]

        """create optimizer and scheduler"""
        optimizer = optim.AdamW(
            [{'params': weight_p, 'weight_decay': hp.weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=hp.Learning_rate)

        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.Learning_rate, max_lr=hp.Learning_rate*10, cycle_momentum=False,
                                                step_size_up=train_size // hp.Batch_size)
        # Initialize the loss function
        if LOSS == 'FocalLoss':
            Loss = FocalLoss(alpha=hp.focal_alpha, gamma=hp.focal_gamma, reduction='mean')
        else:
            Loss = CELoss(weight_CE=weight_loss, DEVICE=DEVICE)
        print(f"Training with loss function: {Loss}")
        """Output files"""
        save_path = "./" + DATASET + "/{}".format(i_fold+1)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_results = save_path + '/' + 'The_results_of_whole_dataset.txt'

        early_stopping = EarlyStopping(
            savepath=save_path, patience=hp.Patience, verbose=True, delta=0)

        """Start training."""
        print('Training...')
        for epoch in range(1, hp.Epoch + 1):
            if early_stopping.early_stop == True:
                break
            train_pbar = tqdm(
                enumerate(BackgroundGenerator(train_dataset_loader)),
                total=len(train_dataset_loader))

            """train"""
            train_losses_in_epoch = []
            model.train()
            for train_i, train_data in train_pbar:
                train_compounds, train_proteins, train_labels = train_data
                train_compounds = train_compounds.to(DEVICE)
                train_proteins = train_proteins.to(DEVICE)
                train_labels = train_labels.to(DEVICE)

                optimizer.zero_grad()

                predicted_interaction = model(train_compounds, train_proteins)
                train_loss = Loss(predicted_interaction, train_labels)
                train_losses_in_epoch.append(train_loss.item())
                train_loss.backward()
                optimizer.step()
                scheduler.step()
            train_loss_a_epoch = np.average(
                train_losses_in_epoch)  # 一次epoch的平均训练loss

            """valid"""
            valid_pbar = tqdm(
                enumerate(BackgroundGenerator(valid_dataset_loader)),
                total=len(valid_dataset_loader))
            valid_losses_in_epoch = []
            model.eval()
            Y, P, S = [], [], []
            with torch.no_grad():
                for valid_i, valid_data in valid_pbar:
                    valid_compounds, valid_proteins, valid_labels = valid_data

                    valid_compounds = valid_compounds.to(DEVICE)
                    valid_proteins = valid_proteins.to(DEVICE)
                    valid_labels = valid_labels.to(DEVICE)

                    valid_scores = model(valid_compounds, valid_proteins)
                    valid_loss = Loss(valid_scores, valid_labels)
                    valid_losses_in_epoch.append(valid_loss.item())
                    valid_labels = valid_labels.to('cpu').data.numpy()
                    valid_scores = F.softmax(
                        valid_scores, 1).to('cpu').data.numpy()
                    valid_predictions = np.argmax(valid_scores, axis=1)
                    valid_scores = valid_scores[:, 1]

                    Y.extend(valid_labels)
                    P.extend(valid_predictions)
                    S.extend(valid_scores)

            Precision_dev = precision_score(Y, P)
            Reacll_dev = recall_score(Y, P)
            Accuracy_dev = accuracy_score(Y, P)
            AUC_dev = roc_auc_score(Y, S)
            tpr, fpr, _ = precision_recall_curve(Y, S)
            PRC_dev = auc(fpr, tpr)
            valid_loss_a_epoch = np.average(valid_losses_in_epoch)

            epoch_len = len(str(hp.Epoch))
            print_msg = (f'[{epoch:>{epoch_len}}/{hp.Epoch:>{epoch_len}}] ' +
                         f'train_loss: {train_loss_a_epoch:.5f} ' +
                         f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                         f'valid_AUC: {AUC_dev:.5f} ' +
                         f'valid_PRC: {PRC_dev:.5f} ' +
                         f'valid_Accuracy: {Accuracy_dev:.5f} ' +
                         f'valid_Precision: {Precision_dev:.5f} ' +
                         f'valid_Reacll: {Reacll_dev:.5f} ')
            print(print_msg)

            '''save checkpoint and make decision when early stop'''
            # early_stopping(Accuracy_dev, model, epoch)
            early_stopping(Accuracy_dev, model, epoch, train_loss, valid_loss, AUC_dev, PRC_dev)

        '''load best checkpoint'''
        model.load_state_dict(torch.load(
            early_stopping.savepath + '/valid_best_checkpoint.pth'))

        '''test model'''
        trainset_test_stable_results, _, _, _, _, _ = test_model(
            model, train_dataset_loader, save_path, DATASET, Loss, DEVICE, dataset_class="Train", FOLD_NUM=1)
        validset_test_stable_results, _, _, _, _, _ = test_model(
            model, valid_dataset_loader, save_path, DATASET, Loss, DEVICE, dataset_class="Valid", FOLD_NUM=1)
        testset_test_stable_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_model(
            model, test_dataset_loader, save_path, DATASET, Loss, DEVICE, dataset_class="Test", FOLD_NUM=1)
        AUC_List_stable.append(AUC_test)
        Accuracy_List_stable.append(Accuracy_test)
        AUPR_List_stable.append(PRC_test)
        Recall_List_stable.append(Recall_test)
        Precision_List_stable.append(Precision_test)
        save_features(model, test_dataset_loader, save_dir=f'./{DATASET}/{i_fold + 1}')
        with open(save_path + '/' + "The_results_of_whole_dataset.txt", 'a') as f:
            f.write("Test the stable model" + '\n')
            f.write(trainset_test_stable_results + '\n')
            f.write(validset_test_stable_results + '\n')
            f.write(testset_test_stable_results + '\n')

    show_result(DATASET, Accuracy_List_stable, Precision_List_stable,
                Recall_List_stable, AUC_List_stable, AUPR_List_stable, Ensemble=False)

    ensemble_run_model(SEED, DATASET, K_Fold)

def ensemble_run_model(SEED, DATASET, K_Fold):
    '''设置随机种子'''
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    '''初始化超参数'''
    hp = hyperparameter()

    '''加载数据集'''
    assert DATASET in ["DrugBank", "Celegans", "Human"]
    print("Train in " + DATASET)
    print("加载数据中...")
    dir_input = f'./DataSets/{DATASET}.txt'
    with open(dir_input, "r") as f:
        data_list = f.read().strip().split('\n')
    print("数据加载完成")

    weight_loss = None

    '''数据集打乱顺序'''
    print("数据打乱中...")
    data_list = shuffle_dataset(data_list, SEED)

    '''划分训练集和测试集'''
    split_pos = len(data_list) - int(len(data_list) * 0.2)
    test_data_list = data_list[split_pos:]
    print('测试集大小: {}'.format(len(test_data_list)))

    save_path = f"./{DATASET}/ensemble"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    test_dataset = CustomDataSet(test_data_list)
    test_dataset_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                                     collate_fn=collate_fn, drop_last=False)

    # 初始化模型并加载权重
    models = []
    val_losses = []  # 用于存储每个模型在验证集上的损失
    for i in range(K_Fold):
        model = HFEDTI(hp).to(DEVICE)
        try:
            model.load_state_dict(torch.load(f'./{DATASET}/{i+1}/valid_best_checkpoint.pth',
                                             map_location=torch.device(DEVICE)))
        except FileNotFoundError as e:
            print('-' * 25 + '错误' + '-' * 25)
            print(f'无法加载预训练模型: {e}')
            print('请确保完成 K-Fold 训练')
            print('-' * 55)
            sys.exit(1)
        models.append(model)

        # 计算每个模型在验证集上的损失
        val_loss = 0.0
        val_data_list = data_list[i * len(data_list) // K_Fold: (i + 1) * len(data_list) // K_Fold]
        val_dataset = CustomDataSet(val_data_list)
        val_dataset_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0,
                                         collate_fn=collate_fn, drop_last=False)
        model.eval()
        with torch.no_grad():
            for data in val_dataset_loader:
                drug, protein, labels = data
                drug, protein, labels = drug.to(DEVICE), protein.to(DEVICE), labels.to(DEVICE)
                output = model(drug, protein)
                loss = CELoss(weight_CE=weight_loss, DEVICE=DEVICE)(output, labels)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_dataset_loader))

    # 根据验证集损失对模型进行排名
    val_losses = torch.FloatTensor(val_losses).to(DEVICE)
    ranks = torch.argsort(torch.argsort(val_losses))  # 获取每个模型的排名（损失越小，排名越高）
    weights = (K_Fold + 1 - ranks) / (K_Fold * (K_Fold + 1) / 2)  # 基于排名的权重分配
    weights = weights.to(DEVICE)

    # 定义损失函数
    Loss = CELoss(weight_CE=weight_loss, DEVICE=DEVICE)

    # 进行集成模型测试
    predictions = []
    for data in test_dataset_loader:
        drug, protein, labels = data
        # drug, protein = drug.to(DEVICE), protein.to(DEVICE)
        drug, protein = drug.to(DEVICE), protein.to(DEVICE)
        # 集成每个模型的特征输出
        features = []
        for model in models:
            model.eval()
            with torch.no_grad():
                feature = model.extract_features(drug, protein)
                features.append(feature)

        # 将 K 个模型的特征输出加权平均
        features = torch.stack(features)  # shape: [K, batch_size, feature_dim]
        aggregated_feature = (features * weights.view(-1, 1, 1)).sum(dim=0)  # 加权求和

        # 传递给全连接层进行预测
        fully1 = model.leaky_relu(model.fc1(aggregated_feature))
        fully2 = model.leaky_relu(model.fc2(fully1))
        fully3 = model.leaky_relu(model.fc3(fully2))
        aggregated_output = model.out(fully3)

        predictions.append(aggregated_output)

    # 将预测结果调整为张量格式
    predictions = torch.cat(predictions, dim=0)

    # 评估模型性能
    test_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_model(
        models, test_dataset_loader, save_path, DATASET, Loss, DEVICE, dataset_class="Test", save=True, FOLD_NUM=K_Fold
    )

    # 打印最终结果
    show_result(DATASET, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test, Ensemble=True)

if __name__ == '__main__':
    SEED = 114514
    DATASET = 'DrugBank'
    MODEL = HFEDTI
    K_Fold = 10
    LOSS = 'FocalLoss'

    run_model(SEED, DATASET, MODEL, K_Fold, LOSS)