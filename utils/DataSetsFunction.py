import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

CHARPROTLEN = 25


def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN, dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X

class CustomDataSet(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __getitem__(self, item):
        return self.pairs[item]


    def __len__(self):
        return len(self.pairs)

def collate_fn(batch_data):
    """
       处理兼容两种数据格式的 collate_fn 函数：
       1. 第一种数据形式：SMILES、蛋白质序列、标签
       2. 第二种数据形式：化合物ID、蛋白质ID、SMILES、蛋白质序列、标签
       """
    N = len(batch_data)

    compound_max = 100  # SMILES最大长度
    protein_max = 1000  # 蛋白质序列最大长度

    # 初始化张量
    compound_new = torch.zeros((N, compound_max), dtype=torch.long)
    protein_new = torch.zeros((N, protein_max), dtype=torch.long)
    labels_new = torch.zeros(N, dtype=torch.long)

    # 初始化ID列表（仅当有化合物ID和蛋白质ID时使用）
    compound_ids = []
    protein_ids = []

    for i, pair in enumerate(batch_data):
        pair = pair.strip().split()

        if len(pair) == 3:  # 第一种形式
            compoundstr, proteinstr, label = pair
            compound_id, protein_id = None, None  # 无ID字段

        elif len(pair) == 5:  # 第二种形式
            compound_id, protein_id, compoundstr, proteinstr, label = pair
            compound_ids.append(compound_id)
            protein_ids.append(protein_id)

        else:
            raise ValueError(f"Invalid data format: {pair}")

        # 将SMILES和蛋白质序列映射到整数索引
        compoundint = torch.from_numpy(label_smiles(compoundstr, CHARISOSMISET, compound_max))
        compound_new[i] = compoundint

        proteinint = torch.from_numpy(label_sequence(proteinstr, CHARPROTSET, protein_max))
        protein_new[i] = proteinint

        # 标签是字符串，转为 float 再转为整数
        labels_new[i] = int(float(label))

    # 如果无ID字段，返回空列表
    if not compound_ids:
        compound_ids = ["N/A"] * N
    if not protein_ids:
        protein_ids = ["N/A"] * N
    return (compound_new, protein_new, labels_new)



# casestudy

'''class CustomDataSet(Dataset):
    def __init__(self, drug_data, protein_data, labels):
        """
        初始化数据集
        :param drug_data: 药物数据（SMILES 字符串列表）
        :param protein_data: 蛋白质数据（蛋白质序列列表）
        :param labels: 标签数据（0 或 1 的列表）
        """
        self.drug_data = drug_data  # SMILES 字符串列表
        self.protein_data = protein_data  # 蛋白质序列列表
        self.labels = labels  # 标签列表

        # 检查数据长度是否一致
        if len(self.drug_data) != len(self.protein_data) or len(self.drug_data) != len(self.labels):
            raise ValueError("drug_data, protein_data, and labels must have the same length!")

    def __getitem__(self, item):
        v_d = self.drug_data[item]  # SMILES 字符串
        v_p = self.protein_data[item]  # 蛋白质序列
        labels = self.labels[item]  # 标签
        return v_d, v_p, labels  # 返回元组

    def __len__(self):
        """
        返回数据集的大小
        :return: 数据集长度
        """
        return len(self.drug_data)


def collate_fn(batch_data):
    """
    处理批次数据的 collate_fn 函数。
    输入：batch_data 是一个列表，每个元素是一个元组 (v_d, v_p, labels)。
    输出：返回一个元组 (compound_new, protein_new, labels_new)。
    """
    N = len(batch_data)  # 批次大小

    compound_max = 100  # SMILES 最大长度
    protein_max = 1000  # 蛋白质序列最大长度

    # 初始化张量
    compound_new = torch.zeros((N, compound_max), dtype=torch.long)
    protein_new = torch.zeros((N, protein_max), dtype=torch.long)
    labels_new = torch.zeros(N, dtype=torch.long)

    for i, (v_d, v_p, labels) in enumerate(batch_data):
        # 假设 v_d 和 v_p 是字符串
        if isinstance(v_d, str):
            # 将 SMILES 转换为整数索引
            compoundint = torch.from_numpy(label_smiles(v_d, CHARISOSMISET, compound_max))
            compound_new[i] = compoundint

        if isinstance(v_p, str):
            # 将蛋白质序列转换为整数索引
            proteinint = torch.from_numpy(label_sequence(v_p, CHARPROTSET, protein_max))
            protein_new[i] = proteinint

        # 标签是字符串，转为 float 再转为整数
        labels_new[i] = int(float(labels))

    # 返回批次数据
    return compound_new, protein_new, labels_new
'''