from RunModel import run_model,ensemble_run_model
from model import HFEDTI

if __name__ == '__main__':
    SEED = 114514
    DATASET = 'DrugBank'  # 选择数据集
    MODEL = HFEDTI  # 替换为你的模型类
    K_Fold = 10
    LOSS = 'FocalLoss'

    run_model(SEED, DATASET, MODEL, K_Fold, LOSS)



