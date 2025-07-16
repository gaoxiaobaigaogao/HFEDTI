import numpy as np
import torch
import os
import glob
import shutil
import json  # 增加缺失的json模块导入


def find_global_best(base_dir="./Human"):
    """
    在所有折中寻找全局最佳模型
    :param base_dir: 存放各折模型的根目录（假设目录结构为 base_dir/1/, base_dir/2/...）
    :return: 返回全局最佳模型路径
    """
    all_fold_dirs = sorted([d for d in glob.glob(f"{base_dir}/*") if os.path.isdir(d)],
                           key=lambda x: int(os.path.basename(x)))  # 增加折数排序

    candidates = []

    for fold_dir in all_fold_dirs:
        fold_num = os.path.basename(fold_dir)
        log_path = os.path.join(fold_dir, f"fold_{fold_num}_score_log.json")

        try:
            with open(log_path) as f:
                log_data = json.load(f)
        except FileNotFoundError:
            print(f"警告：跳过缺失日志文件的折 {fold_num}")
            continue

        model_path = os.path.join(
            fold_dir,
            f"valid_best_k_{fold_num}_epoch_{log_data['best_epoch']}.pth"
        )

        # 增加模型存在性校验
        if os.path.exists(model_path):
            candidates.append({
                "path": model_path,
                "fold": fold_num,
                "epoch": log_data["best_epoch"],
                "score": log_data["best_score"]
            })
        else:
            print(f"警告：折{fold_num}的最佳模型文件不存在")

    if not candidates:
        raise ValueError("没有找到有效的候选模型")

    global_best = max(candidates, key=lambda x: x["score"])

    global_dir = os.path.join(base_dir, "global_best")
    os.makedirs(global_dir, exist_ok=True)

    dst_path = os.path.join(
        global_dir,
        f"global_best_fold{global_best['fold']}_epoch{global_best['epoch']}.pth"
    )

    shutil.copyfile(global_best["path"], dst_path)
    return dst_path


class EarlyStopping:
    def __init__(self, savepath=None, patience=15, verbose=False, delta=0, num_n_fold=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = -np.inf
        self.early_stop = False
        self.delta = delta
        self.num_n_fold = num_n_fold
        self.savepath = savepath
        self.best_epoch = 0
        self.score_log = []
        self.train_loss_log = []  # 新增：训练集损失日志
        self.valid_log = []  # 新增：验证集日志

        # 自动创建保存目录
        if savepath and not os.path.exists(savepath):
            os.makedirs(savepath)

    def __call__(self, score, model, num_epoch, train_loss, valid_loss, auc, aupr):
        self.score_log.append((num_epoch, score))  # 始终记录所有epoch分数
        self.train_loss_log.append(train_loss)  # 新增：记录训练集损失
        self.valid_log.append((num_epoch, valid_loss, auc, aupr))  # 新增：记录验证集信息

        if self.best_score == -np.inf:
            self._update_checkpoint(score, model, num_epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self._update_checkpoint(score, model, num_epoch)
            self.counter = 0

    def _update_checkpoint(self, score, model, epoch):
        """封装状态更新逻辑"""
        self.best_score = score
        self.best_epoch = epoch
        self.save_checkpoint(model, epoch)

        if self.verbose:
            print(f'发现新最佳检查点（epoch {epoch}）: {score:.4f}')

    def save_checkpoint(self, model, epoch):
        """统一保存逻辑"""
        filename = f'valid_best_k_{self.num_n_fold}_epoch_{epoch}.pth'
        full_path = os.path.join(self.savepath, filename)

        # 删除旧的最佳模型（如果有）
        old_files = glob.glob(os.path.join(self.savepath, f"valid_best_k_{self.num_n_fold}_epoch_*.pth"))
        for f in old_files:
            try:
                os.remove(f)
            except Exception as e:
                print(f"删除旧模型失败：{str(e)}")

        # 保存新模型
        torch.save(model.state_dict(), full_path)

        # 复制一份并重命名为valid_best_checkpoint.pth
        copy_path = os.path.join(self.savepath, "valid_best_checkpoint.pth")
        try:
            shutil.copyfile(full_path, copy_path)
            print(f"成功复制模型到 {copy_path}")
        except Exception as e:
            print(f"复制模型失败：{str(e)}")

        # 保存日志文件
        log_data = {
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "full_log": self.score_log
        }
        with open(os.path.join(self.savepath, f"fold_{self.num_n_fold}_score_log.json"), "w") as f:
            json.dump(log_data, f, indent=2)

    def save_logs_to_txt(self):
        # 保存测试集结果
        with open(os.path.join(self.savepath, f"test_results_fold_{self.num_n_fold}.txt"), "w") as f:
            f.write("| 最优轮数 | AUC | AUPR | Test Loss | Accuracy | Recall | Precision |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
            # 这里暂时无法获取测试集损失、准确率、召回率、精确率，需要在RunModel.py中传递
            # 假设在RunModel.py中调用时会补充这些信息
            f.write(f"| {self.best_epoch} | {self.best_score} | | | | | |\n")

        # 保存训练集结果
        with open(os.path.join(self.savepath, f"train_results_fold_{self.num_n_fold}.txt"), "w") as f:
            f.write("| 轮数 | Train Loss |\n")
            f.write("| --- | --- |\n")
            for epoch, loss in enumerate(self.train_loss_log, start=1):
                f.write(f"| {epoch} | {loss} |\n")

        # 保存验证集结果
        with open(os.path.join(self.savepath, f"valid_results_fold_{self.num_n_fold}.txt"), "w") as f:
            f.write("| 轮数 | AUC | AUPR | Val Loss |\n")
            f.write("| --- | --- | --- | --- |\n")
            for epoch, valid_loss, auc, aupr in self.valid_log:
                f.write(f"| {epoch} | {auc} | {aupr} | {valid_loss} |\n")