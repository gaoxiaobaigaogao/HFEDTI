
class hyperparameter():
    def __init__(self):
        self.Learning_rate = 1e-5
        self.Epoch = 100
        self.Batch_size = 16
        self.Patience = 50
        self.decay_interval = 10
        self.lr_decay = 0.5
        self.weight_decay = 1e-4
        self.embed_dim = 64
        self.protein_kernel = [4, 8, 12]
        self.drug_kernel = [4, 6, 8]
        self.conv = 40
        self.char_dim = 64
        self.loss_epsilon = 1
        self.loss_epsilon = 1e-5
        self.focal_alpha = 0.25  # Focal Loss 的 alpha
        self.focal_gamma = 2.0  # Focal Loss 的 gamma
        self.drug_in_channels = 64  # 根据实际需求设置适当的值
        self.drug_out_channels = 160  # 根据实际需求设置适当的值
        self.protein_in_channels = 64  # 设置蛋白质的输入通道数
        self.protein_out_channels = 160  # 设置蛋白质的输出通道数
        self.attention_dim = self.conv * 4  # 计算 attention_dim
        self.distance_metric = 'euclidean'  # 或 'cosine'
        self.mix_attention_head = 8  # 设置 mix_attention_head 的默认值
        self.loss_epsilon = 1.0
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        self.temperature = 0.5  # 添加 temperature 属性
        self.alpha = 1.0
        self.lstm_hidden_size = 256  # 添加LSTM隐藏层维度
        self.char_dim = 128
