"""
1. ER值
2. J-MAPE评估
3. 消融实验
4. 总体实验
5. 评估指标
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import matplotlib.pyplot as plt
import os
import json
import math
from typing import List, Dict, Tuple, Any, Optional
import random
from tqdm import tqdm
import warnings
import pickle
import torch.nn.functional as F
from dataclasses import dataclass, field
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.stats import pearsonr

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

# ===================== 全局配置 =====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 固定使用CPU，适用于Windows + PyCharm本地实验。
DEVICE = torch.device("cpu")

# 避免PyTorch默认占满全部CPU线程；可根据电脑性能自行调整。
_CPU_COUNT = os.cpu_count() or 1
torch.set_num_threads(max(1, _CPU_COUNT // 2))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # 若PyTorch并行环境已初始化，则保持当前设置。
    pass

print(f"使用设备: CPU，计算线程数: {torch.get_num_threads()}")

# 数据集路径配置
DATA_PATHS = {
    "Helpdesk": r"C:\Users\28726\Desktop\Helpdesk_Final_Numeric.csv",
    "Sepsis": r"C:\Users\28726\Desktop\Spesis\Sepsis_Core_Data.csv",
    "BPIC2012": r"C:\Users\28726\Desktop\BPIC2012\BPI_Challenge_2012_Processed.csv",
    "HEL": r"C:\Users\28726\Desktop\helpdesk2.csv",
    "RTF": r"C:\Users\28726\Desktop\RTF\RTFMP_Core_Data.csv",
    "BPIC2017": r"C:\Users\28726\Desktop\BPIC2017\BPI_Challenge_2017.csv",
    "BPIC2019": r"C:\Users\28726\Desktop\BPIC2019\BPI_Challenge_2019.csv",
    "BPIC2020": r"C:\Users\28726\Desktop\BPIC2020\BPI_Challenge_2020.csv"
}

# 数据集列名映射
COLUMN_MAPPINGS = {
    "Helpdesk": {"case_col": "CaseID", "activity_col": "ActivityII", "time_col": "CompleteTimestamp"},
    "Sepsis": {"case_col": "caseID", "activity_col": "activity", "time_col": "timestamp"},
    "BPIC2012": {"case_col": "trace_id", "activity_col": "event_concept_name", "time_col": "time_timestamp"},
    "HEL": {"case_col": "CaseID", "activity_col": "ActivityID", "time_col": "CompleteTimestamp"},
    "RTF": {"case_col": "caseID", "activity_col": "activity", "time_col": "timestamp"},
    "BPIC2017": {"case_col": "caseID", "activity_col": "activity", "time_col": "timestamp"},
    "BPIC2019": {"case_col": "caseID", "activity_col": "activity", "time_col": "timestamp"},
    "BPIC2020": {"case_col": "caseID", "activity_col": "activity", "time_col": "timestamp"}
}


#动态配置类
@dataclass
class DynamicConfig:
    train_size: int
    use_contrastive: bool = False
    use_attention: bool = False
    dataset_name: str = "unknown"
    aggregation_type: str = "equisize"
    seed: int = SEED

    # 实验框架：100个时间步；前25/50/75训练；随后25步测试
    base_intervals: int = 100
    window_size: int = 10
    forecast_horizon: int = 1
    train_horizon: int = 5
    test_intervals: int = 25
    alpha: float = 0.5
    edge_threshold: float = 0.5
    jmape_epsilon: float = 1.0
    use_log_transform: bool = True

    # 对比学习参数
    contrastive_dim: int = 128
    contrastive_proj_dim: int = 64
    contrastive_temperature: float = 0.1
    contrastive_epochs: int = 30
    contrastive_lr: float = 1e-3
    augmentation_prob: float = 0.1
    noise_level: float = 0.05
    finetune_contrastive: bool = True

    # 注意力参数
    num_attention_heads: int = 4
    attention_dropout: float = 0.1

    # 长期递归训练参数
    teacher_forcing_start: float = 1.0
    teacher_forcing_end: float = 0.2
    grad_clip: float = 1.0

    # 结果目录
    result_dir: str = field(default="")

    def __post_init__(self):
        if self.train_size not in (25, 50, 75):
            raise ValueError("train_size必须为25、50或75")

        # 主实验保持相同模型容量，只改变历史观测长度，增强可比性
        self.hidden_dim = 64
        self.num_layers = 1
        self.dropout = 0.0
        self.lr = 3e-4
        self.batch_size = 4 if self.train_size > 25 else 2
        self.l2_lambda = 1e-5
        self.high_freq_threshold = 1
        self.high_freq_weight = 1.2

        if self.train_size == 25:
            self.epochs = 80
            self.patience = 12
            self.val_steps = 5
            self.contrastive_dim = 64
            self.contrastive_proj_dim = 32
            self.contrastive_epochs = 20
        elif self.train_size == 50:
            self.epochs = 100
            self.patience = 15
            self.val_steps = 8
        else:
            self.epochs = 120
            self.patience = 18
            self.val_steps = 10

        # 兼容旧代码字段；正式预测不再进行人工缩放
        self.scale_factor = 1.0
        self.val_window = self.val_steps

        if self.hidden_dim % self.num_attention_heads != 0:
            self.num_attention_heads = 2

        if not self.result_dir:
            mode_str = "contrastive" if self.use_contrastive else "basic"
            mode_str += "_attn" if self.use_attention else ""
            self.result_dir = os.path.join(
                "results", self.dataset_name, self.aggregation_type,
                mode_str, f"train_{self.train_size}", f"seed_{self.seed}"
            )
        os.makedirs(self.result_dir, exist_ok=True)

    def get_model_params(self) -> Dict:
        return {
            "hidden_dim": self.hidden_dim, "num_layers": self.num_layers,
            "dropout": self.dropout, "lr": self.lr, "epochs": self.epochs,
            "batch_size": self.batch_size, "l2_lambda": self.l2_lambda,
            "patience": self.patience, "scale_factor": self.scale_factor,
            "val_window": self.val_window, "val_steps": self.val_steps,
            "window_size": self.window_size, "train_horizon": self.train_horizon,
            "test_intervals": self.test_intervals,
            "teacher_forcing_start": self.teacher_forcing_start,
            "teacher_forcing_end": self.teacher_forcing_end,
            "use_log_transform": self.use_log_transform,
            "edge_threshold": self.edge_threshold, "alpha": self.alpha,
            "use_attention": self.use_attention,
            "num_attention_heads": self.num_attention_heads,
            "attention_dropout": self.attention_dropout,
            "use_contrastive": self.use_contrastive,
            "contrastive_dim": self.contrastive_dim,
            "contrastive_proj_dim": self.contrastive_proj_dim,
            "finetune_contrastive": self.finetune_contrastive,
            "dataset_name": self.dataset_name,
            "aggregation_type": self.aggregation_type,
            "seed": self.seed,
        }


#数据加载模块
class ProcessDataLoader:
    """数据加载器"""

    def __init__(self, dataset_name: str): # 初始化，加载数据
        self.dataset_name = dataset_name
        self.file_path = DATA_PATHS[dataset_name]
        col_map = COLUMN_MAPPINGS[dataset_name]
        self.case_col = col_map["case_col"]
        self.activity_col = col_map["activity_col"]
        self.time_col = col_map["time_col"]

        self.df = self._load_preprocess()
        self.activity2id, self.id2activity = self._create_activity_mapping()
        self.num_activities = len(self.activity2id)
        print(f"数据集: {dataset_name}, 案例数: {self.df[self.case_col].nunique()}, 活动数: {self.num_activities}")

    def _load_preprocess(self) -> pd.DataFrame:# 加载CSV并预处理
        df = pd.read_csv(self.file_path)
        for col in [self.case_col, self.activity_col, self.time_col]:
            if col not in df.columns:
                raise ValueError(f"列 {col} 不在数据集中")
            # 统一转换为UTC，兼容带时区以及冬令时/夏令时偏移
        df[self.time_col] = pd.to_datetime(
            df[self.time_col],
            errors="coerce",
            utc=True,
        )
        invalid_time_count = df[self.time_col].isna().sum()
        if invalid_time_count > 0:
            print(
                f"警告：时间戳解析失败 {invalid_time_count} 条，"
                f"这些记录将被删除"
            )
        # 删除空值/排序
        df = df.dropna(subset=[self.case_col, self.activity_col, self.time_col])
        df = df.sort_values([self.case_col, self.time_col]).reset_index(drop=True)
        return df

    def _create_activity_mapping(self) -> Tuple[Dict[str, int], Dict[int, str]]:# 创建活动↔ID映射
        activities = sorted(self.df[self.activity_col].unique())
        activity2id = {act: i for i, act in enumerate(activities)}
        id2activity = {i: act for act, i in activity2id.items()}
        return activity2id, id2activity

    def extract_direct_follows(self) -> pd.DataFrame:# 提取直接后继关系
        direct_follows = []
        trace_counter = 0
        trace_map = {}

        for case, group in self.df.groupby(self.case_col):
            trace_map[case] = trace_counter
            trace_counter += 1
            activities = group[self.activity_col].tolist()
            timestamps = group[self.time_col].tolist()

            for i in range(len(activities) - 1):
                direct_follows.append({
                    "trace_no": trace_map[case],
                    "a1": activities[i],
                    "a2": activities[i + 1],
                    "timestamp": timestamps[i + 1]
                })

        sap_df = pd.DataFrame(direct_follows)
        sap_df = sap_df.sort_values("timestamp").reset_index(drop=True)
        sap_df["a1_encoded"] = sap_df["a1"].map(self.activity2id)
        sap_df["a2_encoded"] = sap_df["a2"].map(self.activity2id)
        print(f"直接后继关系数量: {len(sap_df)}")
        return sap_df


#DFG聚合模块
class DFAggregator:
    """DFG聚合器：保留等时与均衡两种聚合，但使用向量化计数。"""

    @staticmethod
    def equitemporal_aggregation(sap_df: pd.DataFrame, num_intervals: int,
                                 num_activities: int) -> np.ndarray:
        M = np.zeros((num_intervals, num_activities, num_activities), dtype=np.float32)
        if sap_df.empty:
            return M
        timestamps = pd.to_datetime(sap_df["timestamp"]).astype("int64").to_numpy()
        ts_min, ts_max = timestamps.min(), timestamps.max()
        if ts_min == ts_max:
            interval_ids = np.zeros(len(sap_df), dtype=np.int64)
        else:
            edges = np.linspace(ts_min, ts_max, num_intervals + 1)
            interval_ids = np.searchsorted(edges, timestamps, side="right") - 1
            interval_ids = np.clip(interval_ids, 0, num_intervals - 1)
        a1 = sap_df["a1_encoded"].to_numpy(dtype=np.int64)
        a2 = sap_df["a2_encoded"].to_numpy(dtype=np.int64)
        np.add.at(M, (interval_ids, a1, a2), 1.0)
        return M

    @staticmethod
    def equisize_aggregation(sap_df: pd.DataFrame, num_intervals: int,
                             num_activities: int) -> np.ndarray:
        M = np.zeros((num_intervals, num_activities, num_activities), dtype=np.float32)
        total = len(sap_df)
        if total == 0:
            return M
        interval_ids = np.minimum(
            np.arange(total, dtype=np.int64) * num_intervals // total,
            num_intervals - 1,
        )
        a1 = sap_df["a1_encoded"].to_numpy(dtype=np.int64)
        a2 = sap_df["a2_encoded"].to_numpy(dtype=np.int64)
        np.add.at(M, (interval_ids, a1, a2), 1.0)
        return M


#对比学习模块
class DataAugmentation:
    @staticmethod
    def random_edge_dropout(dfg_matrix: np.ndarray, p: float = 0.1) -> np.ndarray:
        mask = np.random.binomial(1, 1 - p, size=dfg_matrix.shape)
        return dfg_matrix * mask

    @staticmethod
    def random_edge_perturbation(dfg_matrix: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
        noise = np.random.normal(0, noise_level, size=dfg_matrix.shape)
        return np.maximum(dfg_matrix + noise, 0)

    @staticmethod
    def random_node_masking(dfg_matrix: np.ndarray, mask_prob: float = 0.1) -> np.ndarray:
        n = dfg_matrix.shape[0]
        mask_nodes = np.random.rand(n) < mask_prob
        masked_dfg = dfg_matrix.copy()
        for i in range(n):
            if mask_nodes[i]:
                masked_dfg[i, :] = 0
                masked_dfg[:, i] = 0
        return masked_dfg

    @staticmethod
    def generate_positive_pair(dfg_matrix: np.ndarray, aug_prob: float = 0.1,
                               noise_level: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        aug_type = random.choice(['dropout', 'perturbation', 'masking'])
        if aug_type == 'dropout':
            view1 = DataAugmentation.random_edge_dropout(dfg_matrix, aug_prob)
            view2 = DataAugmentation.random_edge_perturbation(dfg_matrix, noise_level)
        elif aug_type == 'perturbation':
            view1 = DataAugmentation.random_edge_perturbation(dfg_matrix, noise_level)
            view2 = DataAugmentation.random_edge_dropout(dfg_matrix, aug_prob)
        else:
            view1 = DataAugmentation.random_node_masking(dfg_matrix, aug_prob)
            view2 = DataAugmentation.random_edge_perturbation(dfg_matrix, noise_level)
        return view1, view2


class ContrastiveDFGEncoder(nn.Module):
    def __init__(self, num_activities: int, hidden_dim: int = 128, proj_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.num_activities = num_activities

        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        cnn_output_size = 64
        self.encoder = nn.Sequential(
            nn.Linear(cnn_output_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, dfg_matrix: torch.Tensor, return_projection: bool = True):
        batch_size = dfg_matrix.shape[0]
        x = dfg_matrix.unsqueeze(1)
        cnn_features = self.cnn_encoder(x)
        cnn_features = cnn_features.view(batch_size, -1)
        h = self.encoder(cnn_features)
        if return_projection:
            z = self.projector(h)
            return z
        return h

    def encode(self, dfg_matrix: torch.Tensor) -> torch.Tensor:
        batch_size = dfg_matrix.shape[0]
        x = dfg_matrix.unsqueeze(1)
        cnn_features = self.cnn_encoder(x)
        cnn_features = cnn_features.view(batch_size, -1)
        h = self.encoder(cnn_features)
        return h


class StableNTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.shape[0]
        if batch_size < 2:
            raise ValueError("NT-Xent loss requires batch_size >= 2")
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        features = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = torch.matmul(features, features.T)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=features.device)
        pair_ids = torch.arange(batch_size, device=features.device).repeat(2)
        positives_mask = pair_ids.unsqueeze(0).eq(pair_ids.unsqueeze(1)) & ~mask
        negatives_mask = ~positives_mask & ~mask
        positives = similarity_matrix[positives_mask].view(2 * batch_size, 1)
        negatives = similarity_matrix[negatives_mask].view(2 * batch_size, -1)
        logits = torch.cat([positives, negatives], dim=1) / self.temperature
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=features.device)
        return self.cross_entropy(logits, labels)


#预测模型定义
class UnifiedDFGPredictor(nn.Module):
    """统一DFG预测模型。输出为非负的log1p频次。"""

    def __init__(self, num_activities: int, config: DynamicConfig,
                 contrastive_encoder: ContrastiveDFGEncoder = None):
        super().__init__()
        self.config = config
        self.num_activities = num_activities
        self.use_attention = config.use_attention
        self.use_contrastive = config.use_contrastive

        if self.use_contrastive:
            self.contrastive_encoder = contrastive_encoder or ContrastiveDFGEncoder(
                num_activities, config.contrastive_dim, config.contrastive_proj_dim
            )
            lstm_input_dim = config.contrastive_dim
        else:
            lstm_input_dim = num_activities * num_activities

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim, hidden_size=config.hidden_dim,
            num_layers=config.num_layers, batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        if self.use_attention:
            self.multihead_attn = nn.MultiheadAttention(
                embed_dim=config.hidden_dim, num_heads=config.num_attention_heads,
                dropout=config.attention_dropout, batch_first=True,
            )
            self.attn_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.attn_layer_norm = nn.LayerNorm(config.hidden_dim)

        self.output_layer = nn.Linear(config.hidden_dim, num_activities * num_activities)
        self.dropout_layer = nn.Dropout(config.dropout)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor, hidden=None, return_attention=False):
        if self.use_contrastive:
            if x.ndim == 3:
                x = x.unsqueeze(1)
            batch_size, seq_len, n, _ = x.shape
            flat = x.reshape(batch_size * seq_len, n, n)
            embeddings = self.contrastive_encoder.encode(flat)
            x = embeddings.reshape(batch_size, seq_len, -1)

        lstm_out, hidden = self.lstm(x, hidden)
        attn_weights = None
        if self.use_attention:
            attn_out, attn_weights = self.multihead_attn(
                lstm_out, lstm_out, lstm_out, need_weights=return_attention
            )
            lstm_out = self.attn_layer_norm(lstm_out + self.attn_proj(attn_out))

        out = self.softplus(self.output_layer(self.dropout_layer(lstm_out)))
        if return_attention:
            return out, hidden, attn_weights
        return out, hidden


#数据集模块
class ContrastiveDataset(Dataset):
    def __init__(self, dfg_series: np.ndarray, augmentation_prob: float = 0.1,
                 noise_level: float = 0.05, use_log_transform: bool = True):
        self.dfg_series = np.log1p(dfg_series).astype(np.float32) if use_log_transform else dfg_series.astype(np.float32)
        self.augmentation_prob = augmentation_prob
        self.noise_level = noise_level

    def __len__(self) -> int:
        return len(self.dfg_series)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        dfg = self.dfg_series[idx]
        view1, view2 = DataAugmentation.generate_positive_pair(
            dfg, self.augmentation_prob, self.noise_level
        )
        return torch.as_tensor(view1, dtype=torch.float32), torch.as_tensor(view2, dtype=torch.float32)


class UnifiedDataset(Dataset):
    """按时间滑窗构造多步监督样本，不进行任何循环填充。"""

    def __init__(self, dfg_series: np.ndarray, window_size: int = 10,
                 pred_len: int = 5, is_validation: bool = False,
                 use_contrastive: bool = True, use_log_transform: bool = True):
        series = np.asarray(dfg_series, dtype=np.float32)
        self.dfg_series = np.log1p(series) if use_log_transform else series
        self.window_size = window_size
        self.pred_len = pred_len
        self.use_contrastive = use_contrastive
        self.samples = self._create_samples()
        self.has_samples = bool(self.samples)

    def _create_samples(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(self.dfg_series) - self.window_size - self.pred_len + 1
        if n_samples <= 0:
            return []
        samples = []
        for i in range(n_samples):
            src = self.dfg_series[i:i + self.window_size]
            tgt = self.dfg_series[i + self.window_size:i + self.window_size + self.pred_len]
            if not self.use_contrastive:
                src = src.reshape(self.window_size, -1)
                tgt = tgt.reshape(self.pred_len, -1)
            samples.append((src.astype(np.float32), tgt.astype(np.float32)))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src, tgt = self.samples[idx]
        return torch.from_numpy(src), torch.from_numpy(tgt)


#损失函数模块
class DynamicWeightedMSELoss(nn.Module):
    """在log1p频次空间使用加权Smooth-L1，降低极高频边对训练的支配。"""
    def __init__(self, config: DynamicConfig):
        super().__init__()
        self.threshold = math.log1p(config.high_freq_threshold) if config.use_log_transform else config.high_freq_threshold
        self.weight = config.high_freq_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raw = F.smooth_l1_loss(pred, target, reduction="none")
        weights = torch.ones_like(target)
        weights[target > self.threshold] = self.weight
        return (raw * weights).mean()


#训练和预测模块
def train_contrastive_encoder(encoder: ContrastiveDFGEncoder, train_loader: TorchDataLoader,
                              params: Dict[str, Any], device: str = 'cuda') -> ContrastiveDFGEncoder:
    if len(train_loader) == 0:
        raise ValueError("对比学习DataLoader为空")
    encoder.to(device)
    criterion = StableNTXentLoss(temperature=params["temperature"])
    optimizer = optim.Adam(encoder.parameters(), lr=params["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params["epochs"], eta_min=1e-6
    )
    print(f"开始对比学习预训练 (epochs={params['epochs']})...")
    for epoch in range(params["epochs"]):
        encoder.train()
        total = 0.0
        count = 0
        for view1, view2 in train_loader:
            view1, view2 = view1.to(device), view2.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(encoder(view1, True), encoder(view2, True))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            total += loss.item(); count += 1
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == params["epochs"]:
            print(f"Contrastive Epoch {epoch + 1}/{params['epochs']}, Loss: {total / max(count, 1):.6f}")
    return encoder


def _rollout_loss(model: nn.Module, src: torch.Tensor, tgt: torch.Tensor,
                  criterion: nn.Module, teacher_forcing_ratio: float,
                  config: DynamicConfig) -> torch.Tensor:
    """在训练阶段递归若干步，缓解25步测试时的暴露偏差。"""
    window = src
    losses = []
    for step in range(tgt.shape[1]):
        output, _ = model(window, hidden=None)
        pred = output[:, -1:, :]
        target = tgt[:, step:step + 1]
        target_flat = target.reshape(target.shape[0], 1, -1)
        losses.append(criterion(pred, target_flat))
        use_teacher = torch.rand((), device=src.device).item() < teacher_forcing_ratio
        next_value = target if use_teacher else pred.detach()
        if config.use_contrastive:
            if next_value.ndim == 3:
                n = int(math.sqrt(next_value.shape[-1]))
                next_value = next_value.reshape(next_value.shape[0], 1, n, n)
            window = torch.cat([window[:, 1:], next_value], dim=1)
        else:
            next_flat = next_value.reshape(next_value.shape[0], 1, -1)
            window = torch.cat([window[:, 1:], next_flat], dim=1)
    return torch.stack(losses).mean()


def _fit_epochs(model: nn.Module, loader: TorchDataLoader, config: DynamicConfig,
                device, epochs: int, validation_loader: Optional[TorchDataLoader] = None,
                allow_early_stop: bool = False):
    criterion = DynamicWeightedMSELoss(config)
    if hasattr(model, 'contrastive_encoder') and not config.finetune_contrastive:
        for p in model.contrastive_encoder.parameters():
            p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=config.lr, weight_decay=config.l2_lambda)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=max(2, config.patience // 2), factor=0.7, min_lr=1e-6
    )
    train_losses, val_losses = [], []
    best_val = float('inf'); best_epoch = epochs; counter = 0
    best_state = None

    for epoch in tqdm(range(epochs), desc="训练", leave=False):
        model.train()
        if hasattr(model, 'contrastive_encoder') and not config.finetune_contrastive:
            model.contrastive_encoder.eval()
        ratio = config.teacher_forcing_start + (config.teacher_forcing_end - config.teacher_forcing_start) * (epoch / max(epochs - 1, 1))
        total = 0.0; batches = 0
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _rollout_loss(model, src, tgt, criterion, ratio, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
            optimizer.step()
            total += loss.item(); batches += 1
        train_loss = total / max(batches, 1)
        train_losses.append(train_loss)

        if validation_loader is not None:
            model.eval(); total_v = 0.0; count_v = 0
            with torch.no_grad():
                for src, tgt in validation_loader:
                    src, tgt = src.to(device), tgt.to(device)
                    total_v += _rollout_loss(model, src, tgt, criterion, 1.0, config).item()
                    count_v += 1
            val_loss = total_v / max(count_v, 1)
            val_losses.append(val_loss)
            scheduler.step(val_loss)
            if val_loss < best_val - 1e-8:
                best_val = val_loss; best_epoch = epoch + 1; counter = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                counter += 1
                if allow_early_stop and counter >= config.patience:
                    print(f"早停触发 (Epoch {epoch + 1})，最佳Epoch={best_epoch}")
                    break
        else:
            val_losses.append(float('nan'))

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses, best_epoch


def train_model(model: nn.Module, train_loader: TorchDataLoader, val_loader: TorchDataLoader,
                config: DynamicConfig, device: str = 'cuda') -> Tuple[nn.Module, List[float], List[float]]:
    """先内部时间验证确定epoch，再从初始权重用完整训练区间重训。"""
    model.to(device)
    initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model, train_losses, val_losses, best_epoch = _fit_epochs(
        model, train_loader, config, device, config.epochs,
        validation_loader=val_loader, allow_early_stop=True
    )

    full_train_data = getattr(config, "_full_train_data", None)
    if full_train_data is not None:
        model.load_state_dict(initial_state)
        full_dataset = UnifiedDataset(
            full_train_data, config.window_size, config.train_horizon,
            use_contrastive=config.use_contrastive,
            use_log_transform=config.use_log_transform,
        )
        if len(full_dataset) == 0:
            raise ValueError("完整训练区间不足以构造训练样本")
        full_loader = TorchDataLoader(
            full_dataset, batch_size=config.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(config.seed)
        )
        print(f"使用完整前{config.train_size}步重新训练，epoch={best_epoch}")
        model, _, _, _ = _fit_epochs(
            model, full_loader, config, device, best_epoch,
            validation_loader=None, allow_early_stop=False
        )

    best_model_path = os.path.join(config.result_dir, "best_model.pth")
    torch.save(model.state_dict(), best_model_path)
    with open(os.path.join(config.result_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(config.get_model_params(), f, indent=4, ensure_ascii=False)
    return model, train_losses, val_losses


def predict(model: nn.Module, src_seq: torch.Tensor, pred_len: int,
            num_activities: int, config: DynamicConfig, device: str = 'cuda') -> np.ndarray:
    """论文式滚动窗口递归：每轮读取当前完整窗口，不复用旧hidden。"""
    model.eval()
    src = src_seq.to(device).float()
    if config.use_log_transform:
        src = torch.log1p(torch.clamp(src, min=0.0))
    pred_series = []
    with torch.no_grad():
        for _ in range(pred_len):
            output, _ = model(src, hidden=None)
            pred_log = output[:, -1:, :]
            pred_flat = torch.expm1(pred_log) if config.use_log_transform else pred_log
            pred_flat = torch.clamp(pred_flat, min=0.0)
            pred_dfg = pred_flat.reshape(1, num_activities, num_activities)
            pred_series.append(pred_dfg.squeeze(0).cpu().numpy())

            next_model_value = pred_log
            if config.use_contrastive:
                next_matrix = next_model_value.reshape(1, 1, num_activities, num_activities)
                src = torch.cat([src[:, 1:], next_matrix], dim=1)
            else:
                src = torch.cat([src[:, 1:], next_model_value], dim=1)
    return np.asarray(pred_series, dtype=np.float32)


#评估指标模块
def calculate_entropic_relevance(pred_dfg: np.ndarray, real_dfg: np.ndarray) -> float:
    eps = 1e-8
    pred = np.clip(np.asarray(pred_dfg, dtype=np.float64), 0, None)
    real = np.clip(np.asarray(real_dfg, dtype=np.float64), 0, None)
    pred_sum, real_sum = pred.sum(), real.sum()
    if real_sum <= eps:
        return 0.0 if pred_sum <= eps else float(-math.log(eps))
    if pred_sum <= eps:
        return float(-math.log(eps))
    pred_dist = pred / pred_sum
    real_dist = real / real_sum
    return float(-np.sum(real_dist * np.log(np.clip(pred_dist, eps, None))))


def calculate_mape(pred_series: np.ndarray, real_series: np.ndarray) -> Dict[str, float]:
    mape_list = []
    for pred_dfg, real_dfg in zip(pred_series, real_series):
        er_pred = calculate_entropic_relevance(pred_dfg, real_dfg)
        er_real = calculate_entropic_relevance(real_dfg, real_dfg)
        if er_real < 1e-8:
            mape = 0.0
        else:
            mape = np.abs((er_pred - er_real) / er_real) * 100
        mape_list.append(mape)
    mape_array = np.array(mape_list)
    return {
        "mean": float(np.mean(mape_array)),
        "min": float(np.min(mape_array)),
        "max": float(np.max(mape_array)),
        "std": float(np.std(mape_array))
    }


def calculate_jmape_components(pred_dfg: np.ndarray, real_dfg: np.ndarray,
                               alpha: float = 0.5, edge_threshold: float = 0.5,
                               epsilon: float = 1.0) -> Dict[str, float]:
    pred = np.clip(np.asarray(pred_dfg, dtype=np.float64), 0, None)
    real = np.clip(np.asarray(real_dfg, dtype=np.float64), 0, None)
    real_mask = real > 0
    pred_mask = pred >= edge_threshold
    if real_mask.any():
        frequency_mape = float(np.mean(
            np.abs(pred[real_mask] - real[real_mask]) /
            np.maximum(real[real_mask], epsilon)
        ))
    else:
        frequency_mape = 0.0 if not pred_mask.any() else 1.0
    intersection = np.logical_and(real_mask, pred_mask).sum()
    union = np.logical_or(real_mask, pred_mask).sum()
    jaccard = float(intersection / union) if union else 1.0
    jmape = alpha * frequency_mape + (1.0 - alpha) * (1.0 - jaccard)
    return {"jmape": float(jmape), "frequency_mape": frequency_mape,
            "jaccard": jaccard, "structure_error": float(1.0 - jaccard)}


def calculate_jmape(pred_dfg: np.ndarray, real_dfg: np.ndarray,
                    alpha: float = 0.5, edge_threshold: float = 0.5,
                    epsilon: float = 1.0) -> float:
    return calculate_jmape_components(
        pred_dfg, real_dfg, alpha, edge_threshold, epsilon
    )["jmape"]


class EnhancedMetrics:
    """评估指标类"""

    @staticmethod
    def edge_level_metrics(pred_dfg: np.ndarray, real_dfg: np.ndarray, threshold: float = 0.01) -> Dict[str, float]:
        pred_edges = (pred_dfg > threshold).flatten()
        real_edges = (real_dfg > threshold).flatten()

        if np.sum(real_edges) == 0 and np.sum(pred_edges) == 0:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        if np.sum(real_edges) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        precision = precision_score(real_edges, pred_edges, zero_division=0)
        recall = recall_score(real_edges, pred_edges, zero_division=0)
        f1 = f1_score(real_edges, pred_edges, zero_division=0)

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }

    @staticmethod
    def cosine_similarity(pred_dfg: np.ndarray, real_dfg: np.ndarray) -> float:
        pred_flat = pred_dfg.flatten()
        real_flat = real_dfg.flatten()

        if np.linalg.norm(pred_flat) == 0 or np.linalg.norm(real_flat) == 0:
            return 0.0

        cos_sim = np.dot(pred_flat, real_flat) / (np.linalg.norm(pred_flat) * np.linalg.norm(real_flat))
        return float(cos_sim)

    @staticmethod
    def pearson_correlation(pred_dfg: np.ndarray, real_dfg: np.ndarray) -> float:
        pred_flat = pred_dfg.flatten()
        real_flat = real_dfg.flatten()

        if np.std(pred_flat) == 0 or np.std(real_flat) == 0:
            return 0.0

        corr, _ = pearsonr(pred_flat, real_flat)
        return float(corr if not np.isnan(corr) else 0.0)

    @staticmethod
    def top_k_precision(pred_dfg: np.ndarray, real_dfg: np.ndarray, k: int = 5) -> float:
        n = pred_dfg.shape[0]
        total_edges = n * n
        k = min(k, total_edges)

        pred_flat = pred_dfg.flatten()
        top_k_indices = np.argsort(pred_flat)[-k:]
        real_flat = real_dfg.flatten()
        correct = np.sum(real_flat[top_k_indices] > 0)

        return float(correct / k)

    @staticmethod
    def compute_all_metrics(pred_series: List[np.ndarray], real_series: List[np.ndarray]) -> Dict[str, Any]:
        all_metrics = {
            "f1": [],
            "precision": [],
            "recall": [],
            "cosine": [],
            "pearson": [],
            "p@5": []
        }

        for pred, real in zip(pred_series, real_series):
            edge_metrics = EnhancedMetrics.edge_level_metrics(pred, real)
            all_metrics["f1"].append(edge_metrics["f1"])
            all_metrics["precision"].append(edge_metrics["precision"])
            all_metrics["recall"].append(edge_metrics["recall"])
            all_metrics["cosine"].append(EnhancedMetrics.cosine_similarity(pred, real))
            all_metrics["pearson"].append(EnhancedMetrics.pearson_correlation(pred, real))
            all_metrics["p@5"].append(EnhancedMetrics.top_k_precision(pred, real, k=5))

        summary = {}
        for metric_name, values in all_metrics.items():
            values_arr = np.array(values)
            summary[metric_name] = {
                "mean": float(np.mean(values_arr)),
                "std": float(np.std(values_arr)),
                "min": float(np.min(values_arr)),
                "max": float(np.max(values_arr)),
                "list": [float(v) for v in values_arr]
            }

        return summary


#可视化模块
def plot_er_results(er_pred: List[float], er_real: List[float], dataset_name: str,
                    aggregation_type: str, train_size: int, save_dir: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    x = range(1, len(er_pred) + 1)
    ax1.plot(x, er_real, 'b-o', label='Actual ER', linewidth=2, markersize=4)
    ax1.plot(x, er_pred, 'r--x', label='Predicted ER', linewidth=2, markersize=4)
    ax1.set_xlabel('Test Data Point Index', fontsize=12)
    ax1.set_ylabel('Entropic Relevance', fontsize=12, color='blue')
    ax1.set_title(f'{dataset_name} ({aggregation_type}, Train={train_size}) - ER Prediction', fontsize=14)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='y', labelcolor='blue')
    abs_error = [abs(p - r) for p, r in zip(er_pred, er_real)]
    ax2.bar(x, abs_error, color='green', alpha=0.6, width=0.8)
    ax2.set_xlabel('Test Data Point Index', fontsize=12)
    ax2.set_ylabel('Absolute Error', fontsize=12, color='green')
    ax2.set_title('Prediction Absolute Error', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='y', labelcolor='green')
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"ER_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ER结果图已保存: {save_path}")


def plot_newscore_results(newscore_list: List[float], dataset_name: str,
                          aggregation_type: str, train_size: int, save_dir: str):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    x = range(1, len(newscore_list) + 1)
    mean_ns = np.mean(newscore_list)
    ax.plot(x, newscore_list, 'darkgreen', marker='s', linewidth=2.5, markersize=7, label=f'J-MAPE (α=0.5)')
    ax.axhline(y=mean_ns, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_ns:.4f}')
    ax.set_xlabel('Test Data Point Index', fontsize=12)
    ax.set_ylabel('J-MAPE (Lower is Better)', fontsize=12)
    ax.set_title(f'{dataset_name} ({aggregation_type}, Train={train_size}) - J-MAPE', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    save_path = os.path.join(save_dir, f"JMAPE_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"J-MAPE结果图已保存: {save_path}")


def generate_ablation_figure(results_data: Dict, dataset_name: str,
                             aggregation_type: str, save_dir: str):
    train_sizes = [25, 50, 75]
    colors = ['#4C72B0', '#55A868', '#DD8452']
    labels = ['LSTM', 'LSTM+Attention', 'LSTM+Contrastive+Attention']
    model_keys = ['lstm_only', 'lstm_attention', 'lstm_contrastive_attention']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    for idx, train_size in enumerate(train_sizes):
        if train_size not in results_data:
            continue

        ax = axes[idx]
        results = results_data[train_size]

        means = []
        stds = []
        valid_labels = []
        valid_colors = []

        for i, (key, label) in enumerate(zip(model_keys, labels)):
            if results.get(key) is not None:
                means.append(results[key]['mean'])
                stds.append(results[key]['std'])
                valid_labels.append(label)
                valid_colors.append(colors[i])

        if not means:
            continue

        n_bars = len(means)
        bar_width = 1.0 / n_bars
        x = np.arange(n_bars) - (n_bars - 1) / 2
        x = x * bar_width

        bars = ax.bar(x, means, color=valid_colors, edgecolor='black',
                      linewidth=1, width=bar_width, alpha=0.85)

        ax.set_ylabel('J-MAPE', fontsize=11, fontweight='bold')
        ax.set_title(f'Training Size = {train_size}', fontsize=12, fontweight='bold', pad=10)

        ax.set_xticks(x)
        ax.set_xticklabels(valid_labels, fontsize=8, rotation=10, ha='right')

        y_max = max(means) * 1.15
        ax.set_ylim([0, y_max])

        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)

        if len(means) >= 2:
            imp_attn = ((means[0] - means[1]) / means[0]) * 100 if means[0] > 0 else 0
            if abs(imp_attn) > 0.001:
                arrow = '↓' if imp_attn > 0 else '↑'
                ax.text(0.5, 0.95, f'{arrow} Attention: {abs(imp_attn):.1f}%',
                        transform=ax.transAxes, ha='center', va='top',
                        fontsize=9, color='#2E8B57' if imp_attn > 0 else '#CD5C5C')

        if len(means) >= 3:
            imp_contrastive = ((means[1] - means[2]) / means[1]) * 100 if means[1] > 0 else 0
            if abs(imp_contrastive) > 0.001:
                arrow = '↓' if imp_contrastive > 0 else '↑'
                ax.text(0.5, 0.88, f'{arrow} Contrastive: {abs(imp_contrastive):.1f}%',
                        transform=ax.transAxes, ha='center', va='top',
                        fontsize=9, color='#2E8B57' if imp_contrastive > 0 else '#CD5C5C')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(f'Ablation Study - {dataset_name} ({aggregation_type})',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    save_path = os.path.join(save_dir, f'ablation_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"消融实验结果图已保存: {save_path}")


class EnhancedMetricsVisualizer:
    """指标可视化 - 双模型对比"""

    @staticmethod
    def plot_model_comparison(results_dict: Dict, dataset_name: str, train_size: int, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        metrics_config = {
            "f1": {"title": "F1 Score", "color": "#1f77b4", "ylabel": "Score", "ylim": [0, 1]},
            "precision": {"title": "Precision", "color": "#ff7f0e", "ylabel": "Score", "ylim": [0, 1]},
            "recall": {"title": "Recall", "color": "#2ca02c", "ylabel": "Score", "ylim": [0, 1]},
            "cosine": {"title": "Cosine Similarity", "color": "#d62728", "ylabel": "Similarity", "ylim": [-1, 1]},
            "pearson": {"title": "Pearson Correlation", "color": "#9467bd", "ylabel": "Correlation", "ylim": [-1, 1]},
            "p@5": {"title": "Precision@5", "color": "#8c564b", "ylabel": "Precision", "ylim": [0, 1]}
        }

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        model_labels = {
            "attention": "LSTM + Attention",
            "contrastive": "LSTM + Contrast + Attention"
        }
        model_colors = {
            "attention": "#1f77b4",
            "contrastive": "#ff7f0e"
        }

        for idx, (metric_name, config) in enumerate(metrics_config.items()):
            ax = axes[idx]

            for model_name in ["attention", "contrastive"]:
                if model_name in results_dict and metric_name in results_dict[model_name]:
                    data = results_dict[model_name][metric_name]
                    values = data["list"]
                    mean_val = data["mean"]
                    x = range(1, len(values) + 1)

                    ax.plot(x, values, color=model_colors[model_name],
                            marker='o', linewidth=2, markersize=4,
                            label=f"{model_labels[model_name]} (mean={mean_val:.4f})")
                    ax.axhline(y=mean_val, color=model_colors[model_name],
                               linestyle='--', alpha=0.5)

            ax.set_xlabel('Test Point', fontsize=10)
            ax.set_ylabel(config["ylabel"], fontsize=10)
            ax.set_title(config["title"], fontsize=12, fontweight='bold')
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(config["ylim"])

        plt.suptitle(f'Model Comparison - {dataset_name} (Train Size={train_size})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'model_comparison_train{train_size}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"对比曲线图已保存: {os.path.join(save_dir, f'model_comparison_train{train_size}.png')}")

    @staticmethod
    def plot_bar_comparison(results_dict: Dict, dataset_name: str, train_size: int, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        metrics = ["f1", "precision", "recall", "cosine", "pearson", "p@5"]
        metric_labels = ["F1 Score", "Precision", "Recall", "Cosine", "Pearson", "P@5"]
        model_names = ["attention", "contrastive"]
        model_labels = ["LSTM+Attention", "LSTM+Contrast+Attention"]
        colors = ["#1f77b4", "#ff7f0e"]

        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))

        for idx, (model_name, label, color) in enumerate(zip(model_names, model_labels, colors)):
            means = []
            stds = []
            for metric in metrics:
                if model_name in results_dict and metric in results_dict[model_name]:
                    means.append(results_dict[model_name][metric]["mean"])
                    stds.append(results_dict[model_name][metric]["std"])
                else:
                    means.append(0)
                    stds.append(0)

            offset = (idx - 0.5) * width
            bars = ax.bar(x + offset, means, width, label=label, color=color,
                          edgecolor='black', alpha=0.8, yerr=stds, capsize=3)

            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Model Comparison - {dataset_name} (Train Size={train_size})',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, rotation=15, ha='right')
        ax.legend(loc='upper right')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')

        if "attention" in results_dict and "contrastive" in results_dict:
            improvements = []
            for metric in metrics:
                attn_val = results_dict["attention"][metric]["mean"]
                cont_val = results_dict["contrastive"][metric]["mean"]
                if attn_val > 0:
                    imp = (cont_val - attn_val) / attn_val * 100
                    improvements.append(imp)

            if improvements:
                avg_imp = np.mean(improvements)
                arrow = "↑" if avg_imp > 0 else "↓"
                ax.text(0.5, 0.95, f'{arrow} Average Improvement: {abs(avg_imp):.1f}%',
                        transform=ax.transAxes, ha='center', va='top',
                        fontsize=11, fontweight='bold',
                        color='green' if avg_imp > 0 else 'red')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'bar_comparison_train{train_size}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"柱状对比图已保存: {os.path.join(save_dir, f'bar_comparison_train{train_size}.png')}")

    @staticmethod
    def generate_comparison_report(results_dict: Dict, dataset_name: str, train_size: int, save_dir: str):
        report = {
            "dataset": dataset_name,
            "train_size": train_size,
            "models": {}
        }

        for model_name in ["attention", "contrastive"]:
            if model_name in results_dict:
                report["models"][model_name] = {}
                for metric_name, data in results_dict[model_name].items():
                    report["models"][model_name][metric_name] = {
                        "mean": data["mean"],
                        "std": data["std"],
                        "min": data["min"],
                        "max": data["max"]
                    }

        if "attention" in results_dict and "contrastive" in results_dict:
            report["improvements"] = {}
            for metric_name in results_dict["attention"].keys():
                attn_val = results_dict["attention"][metric_name]["mean"]
                cont_val = results_dict["contrastive"][metric_name]["mean"]
                if attn_val > 0:
                    improvement = (cont_val - attn_val) / attn_val * 100
                    report["improvements"][metric_name] = round(improvement, 2)

        with open(os.path.join(save_dir, f'comparison_report_train{train_size}.json'), 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

        print("\n" + "=" * 70)
        print(f"模型对比报告 - {dataset_name} (Train Size={train_size})")
        print("=" * 70)
        print(f"\n{'指标':<15} {'LSTM+Attention':<20} {'LSTM+Contrast+Attention':<25} {'改进':<10}")
        print("-" * 70)

        metric_names = {
            "f1": "F1 Score",
            "precision": "Precision",
            "recall": "Recall",
            "cosine": "Cosine Similarity",
            "pearson": "Pearson Correlation",
            "p@5": "Precision@5"
        }

        for metric_key, metric_display in metric_names.items():
            if metric_key in results_dict["attention"] and metric_key in results_dict["contrastive"]:
                attn_val = results_dict["attention"][metric_key]["mean"]
                cont_val = results_dict["contrastive"][metric_key]["mean"]
                improvement = ((cont_val - attn_val) / attn_val * 100) if attn_val > 0 else 0
                arrow = "↑" if improvement > 0 else "↓" if improvement < 0 else "→"
                print(f"{metric_display:<15} {attn_val:<20.4f} {cont_val:<25.4f} {arrow} {abs(improvement):.1f}%")

        print("=" * 70)
        return report


#工具函数
def prepare_data(dataset_name: str, train_size: int, aggregation_type: str,
                 base_intervals: int = 100, test_intervals: int = 25):
    data_loader = ProcessDataLoader(dataset_name)
    sap_df = data_loader.extract_direct_follows()

    agg_func = DFAggregator.equisize_aggregation if aggregation_type == "equisize" else DFAggregator.equitemporal_aggregation
    dfg_series = agg_func(sap_df, base_intervals, data_loader.num_activities)

    # 严格遵循实验设计：
    # Train-25:  0:25  -> 25:50
    # Train-50:  0:50  -> 50:75
    # Train-75:  0:75  -> 75:100
    test_start = train_size
    test_end = train_size + test_intervals

    if base_intervals != len(dfg_series):
        raise ValueError(
            f"DFG时间步数量异常：期望 {base_intervals}，实际 {len(dfg_series)}"
        )
    if test_end > len(dfg_series):
        raise ValueError(
            f"数据长度不足：训练截止点={train_size}，测试长度={test_intervals}，"
            f"总时间步={len(dfg_series)}"
        )

    train_data = dfg_series[:train_size]
    test_data = dfg_series[test_start:test_end]

    if len(test_data) != test_intervals:
        raise RuntimeError(
            f"测试集切分错误：期望 {test_intervals} 步，实际 {len(test_data)} 步"
        )

    return data_loader, dfg_series, train_data, test_data


def create_dataloaders(train_data, val_data, config, use_contrastive: bool):
    """严格时间切分：fit区间不含验证目标，val保留前置窗口上下文。"""
    del val_data  # 为兼容旧调用保留参数
    train_data = np.asarray(train_data, dtype=np.float32)
    min_fit = config.window_size + config.train_horizon
    max_val = len(train_data) - min_fit
    if max_val < 1:
        return None, None
    val_steps = min(config.val_steps, max_val)
    split_idx = len(train_data) - val_steps
    fit_data = train_data[:split_idx]
    val_context_start = max(0, split_idx - config.window_size)
    val_context = train_data[val_context_start:]

    train_dataset = UnifiedDataset(
        fit_data, config.window_size, config.train_horizon,
        use_contrastive=use_contrastive,
        use_log_transform=config.use_log_transform,
    )
    val_dataset = UnifiedDataset(
        val_context, config.window_size, config.train_horizon,
        use_contrastive=use_contrastive,
        use_log_transform=config.use_log_transform,
    )
    if not train_dataset.has_samples or not val_dataset.has_samples:
        return None, None
    config._full_train_data = train_data
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = TorchDataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, generator=generator
    )
    val_loader = TorchDataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )
    return train_loader, val_loader


def get_val_data(train_data, config):
    """兼容旧接口；真正的时间切分由create_dataloaders完成。"""
    return np.asarray(train_data, dtype=np.float32)


def train_contrastive_encoder_if_needed(train_data, num_activities, config, device, pretrained_path: str):
    if os.path.exists(pretrained_path):
        encoder = ContrastiveDFGEncoder(
            num_activities, config.contrastive_dim, config.contrastive_proj_dim
        )
        try:
            encoder.load_state_dict(torch.load(pretrained_path, map_location=device))
            print("加载当前数据集/聚合/训练截点对应的预训练编码器")
            return encoder
        except RuntimeError:
            print("预训练编码器结构不匹配，将重新训练")

    dataset = ContrastiveDataset(
        train_data, config.augmentation_prob, config.noise_level,
        use_log_transform=config.use_log_transform,
    )
    if len(dataset) < 2:
        raise ValueError("对比学习至少需要2个DFG时间步")
    batch_size = max(2, min(config.batch_size, len(dataset)))
    loader = TorchDataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        generator=torch.Generator().manual_seed(config.seed)
    )
    encoder = ContrastiveDFGEncoder(
        num_activities, config.contrastive_dim, config.contrastive_proj_dim
    )
    params = {"temperature": config.contrastive_temperature,
              "epochs": config.contrastive_epochs, "lr": config.contrastive_lr}
    encoder = train_contrastive_encoder(encoder, loader, params, device)
    torch.save(encoder.state_dict(), pretrained_path)
    return encoder


def run_er_evaluation(dataset_name: str, train_size: int, aggregation_type: str,
                      use_attention: bool = False) -> Optional[Dict[str, Any]]:
    print(f"\n{'=' * 80}")
    print(f"ER评估实验: {dataset_name} | 聚合: {aggregation_type} | 训练集: {train_size}")
    print(f"模型: {'Attention-LSTM' if use_attention else 'LSTM'}")
    print(f"{'=' * 80}")

    try:
        config = DynamicConfig(train_size, use_contrastive=False, use_attention=use_attention, dataset_name=dataset_name, aggregation_type=aggregation_type)
        data_loader, dfg_series, train_data, test_data = prepare_data(
            dataset_name, train_size, aggregation_type, config.base_intervals, config.test_intervals
        )

        print(f"训练数据形状: {train_data.shape}, 测试数据形状: {test_data.shape}")

        val_data = get_val_data(train_data, config)
        train_loader, val_loader = create_dataloaders(train_data, val_data, config, use_contrastive=False)

        if train_loader is None:
            print(f"训练样本不足，跳过")
            return None

        model = UnifiedDFGPredictor(num_activities=data_loader.num_activities, config=config)
        model, train_losses, val_losses = train_model(model, train_loader, val_loader, config, DEVICE)

        if config.use_contrastive:
            test_src = train_data[-config.window_size:]
            test_src = torch.FloatTensor(test_src).unsqueeze(0)
        else:
            test_src = train_data[-config.window_size:].reshape(1, config.window_size, -1)
            test_src = torch.FloatTensor(test_src)

        pred_series = predict(model, test_src, len(test_data), data_loader.num_activities, config, DEVICE)

        mape_stats = calculate_mape(pred_series, test_data)
        er_pred = [calculate_entropic_relevance(p, r) for p, r in zip(pred_series, test_data)]
        er_real = [calculate_entropic_relevance(r, r) for r in test_data]

        print(f"MAPE: 均值={mape_stats['mean']:.2f}%, 最小值={mape_stats['min']:.2f}%, 最大值={mape_stats['max']:.2f}%")
        print(f"ER: 预测均值={np.mean(er_pred):.4f}, 实际均值={np.mean(er_real):.4f}")

        plot_er_results(er_pred, er_real, dataset_name, aggregation_type, train_size, config.result_dir)

        result = {
            "experiment_type": "ER_Evaluation",
            "model_type": "Attention-LSTM" if use_attention else "LSTM",
            "dataset": dataset_name,
            "train_size": train_size,
            "aggregation_type": aggregation_type,
            "mape_stats": mape_stats,
            "er_pred": [float(x) for x in er_pred],
            "er_real": [float(x) for x in er_real],
            "train_losses": [float(x) for x in train_losses],
            "val_losses": [float(x) for x in val_losses]
        }

        with open(os.path.join(config.result_dir, "er_evaluation_results.json"), 'w') as f:
            json.dump(result, f, indent=4)

        return result

    except Exception as e:
        print(f"ER评估实验失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_jmape_evaluation(dataset_name: str, train_size: int, aggregation_type: str) -> Optional[Dict[str, Any]]:
    print(f"\n{'=' * 80}")
    print(f"J-MAPE评估实验: {dataset_name} | 聚合: {aggregation_type} | 训练集: {train_size}")
    print(f"模型: Contrastive-Attention-LSTM")
    print(f"{'=' * 80}")

    try:
        config = DynamicConfig(train_size, use_contrastive=True, use_attention=True, dataset_name=dataset_name, aggregation_type=aggregation_type)
        data_loader, dfg_series, train_data, test_data = prepare_data(
            dataset_name, train_size, aggregation_type, config.base_intervals, config.test_intervals
        )

        print(f"训练数据形状: {train_data.shape}, 测试数据形状: {test_data.shape}")

        pretrained_path = os.path.join(config.result_dir, f"pretrained_encoder.pth")
        contrastive_encoder = train_contrastive_encoder_if_needed(
            train_data, data_loader.num_activities, config, DEVICE, pretrained_path
        )

        val_data = get_val_data(train_data, config)
        train_loader, val_loader = create_dataloaders(train_data, val_data, config, use_contrastive=True)

        if train_loader is None:
            print(f"训练样本不足，跳过")
            return None

        model = UnifiedDFGPredictor(data_loader.num_activities, config, contrastive_encoder)
        model, train_losses, val_losses = train_model(model, train_loader, val_loader, config, DEVICE)

        test_src = train_data[-config.window_size:]
        test_src = torch.FloatTensor(test_src).unsqueeze(0)
        pred_series = predict(model, test_src, len(test_data), data_loader.num_activities, config, DEVICE)

        jmape_list = [calculate_jmape(p, r, config.alpha, config.edge_threshold, config.jmape_epsilon) for p, r in zip(pred_series, test_data)]
        jmape_stats = {
            "mean": float(np.mean(jmape_list)),
            "min": float(np.min(jmape_list)),
            "max": float(np.max(jmape_list)),
            "std": float(np.std(jmape_list)),
            "list": [float(x) for x in jmape_list]
        }

        print(f"J-MAPE统计: 均值={jmape_stats['mean']:.4f}, 最小值={jmape_stats['min']:.4f}, "
              f"最大值={jmape_stats['max']:.4f}, 标准差={jmape_stats['std']:.4f}")

        plot_newscore_results(jmape_list, dataset_name, aggregation_type, train_size, config.result_dir)

        result = {
            "experiment_type": "JMAPE_Evaluation",
            "model_type": "Contrastive-Attention-LSTM",
            "dataset": dataset_name,
            "train_size": train_size,
            "aggregation_type": aggregation_type,
            "jmape_stats": jmape_stats,
            "train_losses": [float(x) for x in train_losses],
            "val_losses": [float(x) for x in val_losses]
        }

        with open(os.path.join(config.result_dir, "jmape_evaluation_results.json"), 'w') as f:
            json.dump(result, f, indent=4)

        return result

    except Exception as e:
        print(f"J-MAPE评估实验失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_ablation_experiment(dataset_name: str, aggregation_type: str = "equisize") -> Dict:
    print("\n" + "=" * 80)
    print(f"消融实验: {dataset_name}")
    print(f"聚合方式: {aggregation_type}")
    print(f"对比: 纯LSTM vs LSTM+注意力 vs LSTM+对比+注意力")
    print("=" * 80)

    train_sizes = [25, 50, 75]
    main_ablation_dir = f"./ablation_results/{dataset_name}_{aggregation_type}"
    os.makedirs(main_ablation_dir, exist_ok=True)

    all_results = {}

    for train_size in train_sizes:
        print(f"\n{'=' * 50}")
        print(f"训练集大小 = {train_size}")
        print(f"{'=' * 50}")

        results = {"lstm_only": None, "lstm_attention": None, "lstm_contrastive_attention": None}

        try:
            data_loader, dfg_series, train_data, test_data = prepare_data(
                dataset_name, train_size, aggregation_type, 100, 25
            )

            val_data = train_data[-min(30, len(train_data)):] if len(train_data) > 30 else train_data

            # 实验1: 纯LSTM
            print(f"\n@ 纯LSTM模型")
            config_lstm = DynamicConfig(train_size, use_contrastive=False, use_attention=False, dataset_name=dataset_name, aggregation_type=aggregation_type)
            config_lstm.result_dir = os.path.join(main_ablation_dir, f"train_{train_size}", "lstm_only")
            os.makedirs(config_lstm.result_dir, exist_ok=True)

            train_loader, val_loader = create_dataloaders(train_data, val_data, config_lstm, use_contrastive=False)
            if train_loader is None:
                print(f"训练样本不足，跳过")
                continue

            model = UnifiedDFGPredictor(data_loader.num_activities, config_lstm)
            model, _, _ = train_model(model, train_loader, val_loader, config_lstm, DEVICE)

            test_src = train_data[-config_lstm.window_size:].reshape(1, config_lstm.window_size, -1)
            test_src = torch.FloatTensor(test_src)
            pred_series = predict(model, test_src, len(test_data), data_loader.num_activities, config_lstm, DEVICE)

            jmape_list = [calculate_jmape(p, r, 0.5, config_lstm.edge_threshold, config_lstm.jmape_epsilon) for p, r in zip(pred_series, test_data)]
            results["lstm_only"] = {
                "mean": float(np.mean(jmape_list)),
                "std": float(np.std(jmape_list)),
                "list": [float(x) for x in jmape_list]
            }
            print(f"  纯LSTM J-MAPE: {results['lstm_only']['mean']:.6f}")

            # 实验2: LSTM+注意力
            print(f"\n￥ LSTM+注意力模型")
            config_attn = DynamicConfig(train_size, use_contrastive=False, use_attention=True, dataset_name=dataset_name, aggregation_type=aggregation_type)
            config_attn.result_dir = os.path.join(main_ablation_dir, f"train_{train_size}", "lstm_attention")
            os.makedirs(config_attn.result_dir, exist_ok=True)

            train_loader_attn, val_loader_attn = create_dataloaders(train_data, val_data, config_attn,
                                                                    use_contrastive=False)
            model_attn = UnifiedDFGPredictor(data_loader.num_activities, config_attn)
            model_attn, _, _ = train_model(model_attn, train_loader_attn, val_loader_attn, config_attn, DEVICE)

            pred_series_attn = predict(model_attn, test_src, len(test_data), data_loader.num_activities, config_attn,
                                       DEVICE)
            jmape_list_attn = [calculate_jmape(p, r, 0.5, config_attn.edge_threshold, config_attn.jmape_epsilon) for p, r in zip(pred_series_attn, test_data)]
            results["lstm_attention"] = {
                "mean": float(np.mean(jmape_list_attn)),
                "std": float(np.std(jmape_list_attn)),
                "list": [float(x) for x in jmape_list_attn]
            }
            print(f"  LSTM+注意力 J-MAPE: {results['lstm_attention']['mean']:.6f}")

            # 实验3: LSTM+对比+注意力
            print(f"\n& LSTM+对比+注意力模型")
            config_contrastive = DynamicConfig(train_size, use_contrastive=True, use_attention=True, dataset_name=dataset_name, aggregation_type=aggregation_type)
            config_contrastive.result_dir = os.path.join(main_ablation_dir, f"train_{train_size}", "lstm_contrastive")
            os.makedirs(config_contrastive.result_dir, exist_ok=True)

            pretrained_path = os.path.join(config_contrastive.result_dir, f"pretrained_encoder.pth")
            contrastive_encoder = train_contrastive_encoder_if_needed(
                train_data, data_loader.num_activities, config_contrastive, DEVICE, pretrained_path
            )

            train_loader_cl, val_loader_cl = create_dataloaders(train_data, val_data, config_contrastive,
                                                                use_contrastive=True)
            model_contrastive = UnifiedDFGPredictor(data_loader.num_activities, config_contrastive, contrastive_encoder)
            model_contrastive, _, _ = train_model(model_contrastive, train_loader_cl, val_loader_cl, config_contrastive,
                                                  DEVICE)

            test_src_cl = train_data[-config_contrastive.window_size:]
            test_src_cl = torch.FloatTensor(test_src_cl).unsqueeze(0)
            pred_series_cl = predict(model_contrastive, test_src_cl, len(test_data), data_loader.num_activities,
                                     config_contrastive, DEVICE)

            jmape_list_cl = [calculate_jmape(p, r, 0.5, config_contrastive.edge_threshold, config_contrastive.jmape_epsilon) for p, r in zip(pred_series_cl, test_data)]
            results["lstm_contrastive_attention"] = {
                "mean": float(np.mean(jmape_list_cl)),
                "std": float(np.std(jmape_list_cl)),
                "list": [float(x) for x in jmape_list_cl]
            }
            print(f"  LSTM+对比+注意力 J-MAPE: {results['lstm_contrastive_attention']['mean']:.6f}")

            all_results[train_size] = results

            with open(os.path.join(main_ablation_dir, f"train_{train_size}", "ablation_results.json"), 'w') as f:
                json.dump(results, f, indent=4)

        except Exception as e:
            print(f"训练集大小 {train_size} 实验失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    if all_results:
        pkl_save_path = os.path.join(main_ablation_dir, "ablation_results.pkl")
        with open(pkl_save_path, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"\n@完整结果已保存到: {pkl_save_path}")

        json_save_path = os.path.join(main_ablation_dir, "ablation_results.json")
        json_results = {}
        for ts, res in all_results.items():
            json_results[str(ts)] = {}
            for model, stats in res.items():
                if stats is not None:
                    json_results[str(ts)][model] = {
                        "mean": float(stats["mean"]),
                        "std": float(stats["std"]),
                        "list": [float(x) for x in stats["list"]]
                    }
                else:
                    json_results[str(ts)][model] = None
        with open(json_save_path, 'w') as f:
            json.dump(json_results, f, indent=4)
        print(f"@JSON格式结果已保存到: {json_save_path}")

    generate_ablation_figure(all_results, dataset_name, aggregation_type, main_ablation_dir)

    return all_results


def run_dual_model_evaluation(dataset_name: str, train_size: int, aggregation_type: str = "equisize"):
    print(f"\n{'=' * 80}")
    print(f"双模型对比评估: {dataset_name} | Train={train_size} | Agg={aggregation_type}")
    print(f"对比: LSTM+Attention vs LSTM+Contrast+Attention")
    print(f"{'=' * 80}")

    results = {}

    data_loader, dfg_series, train_data, test_data = prepare_data(
        dataset_name, train_size, aggregation_type, 100, 25
    )

    val_data = train_data[-min(30, len(train_data)):] if len(train_data) > 30 else train_data

    # 模型1: LSTM + Attention
    print("\n@ 训练 LSTM + Attention 模型...")
    try:
        config_attn = DynamicConfig(train_size, use_contrastive=False, use_attention=True, dataset_name=dataset_name, aggregation_type=aggregation_type)
        config_attn.result_dir = f"./dual_results/{dataset_name}_{aggregation_type}/train_{train_size}/attention"
        os.makedirs(config_attn.result_dir, exist_ok=True)

        train_loader_attn, val_loader_attn = create_dataloaders(train_data, val_data, config_attn,
                                                                use_contrastive=False)

        if train_loader_attn is not None:
            model_attn = UnifiedDFGPredictor(data_loader.num_activities, config_attn)
            model_attn, _, _ = train_model(model_attn, train_loader_attn, val_loader_attn, config_attn, DEVICE)

            test_src = train_data[-config_attn.window_size:].reshape(1, config_attn.window_size, -1)
            test_src = torch.FloatTensor(test_src)
            pred_series_attn = predict(model_attn, test_src, len(test_data), data_loader.num_activities, config_attn,
                                       DEVICE)

            results["attention"] = EnhancedMetrics.compute_all_metrics(pred_series_attn, test_data)
            print(f"  & Attention模型完成 - F1: {results['attention']['f1']['mean']:.4f}")
        else:
            print(f"  ！ Attention模型训练样本不足")
            results["attention"] = None

    except Exception as e:
        print(f"  ！ Attention模型失败: {e}")
        results["attention"] = None

    # 模型2: LSTM + Contrast + Attention
    print("\n@ 训练 LSTM + Contrast + Attention 模型...")
    try:
        config_contrastive = DynamicConfig(train_size, use_contrastive=True, use_attention=True, dataset_name=dataset_name, aggregation_type=aggregation_type)
        config_contrastive.result_dir = f"./dual_results/{dataset_name}_{aggregation_type}/train_{train_size}/contrastive"
        os.makedirs(config_contrastive.result_dir, exist_ok=True)

        pretrained_path = os.path.join(config_contrastive.result_dir, "pretrained_encoder.pth")
        contrastive_encoder = train_contrastive_encoder_if_needed(
            train_data, data_loader.num_activities, config_contrastive, DEVICE, pretrained_path
        )

        train_loader_cl, val_loader_cl = create_dataloaders(train_data, val_data, config_contrastive,
                                                            use_contrastive=True)

        if train_loader_cl is not None:
            model_contrastive = UnifiedDFGPredictor(data_loader.num_activities, config_contrastive, contrastive_encoder)
            model_contrastive, _, _ = train_model(model_contrastive, train_loader_cl, val_loader_cl, config_contrastive,
                                                  DEVICE)

            test_src_cl = train_data[-config_contrastive.window_size:]
            test_src_cl = torch.FloatTensor(test_src_cl).unsqueeze(0)
            pred_series_cl = predict(model_contrastive, test_src_cl, len(test_data), data_loader.num_activities,
                                     config_contrastive, DEVICE)

            results["contrastive"] = EnhancedMetrics.compute_all_metrics(pred_series_cl, test_data)
            print(f"  ✅ Contrastive模型完成 - F1: {results['contrastive']['f1']['mean']:.4f}")
        else:
            print(f"  ！ Contrastive模型训练样本不足")
            results["contrastive"] = None

    except Exception as e:
        print(f"  ！ Contrastive模型失败: {e}")
        results["contrastive"] = None

    if results.get("attention") is not None and results.get("contrastive") is not None:
        save_dir = f"./dual_results/{dataset_name}_{aggregation_type}/train_{train_size}"
        os.makedirs(save_dir, exist_ok=True)

        EnhancedMetricsVisualizer.plot_model_comparison(results, dataset_name, train_size, save_dir)
        EnhancedMetricsVisualizer.plot_bar_comparison(results, dataset_name, train_size, save_dir)
        EnhancedMetricsVisualizer.generate_comparison_report(results, dataset_name, train_size, save_dir)

        print(f"\n 所有结果已保存到: {save_dir}")
    else:
        print("\n！ 至少有一个模型失败，无法生成对比图")

    return results


def run_full_dual_evaluation(dataset_name: str, aggregation_type: str = "equisize"):
    print("\n" + "=" * 80)
    print("完整双模型对比评估")
    print(f"数据集: {dataset_name} | 聚合方式: {aggregation_type}")
    print("对比: LSTM+Attention vs LSTM+Contrast+Attention")
    print("训练集大小: 25, 50, 75")
    print("=" * 80)

    train_sizes = [25, 50, 75]
    all_results = {}

    for train_size in train_sizes:
        print(f"\n{'=' * 50}")
        print(f"训练集大小: {train_size}")
        print(f"{'=' * 50}")
        results = run_dual_model_evaluation(dataset_name, train_size, aggregation_type)
        all_results[train_size] = results

    summary_dir = f"./dual_results/{dataset_name}_{aggregation_type}"
    os.makedirs(summary_dir, exist_ok=True)

    generate_summary_comparison(all_results, dataset_name, aggregation_type, summary_dir)

    return all_results


def generate_summary_comparison(all_results: Dict, dataset_name: str, aggregation_type: str, save_dir: str):
    train_sizes = [25, 50, 75]
    metrics = ["f1", "precision", "recall", "cosine", "pearson", "p@5"]
    metric_labels = ["F1 Score", "Precision", "Recall", "Cosine", "Pearson", "P@5"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    model_colors = {"attention": "#1f77b4", "contrastive": "#ff7f0e"}
    model_labels = {"attention": "LSTM+Attention", "contrastive": "LSTM+Contrast+Attention"}

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]

        for model_name in ["attention", "contrastive"]:
            means = []
            stds = []

            for ts in train_sizes:
                if ts in all_results and all_results[ts].get(model_name) is not None:
                    metric_data = all_results[ts][model_name].get(metric)
                    if metric_data:
                        means.append(metric_data["mean"])
                        stds.append(metric_data["std"])
                    else:
                        means.append(0)
                        stds.append(0)
                else:
                    means.append(0)
                    stds.append(0)

            ax.errorbar(train_sizes, means, yerr=stds,
                        color=model_colors[model_name], marker='o', markersize=8,
                        linewidth=2, capsize=5, capthick=2,
                        label=model_labels[model_name])

        ax.set_xlabel('Training Size', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'{label} across Training Sizes', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(train_sizes)
        ax.set_ylim([0, 1.05])

    plt.suptitle(f'Dual Model Comparison - {dataset_name} ({aggregation_type})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'summary_comparison_across_sizes.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n汇总对比图已保存: {os.path.join(save_dir, 'summary_comparison_across_sizes.png')}")


def run_overall_evaluation(aggregation_type: str = "equisize"):
    print("\n" + "=" * 80)
    print("整体评估：多数据集汇总对比")
    print(f"聚合方式: {aggregation_type}")
    print("=" * 80)

    datasets = ["Helpdesk", "Sepsis", "BPIC2012", "HEL", "RTF", "BPIC2017", "BPIC2019", "BPIC2020"]
    train_sizes = [25, 50, 75]

    overall_results = {dataset: {} for dataset in datasets}

    for dataset_name in datasets:
        print(f"\n@ 处理数据集: {dataset_name}")
        if dataset_name not in DATA_PATHS:
            print(f" ！ {dataset_name} 数据路径不存在，跳过")
            continue

        try:
            all_results = {}
            for train_size in train_sizes:
                print(f"\n  训练集大小: {train_size}")
                results = run_dual_model_evaluation(dataset_name, train_size, aggregation_type)
                all_results[train_size] = results
            overall_results[dataset_name] = all_results
        except Exception as e:
            print(f"  ！ {dataset_name} 评估失败: {e}")
            overall_results[dataset_name] = {}

    generate_overall_summary(overall_results, aggregation_type)

    return overall_results


def generate_overall_summary(overall_results: Dict, aggregation_type: str):
    save_dir = f"./dual_results/Overall_{aggregation_type}"
    os.makedirs(save_dir, exist_ok=True)

    datasets = ["Helpdesk", "Sepsis", "BPIC2012", "HEL", "RTF", "BPIC2017", "BPIC2019", "BPIC2020"]
    train_sizes = [25, 50, 75]
    metrics = ["f1", "precision", "recall", "cosine", "pearson", "p@5"]
    metric_labels = ["F1 Score", "Precision", "Recall", "Cosine", "Pearson", "P@5"]
    model_names = ["attention", "contrastive"]
    model_labels = ["LSTM+Attention", "LSTM+Contrast+Attention"]
    model_colors = ["#1f77b4", "#ff7f0e"]

    # 图1: 每个指标下两个模型的平均性能对比
    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))
    axes1 = axes1.flatten()

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes1[idx]

        x = np.arange(len(datasets))
        width = 0.35

        for m_idx, (model_name, model_label, color) in enumerate(zip(model_names, model_labels, model_colors)):
            means = []
            for ds in datasets:
                ds_means = []
                if ds in overall_results and overall_results[ds]:
                    for ts in train_sizes:
                        if ts in overall_results[ds] and overall_results[ds][ts].get(model_name) is not None:
                            if metric in overall_results[ds][ts][model_name]:
                                ds_means.append(overall_results[ds][ts][model_name][metric]["mean"])
                means.append(np.mean(ds_means) if ds_means else 0)

            offset = (m_idx - 0.5) * width
            bars = ax.bar(x + offset, means, width, label=model_label, color=color, alpha=0.8)

            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7)

        ax.set_xlabel('Dataset', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'{label} - Overall Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Overall Model Comparison Across All Datasets ({aggregation_type})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'overall_comparison_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图2: 每个数据集的F1分数对比（热力图）
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

    for m_idx, (model_name, model_label, color) in enumerate(zip(model_names, model_labels, model_colors)):
        ax = axes2[m_idx]

        data_matrix = []
        for ds in datasets:
            row = []
            for ts in train_sizes:
                if ds in overall_results and overall_results[ds]:
                    if ts in overall_results[ds] and overall_results[ds][ts].get(model_name) is not None:
                        if "f1" in overall_results[ds][ts][model_name]:
                            row.append(overall_results[ds][ts][model_name]["f1"]["mean"])
                        else:
                            row.append(0)
                    else:
                        row.append(0)
                else:
                    row.append(0)
            data_matrix.append(row)

        im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

        ax.set_xticks(range(len(train_sizes)))
        ax.set_yticks(range(len(datasets)))
        ax.set_xticklabels([f'Size={ts}' for ts in train_sizes])
        ax.set_yticklabels(datasets)
        ax.set_title(f'{model_label} - F1 Score Heatmap', fontsize=12, fontweight='bold')

        for i in range(len(datasets)):
            for j in range(len(train_sizes)):
                ax.text(j, i, f'{data_matrix[i][j]:.3f}', ha='center', va='center', fontsize=8)

        plt.colorbar(im, ax=ax)

    plt.suptitle(f'F1 Score Comparison - All Datasets ({aggregation_type})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'overall_f1_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图3: 改进百分比汇总
    fig3, ax = plt.subplots(figsize=(12, 6))

    improvements = {}
    for ds in datasets:
        ds_improvements = []
        for metric in metrics:
            attn_vals = []
            cont_vals = []
            for ts in train_sizes:
                if ds in overall_results and overall_results[ds]:
                    if ts in overall_results[ds] and overall_results[ds][ts].get("attention") is not None:
                        if metric in overall_results[ds][ts]["attention"]:
                            attn_vals.append(overall_results[ds][ts]["attention"][metric]["mean"])
                    if ts in overall_results[ds] and overall_results[ds][ts].get("contrastive") is not None:
                        if metric in overall_results[ds][ts]["contrastive"]:
                            cont_vals.append(overall_results[ds][ts]["contrastive"][metric]["mean"])

            avg_attn = np.mean(attn_vals) if attn_vals else 0
            avg_cont = np.mean(cont_vals) if cont_vals else 0
            if avg_attn > 0:
                improvement = (avg_cont - avg_attn) / avg_attn * 100
                ds_improvements.append(improvement)

        improvements[ds] = np.mean(ds_improvements) if ds_improvements else 0

    bars = ax.bar(improvements.keys(), improvements.values(),
                  color=['green' if v > 0 else 'red' for v in improvements.values()],
                  edgecolor='black', alpha=0.8)

    for bar, val in zip(bars, improvements.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('Contrastive Learning Improvement over Attention (Average over Train Sizes)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'overall_improvement_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 保存汇总JSON
    summary_json = {}
    for ds in datasets:
        summary_json[ds] = {}
        for ts in train_sizes:
            summary_json[ds][str(ts)] = {}
            for model_name in model_names:
                if ds in overall_results and overall_results[ds]:
                    if ts in overall_results[ds] and overall_results[ds][ts].get(model_name) is not None:
                        summary_json[ds][str(ts)][model_name] = {}
                        for metric in metrics:
                            if metric in overall_results[ds][ts][model_name]:
                                summary_json[ds][str(ts)][model_name][metric] = {
                                    "mean": overall_results[ds][ts][model_name][metric]["mean"],
                                    "std": overall_results[ds][ts][model_name][metric]["std"]
                                }

    with open(os.path.join(save_dir, 'overall_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, indent=4, ensure_ascii=False)

    print(f"\n 整体评估完成！结果保存至: {save_dir}")
    print(f"   - overall_comparison_bar.png: 多数据集柱状对比图")
    print(f"   - overall_f1_heatmap.png: F1分数热力图")
    print(f"   - overall_improvement_summary.png: 改进百分比汇总")
    print(f"   - overall_summary.json: 汇总数据")


def module5_main():
    """模块5: 评估入口"""
    print("\n" + "=" * 80)
    print("模块5: 增强评估指标体系")
    print("=" * 80)
    print("对比模型:")
    print("  LSTM + Attention")
    print("  LSTM + Contrast + Attention")
    print("\n保留指标:")
    print("  ①F1 Score | Precision | Recall")
    print("  ②Cosine Similarity | Pearson Correlation")
    print("  ③Precision@5")
    print("=" * 80)

    print("\n可用评估模式:")
    print("  1. 单个数据集评估")
    print("  2. 整体评估")

    eval_mode = input("请选择评估模式 (1/2): ").strip()

    if eval_mode == "2":
        agg_type = input("聚合类型 (equisize/equitemporal): ").strip()
        if agg_type not in ["equisize", "equitemporal"]:
            agg_type = "equisize"
            print(f"使用默认聚合类型: {agg_type}")

        print("\n开始整体评估...")
        print("注意：这将运行所有数据集的训练和评估，耗时较长。")
        confirm = input("是否继续? (y/n): ").strip().lower()

        if confirm == 'y':
            run_overall_evaluation(agg_type)
        else:
            print("已取消")
        print("\n" + "=" * 80)
        print("模块5执行完成！")
        print("=" * 80)
        return

    print("\n可用数据集:")
    datasets = list(DATA_PATHS.keys())
    for i, name in enumerate(datasets, 1):
        print(f"  {i}. {name}")

    print(f"\n可直接输入数据集名称，或输入数字选择:")
    print(f"  1: Helpdesk, 2: Sepsis, 3: BPIC2012, 4: HEL, 5:RTF, 6:BPIC2017, 7:BPIC2019, 8:BPIC2020")

    choice = input(f"请输入选择: ").strip()

    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(datasets):
            dataset_name = datasets[idx]
        else:
            dataset_name = "Helpdesk"
            print(f"无效选择，使用默认数据集: {dataset_name}")
    else:
        dataset_name = choice
        if dataset_name not in DATA_PATHS:
            dataset_name = "Helpdesk"
            print(f"数据集不存在，使用默认数据集: {dataset_name}")

    agg_type = input("聚合类型 (equisize/equitemporal): ").strip()
    if agg_type not in ["equisize", "equitemporal"]:
        agg_type = "equisize"
        print(f"使用默认聚合类型: {agg_type}")

    print("\n选择评估模式:")
    print("  1. 单个训练集对比 (指定训练集大小)")
    print("  2. 完整对比 (25, 50, 75)")

    mode = input("请输入选择 (1/2): ").strip()

    if mode == "1":
        try:
            train_size = int(input("训练集大小 (25/50/75): ").strip())
            if train_size not in [25, 50, 75]:
                train_size = 50
                print(f"使用默认训练集大小: {train_size}")
        except:
            train_size = 50
            print(f"使用默认训练集大小: {train_size}")
        run_dual_model_evaluation(dataset_name, train_size, agg_type)
    else:
        run_full_dual_evaluation(dataset_name, agg_type)

    print("\n" + "=" * 80)
    print("模块5执行完成！")
    print("=" * 80)


def main():
    print("=" * 80)
    print("预测框架")
    print("支持以下模式:")
    print("  1. ER评估")
    print("  2. J-MAPE评估")
    print("  3. 消融实验")
    print("  4. 运行所有实验")
    print("  5. 指标报告")
    print("=" * 80)

    print("\n可用数据集:")
    for i, name in enumerate(DATA_PATHS.keys(), 1):
        print(f"  {i}. {name}")

    try:
        mode = int(input("\n请输入选择 (1/2/3/4/5): ").strip())
    except:
        mode = 3
        print(f"使用默认选择: 消融实验")

    dataset_name = input(f"请输入数据集名称 {list(DATA_PATHS.keys())}: ").strip()
    if not dataset_name or dataset_name not in DATA_PATHS:
        dataset_name = "Helpdesk"
        print(f"使用默认数据集: {dataset_name}")

    if mode == 1:
        print("\n" + "=" * 80)
        print("运行ER评估实验")
        print("=" * 80)

        train_sizes = [25, 50, 75]
        agg_types = ["equisize", "equitemporal"]
        use_attention = input("是否使用注意力机制? (y/n): ").strip().lower() == 'y'

        for train_size in train_sizes:
            for agg_type in agg_types:
                run_er_evaluation(dataset_name, train_size, agg_type, use_attention)

    elif mode == 2:
        print("\n" + "=" * 80)
        print("运行J-MAPE评估实验")
        print("=" * 80)

        train_sizes = [25, 50, 75]
        agg_types = ["equisize", "equitemporal"]

        for train_size in train_sizes:
            for agg_type in agg_types:
                run_jmape_evaluation(dataset_name, train_size, agg_type)

    elif mode == 3:
        print("\n" + "=" * 80)
        print("运行消融实验")
        print("=" * 80)

        agg_type = input("聚合类型 (equisize/equitemporal): ").strip()
        if agg_type not in ["equisize", "equitemporal"]:
            agg_type = "equisize"

        run_ablation_experiment(dataset_name, agg_type)

    elif mode == 4:
        print("\n" + "=" * 80)
        print("运行所有实验")
        print("=" * 80)

        print("\n 运行ER评估实验...")
        for train_size in [25, 50, 75]:
            for agg_type in ["equisize", "equitemporal"]:
                run_er_evaluation(dataset_name, train_size, agg_type, use_attention=False)

        print("\n 运行J-MAPE评估实验...")
        for train_size in [25, 50, 75]:
            for agg_type in ["equisize", "equitemporal"]:
                run_jmape_evaluation(dataset_name, train_size, agg_type)

        print("\n 运行消融实验...")
        for agg_type in ["equisize", "equitemporal"]:
            run_ablation_experiment(dataset_name, agg_type)

    elif mode == 5:
        module5_main()

    print("\n" + "=" * 80)
    print("所有实验执行完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
