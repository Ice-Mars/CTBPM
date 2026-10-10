"""
_transformer.py
Transformer过程模型预测实验（CPU / Windows / PyCharm）
"""

import copy
import json
import math
import os
import random
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")
try:
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

print("使用设备: CPU")


DATA_PATHS: Dict[str, str] = {
    "Helpdesk": r"C:\Users\28726\Desktop\Helpdesk_Final_Numeric.csv",
    "Sepsis": r"C:\Users\28726\Desktop\Spesis\Sepsis_Core_Data.csv",
    "BPIC2012": r"C:\Users\28726\Desktop\BPIC2012\BPI_Challenge_2012_Processed.csv",
    "HEL": r"C:\Users\28726\Desktop\helpdesk2.csv",
    "RTF": r"C:\Users\28726\Desktop\RTF\RTFMP_Core_Data.csv",
    "BPIC2017": r"C:\Users\28726\Desktop\BPIC2017\BPI_Challenge_2017.csv",
    "BPIC2019": r"C:\Users\28726\Desktop\BPIC2019\BPI_Challenge_2019.csv",
    "BPIC2020": r"C:\Users\28726\Desktop\BPIC2020\BPI_Challenge_2020.csv",
}

COLUMN_MAPPINGS: Dict[str, Dict[str, str]] = {
    "Helpdesk": {
        "case_col": "CaseID",
        "activity_col": "ActivityII",
        "time_col": "CompleteTimestamp",
    },
    "Sepsis": {
        "case_col": "caseID",
        "activity_col": "activity",
        "time_col": "timestamp",
    },
    "BPIC2012": {
        "case_col": "trace_id",
        "activity_col": "event_concept_name",
        "time_col": "time_timestamp",
    },
    "HEL": {
        "case_col": "CaseID",
        "activity_col": "ActivityID",
        "time_col": "CompleteTimestamp",
    },
    "RTF": {
        "case_col": "caseID",
        "activity_col": "activity",
        "time_col": "timestamp",
    },
    "BPIC2017": {
        "case_col": "caseID",
        "activity_col": "activity",
        "time_col": "timestamp",
    },
    "BPIC2019": {
        "case_col": "caseID",
        "activity_col": "activity",
        "time_col": "timestamp",
    },
    "BPIC2020": {
        "case_col": "caseID",
        "activity_col": "activity",
        "time_col": "timestamp",
    },
}


class ExperimentConfig:
    def __init__(self) -> None:
        self.base_intervals = 100
        self.window_size = 10
        self.test_intervals = 25
        self.train_sizes = [25, 50, 75]
        self.aggregation_types = ["equisize", "equitemporal"]

        # CPU下适中的模型规模；三个训练长度保持一致，保证可比性
        self.d_model = 64
        self.nhead = 4
        self.num_layers = 2
        self.dim_feedforward = 128
        self.dropout = 0.1

        self.learning_rate = 1e-3
        self.max_epochs = 120
        self.batch_size = 8
        self.early_stopping_patience = 15
        self.val_steps = 5
        self.min_final_epochs = 1

        self.jmape_alpha = 0.5
        self.edge_threshold = 0.5
        self.mape_epsilon = 1.0

        self.result_dir = Path("./transformer_experiment_results_optimized")
        self.result_dir.mkdir(parents=True, exist_ok=True)

        self.run_all_datasets = True
        self.datasets_to_run = list(DATA_PATHS.keys())


config = ExperimentConfig()


class ProcessMiningDataLoader:
    def __init__(self, dataset_name: str) -> None:
        if dataset_name not in DATA_PATHS:
            raise KeyError(f"未知数据集: {dataset_name}")

        self.dataset_name = dataset_name
        self.file_path = Path(DATA_PATHS[dataset_name])

        mapping = COLUMN_MAPPINGS[dataset_name]
        self.case_col = mapping["case_col"]
        self.activity_col = mapping["activity_col"]
        self.time_col = mapping["time_col"]

        self.df = self._load_and_preprocess()
        self.activity2id, self.id2activity = self._create_activity_mapping()
        self.num_activities = len(self.activity2id)

        print(
            f"数据集: {dataset_name}, 案例数: {self.df[self.case_col].nunique()}, "
            f"活动数: {self.num_activities}"
        )

    def _load_and_preprocess(self) -> pd.DataFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV文件不存在: {self.file_path}")

        df = pd.read_csv(self.file_path, keep_default_na=False)

        required = [self.case_col, self.activity_col, self.time_col]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(
                f"{self.dataset_name} 缺少列 {missing}；当前列为: {df.columns.tolist()}"
            )

        timestamps = pd.to_datetime(
            df[self.time_col],
            errors="coerce",
            utc=True,
        ).dt.tz_convert(None)

        df = df.assign(**{self.time_col: timestamps})
        df = df[
            (df[self.case_col].astype(str).str.len() > 0)
            & (df[self.activity_col].astype(str).str.len() > 0)
            & df[self.time_col].notna()
        ].copy()

        df = df.sort_values(
            [self.case_col, self.time_col],
            kind="stable",
        ).reset_index(drop=True)

        if df.empty:
            raise ValueError(f"{self.dataset_name} 清洗后没有有效事件")

        return df

    def _create_activity_mapping(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        activities = sorted(self.df[self.activity_col].astype(str).unique())
        activity2id = {activity: idx for idx, activity in enumerate(activities)}
        id2activity = {idx: activity for activity, idx in activity2id.items()}
        return activity2id, id2activity

    def extract_sorted_direct_follows(self) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []

        for trace_no, (_, group) in enumerate(
            self.df.groupby(self.case_col, sort=False)
        ):
            if len(group) < 2:
                continue

            activities = group[self.activity_col].astype(str).to_numpy()
            timestamps = group[self.time_col].to_numpy()

            frames.append(
                pd.DataFrame(
                    {
                        "trace_no": trace_no,
                        "a1": activities[:-1],
                        "a2": activities[1:],
                        "timestamp": timestamps[1:],
                    }
                )
            )

        if not frames:
            raise ValueError(f"{self.dataset_name} 未提取到直接后继关系")

        sap_df = pd.concat(frames, ignore_index=True)
        sap_df = sap_df.sort_values("timestamp", kind="stable").reset_index(drop=True)
        sap_df["a1_encoded"] = (
            sap_df["a1"].map(self.activity2id).astype(np.int64)
        )
        sap_df["a2_encoded"] = (
            sap_df["a2"].map(self.activity2id).astype(np.int64)
        )

        print(f"直接后继关系数量: {len(sap_df)}")
        return sap_df


class DFAggregator:
    @staticmethod
    def equisize_aggregation(
        sap_df: pd.DataFrame,
        num_intervals: int,
        num_activities: int,
    ) -> np.ndarray:
        matrix = np.zeros(
            (num_intervals, num_activities, num_activities),
            dtype=np.float32,
        )

        total = len(sap_df)
        if total == 0:
            return matrix

        interval_ids = np.minimum(
            np.arange(total, dtype=np.int64) * num_intervals // total,
            num_intervals - 1,
        )
        a1 = sap_df["a1_encoded"].to_numpy(dtype=np.int64)
        a2 = sap_df["a2_encoded"].to_numpy(dtype=np.int64)
        np.add.at(matrix, (interval_ids, a1, a2), 1.0)
        return matrix

    @staticmethod
    def equitemporal_aggregation(
        sap_df: pd.DataFrame,
        num_intervals: int,
        num_activities: int,
    ) -> np.ndarray:
        matrix = np.zeros(
            (num_intervals, num_activities, num_activities),
            dtype=np.float32,
        )

        timestamp_series = pd.to_datetime(
            sap_df["timestamp"],
            errors="coerce",
            utc=True,
        ).dt.tz_convert(None)
        valid = timestamp_series.notna()

        if not valid.any():
            raise ValueError("等时聚合失败：没有有效时间戳")

        timestamps = timestamp_series.loc[valid].astype("int64").to_numpy()
        a1 = sap_df.loc[valid, "a1_encoded"].to_numpy(dtype=np.int64)
        a2 = sap_df.loc[valid, "a2_encoded"].to_numpy(dtype=np.int64)

        ts_min = int(timestamps.min())
        ts_max = int(timestamps.max())

        if ts_min == ts_max:
            interval_ids = np.zeros(len(timestamps), dtype=np.int64)
        else:
            edges = np.linspace(
                ts_min,
                ts_max,
                num_intervals + 1,
                dtype=np.float64,
            )
            interval_ids = np.searchsorted(
                edges,
                timestamps,
                side="right",
            ) - 1
            interval_ids = np.clip(interval_ids, 0, num_intervals - 1)

        np.add.at(matrix, (interval_ids, a1, a2), 1.0)
        return matrix


class PMFDataset(Dataset):
    """
    单步预测样本：
    [D_t, ..., D_(t+W-1)] -> D_(t+W)

    数据已在外部执行log1p转换。
    """

    def __init__(self, dfg_series: np.ndarray, window_size: int) -> None:
        if dfg_series.ndim != 3:
            raise ValueError(f"DFG序列应为3维，当前为: {dfg_series.shape}")

        self.window_size = window_size
        self.flatten_dim = dfg_series.shape[1] * dfg_series.shape[2]
        self.series = dfg_series.reshape(len(dfg_series), self.flatten_dim)

        sample_count = len(self.series) - window_size
        if sample_count <= 0:
            raise ValueError(
                f"数据长度{len(self.series)}不足以使用窗口{window_size}"
            )

        self.inputs = np.stack(
            [
                self.series[index:index + window_size]
                for index in range(sample_count)
            ],
            axis=0,
        ).astype(np.float32)
        self.targets = np.stack(
            [
                self.series[index + window_size]
                for index in range(sample_count)
            ],
            axis=0,
        ).astype(np.float32)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.inputs[index]),
            torch.from_numpy(self.targets[index]),
        )


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float,
        max_len: int = 500,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.encoding[:, :x.size(1)]
        return self.dropout(x)


class ProcessModelTransformer(nn.Module):
    """
    Encoder-only Transformer:
    输入最近W个DFG，使用最后一个时间位置的编码预测下一DFG。
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.position = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.output_projection = nn.Linear(d_model, input_dim)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(src) * math.sqrt(
            self.input_projection.out_features
        )
        x = self.position(x)
        encoded = self.encoder(x)
        # 模型输出为log1p空间，不在这里使用ReLU
        return self.output_projection(encoded[:, -1, :])


def create_model(input_dim: int) -> ProcessModelTransformer:
    return ProcessModelTransformer(
        input_dim=input_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
    )


def temporal_train_val_datasets(
    train_log: np.ndarray,
) -> Tuple[PMFDataset, PMFDataset]:
    """
    最后val_steps个训练时间步作为验证目标。
    验证序列额外包含其前面的window_size个上下文时间步。
    """
    min_fit_length = config.window_size + 1
    max_val_steps = len(train_log) - min_fit_length
    val_steps = min(config.val_steps, max_val_steps)

    if val_steps < 1:
        raise ValueError(
            f"训练区间长度{len(train_log)}不足以进行内部验证"
        )

    split_index = len(train_log) - val_steps
    fit_series = train_log[:split_index]

    val_context_start = split_index - config.window_size
    val_series = train_log[val_context_start:]

    train_dataset = PMFDataset(fit_series, config.window_size)
    val_dataset = PMFDataset(val_series, config.window_size)

    # val_dataset恰好产生val_steps个目标
    return train_dataset, val_dataset


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0

    for src, target in loader:
        src = src.to(DEVICE)
        target = target.to(DEVICE)

        if training:
            optimizer.zero_grad()

        output = model(src)
        loss = criterion(output, target)

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        batch_size = src.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("DataLoader为空")

    return total_loss / total_samples


def select_best_epoch(
    train_log: np.ndarray,
    input_dim: int,
) -> Tuple[int, List[float], List[float]]:
    train_dataset, val_dataset = temporal_train_val_datasets(train_log)

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config.batch_size, len(train_dataset)),
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(config.batch_size, len(val_dataset)),
        shuffle=False,
    )

    model = create_model(input_dim).to(DEVICE)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 1
    no_improvement = 0
    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(1, config.max_epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer)

        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, criterion, None)

        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"  验证阶段 Epoch {epoch:03d}: "
                f"Train={train_loss:.6f}, Val={val_loss:.6f}"
            )

        if no_improvement >= config.early_stopping_patience:
            break

    model.load_state_dict(best_state)
    print(f"  内部验证选择最佳epoch: {best_epoch}, Val={best_val:.6f}")
    return best_epoch, train_losses, val_losses


def train_final_model(
    train_log: np.ndarray,
    input_dim: int,
    epochs: int,
) -> Tuple[nn.Module, List[float]]:
    """
    恢复相同随机种子后，在完整前25/50/75步上重新训练。
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    dataset = PMFDataset(train_log, config.window_size)
    loader = DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=True,
    )

    model = create_model(input_dim).to(DEVICE)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-4,
    )

    final_epochs = max(config.min_final_epochs, int(epochs))
    losses: List[float] = []

    for epoch in range(1, final_epochs + 1):
        loss = run_epoch(model, loader, criterion, optimizer)
        losses.append(float(loss))

        if epoch == 1 or epoch % 10 == 0 or epoch == final_epochs:
            print(
                f"  完整训练 Epoch {epoch:03d}/{final_epochs}: "
                f"Loss={loss:.6f}"
            )

    return model, losses


def recursive_prediction(
    model: nn.Module,
    train_log: np.ndarray,
    pred_len: int,
    num_activities: int,
) -> np.ndarray:
    if len(train_log) < config.window_size:
        raise ValueError("训练数据不足以构造初始预测窗口")

    flatten_dim = num_activities * num_activities
    window = train_log[-config.window_size:].reshape(
        1,
        config.window_size,
        flatten_dim,
    )
    window_tensor = torch.from_numpy(window.astype(np.float32)).to(DEVICE)

    predictions: List[np.ndarray] = []
    model.eval()

    with torch.no_grad():
        for _ in range(pred_len):
            pred_log_flat = model(window_tensor)

            # 从log1p空间恢复原始计数
            pred_flat = torch.expm1(pred_log_flat)
            pred_flat = torch.clamp(pred_flat, min=0.0)

            pred_matrix = pred_flat.reshape(
                num_activities,
                num_activities,
            ).cpu().numpy()
            predictions.append(pred_matrix.astype(np.float32))

            # 下一轮输入仍放回log1p空间
            next_log = torch.log1p(pred_flat).reshape(1, 1, flatten_dim)
            window_tensor = torch.cat(
                [window_tensor[:, 1:, :], next_log],
                dim=1,
            )

    return np.stack(predictions, axis=0)


def calculate_jmape_components(
    pred_dfg: np.ndarray,
    real_dfg: np.ndarray,
) -> Dict[str, float]:
    pred = np.maximum(
        np.nan_to_num(
            np.asarray(pred_dfg, dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        0.0,
    )
    real = np.maximum(
        np.nan_to_num(
            np.asarray(real_dfg, dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        0.0,
    )

    real_mask = real > 0.0
    pred_mask = pred >= config.edge_threshold

    if real_mask.any():
        frequency_mape = float(
            np.mean(
                np.abs(pred[real_mask] - real[real_mask])
                / np.maximum(real[real_mask], config.mape_epsilon)
            )
        )
    else:
        frequency_mape = (
            0.0
            if not pred_mask.any()
            else float(
                np.mean(pred[pred_mask])
                / max(config.mape_epsilon, 1.0)
            )
        )

    intersection = int(np.logical_and(real_mask, pred_mask).sum())
    union = int(np.logical_or(real_mask, pred_mask).sum())
    jaccard = float(intersection / union) if union > 0 else 1.0
    structure_error = 1.0 - jaccard

    jmape = (
        config.jmape_alpha * frequency_mape
        + (1.0 - config.jmape_alpha) * structure_error
    )

    return {
        "jmape": float(jmape),
        "frequency_mape": float(frequency_mape),
        "jaccard": float(jaccard),
        "structure_error": float(structure_error),
    }


def calculate_jmape_series(
    pred_series: np.ndarray,
    real_series: np.ndarray,
) -> Dict[str, Any]:
    if len(pred_series) != len(real_series):
        raise ValueError("预测与测试时间步数不一致")

    components = [
        calculate_jmape_components(pred, real)
        for pred, real in zip(pred_series, real_series)
    ]

    def statistics(key: str) -> Dict[str, float]:
        values = np.asarray(
            [item[key] for item in components],
            dtype=np.float64,
        )
        return {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "std": float(values.std()),
        }

    return {
        "jmape_stats": statistics("jmape"),
        "frequency_mape_stats": statistics("frequency_mape"),
        "jaccard_stats": statistics("jaccard"),
        "structure_error_stats": statistics("structure_error"),
        "jmape_list": [item["jmape"] for item in components],
        "frequency_mape_list": [item["frequency_mape"] for item in components],
        "jaccard_list": [item["jaccard"] for item in components],
        "structure_error_list": [item["structure_error"] for item in components],
    }


# 兼容旧代码函数名称
def calculate_newscore(
    pred_dfg: np.ndarray,
    real_dfg: np.ndarray,
    alpha: float = 0.5,
) -> float:
    original_alpha = config.jmape_alpha
    try:
        config.jmape_alpha = alpha
        return calculate_jmape_components(pred_dfg, real_dfg)["jmape"]
    finally:
        config.jmape_alpha = original_alpha


def calculate_newscore_stats(
    pred_series: np.ndarray,
    real_series: np.ndarray,
    alpha: float = 0.5,
) -> Dict[str, float]:
    original_alpha = config.jmape_alpha
    try:
        config.jmape_alpha = alpha
        return calculate_jmape_series(pred_series, real_series)["jmape_stats"]
    finally:
        config.jmape_alpha = original_alpha


def prepare_dfg_series(
    dataset_name: str,
    aggregation_type: str,
) -> Tuple[ProcessMiningDataLoader, np.ndarray]:
    loader = ProcessMiningDataLoader(dataset_name)
    sap_df = loader.extract_sorted_direct_follows()

    if aggregation_type == "equisize":
        dfg_series = DFAggregator.equisize_aggregation(
            sap_df,
            config.base_intervals,
            loader.num_activities,
        )
    elif aggregation_type == "equitemporal":
        dfg_series = DFAggregator.equitemporal_aggregation(
            sap_df,
            config.base_intervals,
            loader.num_activities,
        )
    else:
        raise ValueError(f"未知聚合类型: {aggregation_type}")

    return loader, dfg_series


def split_train_test(
    dfg_series: np.ndarray,
    train_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    test_start = train_size
    test_end = train_size + config.test_intervals

    if test_end > len(dfg_series):
        raise ValueError(
            f"训练{train_size}步后没有完整的25步连续测试区间"
        )

    train_data = dfg_series[:train_size].copy()
    test_data = dfg_series[test_start:test_end].copy()

    if len(test_data) != config.test_intervals:
        raise RuntimeError("测试数据必须严格为紧随训练区间的25步")

    return train_data, test_data


def plot_results(
    metrics: Dict[str, Any],
    dataset_name: str,
    aggregation_type: str,
    train_size: int,
    save_dir: Path,
) -> None:
    plt.figure(figsize=(12, 7))
    horizon = range(1, config.test_intervals + 1)

    plt.plot(
        horizon,
        metrics["jmape_list"],
        marker="o",
        linewidth=1.8,
        markersize=4,
        label="J-MAPE",
    )
    plt.plot(
        horizon,
        metrics["frequency_mape_list"],
        linewidth=1.3,
        label="Frequency-MAPE",
    )
    plt.plot(
        horizon,
        metrics["structure_error_list"],
        linewidth=1.3,
        label="Structure Error",
    )

    plt.xlabel("Forecast Horizon")
    plt.ylabel("Error")
    plt.title(
        f"{dataset_name} | {aggregation_type} | Train={train_size}"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = save_dir / (
        f"{dataset_name}_{aggregation_type}_"
        f"Train{train_size}_NewScore.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_training_losses(
    train_losses: List[float],
    val_losses: List[float],
    final_losses: List[float],
    dataset_name: str,
    aggregation_type: str,
    train_size: int,
    save_dir: Path,
) -> None:
    plt.figure(figsize=(12, 7))
    plt.plot(
        range(1, len(train_losses) + 1),
        train_losses,
        label="Internal Train Loss",
    )
    plt.plot(
        range(1, len(val_losses) + 1),
        val_losses,
        label="Internal Validation Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Smooth L1 Loss")
    plt.title(
        f"Validation Training | {dataset_name} | "
        f"{aggregation_type} | Train={train_size}"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = save_dir / (
        f"{dataset_name}_{aggregation_type}_"
        f"Train{train_size}_validation_loss.png"
    )
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 7))
    plt.plot(
        range(1, len(final_losses) + 1),
        final_losses,
        label="Full-Training Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Smooth L1 Loss")
    plt.title(
        f"Full Training | {dataset_name} | "
        f"{aggregation_type} | Train={train_size}"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = save_dir / (
        f"{dataset_name}_{aggregation_type}_"
        f"Train{train_size}_final_loss.png"
    )
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def run_experiment(
    dataset_name: str,
    train_size: int,
    aggregation_type: str,
) -> Dict[str, Any]:
    print("\n" + "=" * 88)
    print(
        f"Transformer实验: {dataset_name} | {aggregation_type} | "
        f"训练1-{train_size} | 预测{train_size + 1}-{train_size + 25}"
    )
    print("=" * 88)

    loader, dfg_series = prepare_dfg_series(
        dataset_name,
        aggregation_type,
    )
    train_data, test_data = split_train_test(dfg_series, train_size)

    # log1p无须拟合参数，不会泄漏测试信息
    train_log = np.log1p(train_data).astype(np.float32)
    input_dim = loader.num_activities ** 2

    best_epoch, internal_train_losses, val_losses = select_best_epoch(
        train_log,
        input_dim,
    )
    model, final_train_losses = train_final_model(
        train_log,
        input_dim,
        best_epoch,
    )

    pred_series = recursive_prediction(
        model,
        train_log,
        config.test_intervals,
        loader.num_activities,
    )

    if pred_series.shape != test_data.shape:
        raise RuntimeError(
            f"预测形状与测试形状不一致: "
            f"{pred_series.shape} != {test_data.shape}"
        )

    metrics = calculate_jmape_series(pred_series, test_data)

    print(
        f"J-MAPE={metrics['jmape_stats']['mean']:.4f}, "
        f"Frequency-MAPE={metrics['frequency_mape_stats']['mean']:.4f}, "
        f"Jaccard={metrics['jaccard_stats']['mean']:.4f}"
    )

    result_dir = config.result_dir / dataset_name
    result_dir.mkdir(parents=True, exist_ok=True)

    plot_results(
        metrics,
        dataset_name,
        aggregation_type,
        train_size,
        result_dir,
    )
    plot_training_losses(
        internal_train_losses,
        val_losses,
        final_train_losses,
        dataset_name,
        aggregation_type,
        train_size,
        result_dir,
    )

    pred_path = result_dir / (
        f"pred_series_{dataset_name}_{aggregation_type}_"
        f"Train{train_size}.npy"
    )
    test_path = result_dir / (
        f"test_series_{dataset_name}_{aggregation_type}_"
        f"Train{train_size}.npy"
    )
    model_path = result_dir / (
        f"model_{dataset_name}_{aggregation_type}_"
        f"Train{train_size}_seed{SEED}.pth"
    )

    np.save(pred_path, pred_series)
    np.save(test_path, test_data)
    torch.save(model.state_dict(), model_path)

    result: Dict[str, Any] = {
        "dataset": dataset_name,
        "aggregation_type": aggregation_type,
        "train_size": int(train_size),
        "test_start": int(train_size + 1),
        "test_end": int(train_size + config.test_intervals),
        "num_activities": int(loader.num_activities),
        "window_size": int(config.window_size),
        "best_epoch": int(best_epoch),
        "metric": {
            "name": "J-MAPE",
            "alpha": config.jmape_alpha,
            "edge_threshold": config.edge_threshold,
            "epsilon": config.mape_epsilon,
        },
        # 旧输出兼容
        "newscore_stats": metrics["jmape_stats"],
        "newscore_list": metrics["jmape_list"],
        "train_losses": internal_train_losses,
        "val_losses": val_losses,
        # 新增完整结果
        **metrics,
        "final_train_losses": final_train_losses,
        "pred_series_shape": list(pred_series.shape),
        "test_series_shape": list(test_data.shape),
        "pred_series_path": str(pred_path),
        "test_series_path": str(test_path),
        "model_path": str(model_path),
    }

    result_file = result_dir / (
        f"{dataset_name}_{aggregation_type}_"
        f"Train{train_size}_NewScore.json"
    )
    with result_file.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=4, ensure_ascii=False)

    print(f"结果已保存: {result_file}")
    return result


def run_all_experiments(dataset_name: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for aggregation_type in config.aggregation_types:
        for train_size in config.train_sizes:
            try:
                results.append(
                    run_experiment(
                        dataset_name,
                        train_size,
                        aggregation_type,
                    )
                )
            except Exception as exc:
                print(
                    f"实验失败: {dataset_name} | {aggregation_type} | "
                    f"Train={train_size}: {exc}"
                )

    summary = {
        "dataset": dataset_name,
        "evaluation_metric": "J-MAPE (lower is better)",
        "results": results,
    }
    summary_path = config.result_dir / f"{dataset_name}_summary_NewScore.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    return results


def main() -> None:
    print("=" * 88)
    print("Transformer过程模型预测实验")
    print("前25/50/75步训练，分别递归预测紧随其后的25步")
    print("=" * 88)

    datasets = (
        list(DATA_PATHS.keys())
        if config.run_all_datasets
        else config.datasets_to_run
    )

    failures: Dict[str, str] = {}
    for dataset_name in datasets:
        print(f"\n开始处理数据集: {dataset_name}")
        try:
            run_all_experiments(dataset_name)
        except Exception as exc:
            failures[dataset_name] = str(exc)
            print(f"{dataset_name}整体处理失败: {exc}")

    print("\n" + "=" * 88)
    if failures:
        print("以下数据集处理失败：")
        for name, error in failures.items():
            print(f"  {name}: {error}")
    else:
        print("全部8个数据集处理完成。")
    print(f"结果目录: {config.result_dir.resolve()}")
    print("=" * 88)


if __name__ == "__main__":
    main()
