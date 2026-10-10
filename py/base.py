"""
_base.py
传统基线方法对比实验
"""

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

warnings.filterwarnings("ignore")


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
        self.test_intervals = 25
        self.train_sizes = [25, 50, 75]
        self.aggregation_types = ["equisize", "equitemporal"]

        self.jmape_alpha = 0.5
        self.edge_threshold = 0.5
        self.mape_epsilon = 1.0

        self.result_dir = Path("./baseline_experiment_results_optimized")
        self.result_dir.mkdir(parents=True, exist_ok=True)

        # True：依次运行全部8个数据集；False：只运行 DATASETS_TO_RUN
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

        # keep_default_na=False 避免案例编号“NA”被误判为空值
        df = pd.read_csv(self.file_path, keep_default_na=False)

        required = [self.case_col, self.activity_col, self.time_col]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(
                f"{self.dataset_name} 缺少列 {missing}；当前列为: {df.columns.tolist()}"
            )

        # 带时区与混合偏移统一转换到UTC，再移除时区
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

        # 稳定排序，保证同一时间戳下保留原CSV次序
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

        # sort=False，沿用预处理后的案例顺序
        for trace_no, (_, group) in enumerate(
            self.df.groupby(self.case_col, sort=False)
        ):
            if len(group) < 2:
                continue

            activities = group[self.activity_col].astype(str).to_numpy()
            timestamps = group[self.time_col].to_numpy()

            frame = pd.DataFrame(
                {
                    "trace_no": trace_no,
                    "a1": activities[:-1],
                    "a2": activities[1:],
                    # 直接后继关系采用后继事件时间
                    "timestamp": timestamps[1:],
                }
            )
            frames.append(frame)

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

        # 每个关系按其全局排序位置分配区间，避免 total // intervals 遗漏
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


def calculate_jmape_components(
    pred_dfg: np.ndarray,
    real_dfg: np.ndarray,
    alpha: float = 0.5,
    edge_threshold: float = 0.5,
    epsilon: float = 1.0,
) -> Dict[str, float]:
    pred = np.asarray(pred_dfg, dtype=np.float64)
    real = np.asarray(real_dfg, dtype=np.float64)

    if pred.shape != real.shape:
        raise ValueError(f"预测与真实DFG形状不一致: {pred.shape} != {real.shape}")

    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    pred = np.maximum(pred, 0.0)
    real = np.nan_to_num(real, nan=0.0, posinf=0.0, neginf=0.0)
    real = np.maximum(real, 0.0)

    real_mask = real > 0.0
    pred_mask = pred >= edge_threshold

    if real_mask.any():
        frequency_mape = float(
            np.mean(
                np.abs(pred[real_mask] - real[real_mask])
                / np.maximum(real[real_mask], epsilon)
            )
        )
    else:
        # 真实图无边时：若预测也无边，频次误差为0；否则按预测总量惩罚
        frequency_mape = (
            0.0
            if not pred_mask.any()
            else float(np.mean(pred[pred_mask]) / max(epsilon, 1.0))
        )

    intersection = int(np.logical_and(real_mask, pred_mask).sum())
    union = int(np.logical_or(real_mask, pred_mask).sum())
    jaccard = float(intersection / union) if union > 0 else 1.0
    structure_error = 1.0 - jaccard

    jmape = alpha * frequency_mape + (1.0 - alpha) * structure_error

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
        raise ValueError(
            f"预测步数与测试步数不一致: {len(pred_series)} != {len(real_series)}"
        )

    components = [
        calculate_jmape_components(
            pred,
            real,
            alpha=config.jmape_alpha,
            edge_threshold=config.edge_threshold,
            epsilon=config.mape_epsilon,
        )
        for pred, real in zip(pred_series, real_series)
    ]

    def stats(key: str) -> Dict[str, float]:
        values = np.asarray([item[key] for item in components], dtype=np.float64)
        return {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "std": float(values.std()),
        }

    return {
        "jmape_stats": stats("jmape"),
        "frequency_mape_stats": stats("frequency_mape"),
        "jaccard_stats": stats("jaccard"),
        "structure_error_stats": stats("structure_error"),
        "jmape_list": [item["jmape"] for item in components],
        "frequency_mape_list": [item["frequency_mape"] for item in components],
        "jaccard_list": [item["jaccard"] for item in components],
        "structure_error_list": [item["structure_error"] for item in components],
    }


# 兼容旧函数名称与旧JSON键
def calculate_newscore(
    pred_dfg: np.ndarray,
    real_dfg: np.ndarray,
    alpha: float = 0.5,
) -> float:
    return calculate_jmape_components(
        pred_dfg,
        real_dfg,
        alpha=alpha,
        edge_threshold=config.edge_threshold,
        epsilon=config.mape_epsilon,
    )["jmape"]


def calculate_newscore_stats(
    pred_series: np.ndarray,
    real_series: np.ndarray,
    alpha: float = 0.5,
) -> Dict[str, float]:
    old_alpha = config.jmape_alpha
    try:
        config.jmape_alpha = alpha
        return calculate_jmape_series(pred_series, real_series)["jmape_stats"]
    finally:
        config.jmape_alpha = old_alpha


class BaselineMethods:
    @staticmethod
    def naive_forecast(series: np.ndarray, pred_len: int) -> np.ndarray:
        # 保留原NAV含义：训练区间均值
        mean_value = np.mean(series, axis=0)
        return np.tile(mean_value, (pred_len, 1))

    @staticmethod
    def ar_forecast(
        series: np.ndarray,
        pred_len: int,
        lags: int = 5,
    ) -> np.ndarray:
        predictions: List[np.ndarray] = []

        for feature_idx in range(series.shape[1]):
            ts = np.asarray(series[:, feature_idx], dtype=np.float64)

            if len(ts) < 4 or np.std(ts) < 1e-8:
                predictions.append(np.full(pred_len, ts[-1]))
                continue

            try:
                max_lags = max(1, min(lags, len(ts) // 4))
                fitted = AutoReg(ts, lags=max_lags, old_names=False).fit()
                pred = np.asarray(fitted.forecast(pred_len), dtype=np.float64)
                pred = np.maximum(
                    np.nan_to_num(pred, nan=ts[-1], posinf=ts[-1], neginf=0.0),
                    0.0,
                )
            except Exception:
                pred = np.full(pred_len, ts[-1])

            predictions.append(pred)

        return np.asarray(predictions).T

    @staticmethod
    def arima_forecast(
        series: np.ndarray,
        pred_len: int,
        order: Tuple[int, int, int] = (1, 1, 1),
    ) -> np.ndarray:
        predictions: List[np.ndarray] = []

        for feature_idx in range(series.shape[1]):
            ts = np.asarray(series[:, feature_idx], dtype=np.float64)

            if len(ts) < 5 or np.std(ts) < 1e-8:
                predictions.append(np.full(pred_len, ts[-1]))
                continue

            try:
                actual_order = order if len(ts) >= 10 else (0, 1, 1)
                fitted = ARIMA(ts, order=actual_order).fit()
                pred = np.asarray(fitted.forecast(pred_len), dtype=np.float64)
                pred = np.maximum(
                    np.nan_to_num(pred, nan=ts[-1], posinf=ts[-1], neginf=0.0),
                    0.0,
                )
            except Exception:
                pred = np.full(pred_len, np.mean(ts))

            predictions.append(pred)

        return np.asarray(predictions).T

    @staticmethod
    def garch_forecast(series: np.ndarray, pred_len: int) -> np.ndarray:
        if not ARCH_AVAILABLE:
            print("    未安装 arch，GARCH自动回退为NAV。")
            return BaselineMethods.naive_forecast(series, pred_len)

        predictions: List[np.ndarray] = []

        for feature_idx in range(series.shape[1]):
            ts = np.asarray(series[:, feature_idx], dtype=np.float64)

            if len(ts) < 12 or np.std(ts) < 1e-8:
                predictions.append(np.full(pred_len, ts[-1]))
                continue

            try:
                returns = np.diff(ts)
                scale = np.std(returns)
                if scale < 1e-8:
                    raise ValueError("constant returns")

                standardized = (returns - returns.mean()) / scale
                fitted = arch_model(
                    standardized * 100.0,
                    vol="Garch",
                    p=1,
                    q=1,
                    dist="normal",
                ).fit(disp="off", show_warning=False)

                forecast = fitted.forecast(horizon=pred_len)
                pred_returns = (
                    forecast.mean.iloc[-1].to_numpy() / 100.0 * scale
                    + returns.mean()
                )
                pred = np.maximum(ts[-1] + np.cumsum(pred_returns), 0.0)
                pred = np.nan_to_num(pred, nan=ts[-1], posinf=ts[-1], neginf=0.0)
            except Exception:
                pred = np.full(pred_len, ts[-1])

            predictions.append(pred)

        return np.asarray(predictions).T

    @staticmethod
    def holt_winters_forecast(
        series: np.ndarray,
        pred_len: int,
    ) -> np.ndarray:
        predictions: List[np.ndarray] = []

        for feature_idx in range(series.shape[1]):
            ts = np.asarray(series[:, feature_idx], dtype=np.float64)

            if len(ts) < 4 or np.std(ts) < 1e-8:
                predictions.append(np.full(pred_len, ts[-1]))
                continue

            try:
                seasonal_periods = min(12, max(2, len(ts) // 3))
                if len(ts) >= 2 * seasonal_periods:
                    model = ExponentialSmoothing(
                        ts,
                        trend="add",
                        seasonal="add",
                        seasonal_periods=seasonal_periods,
                        initialization_method="estimated",
                    )
                else:
                    model = ExponentialSmoothing(
                        ts,
                        trend="add",
                        initialization_method="estimated",
                    )

                pred = np.asarray(model.fit().forecast(pred_len), dtype=np.float64)
                pred = np.maximum(
                    np.nan_to_num(pred, nan=ts[-1], posinf=ts[-1], neginf=0.0),
                    0.0,
                )
            except Exception:
                pred = np.full(pred_len, ts[-1])

            predictions.append(pred)

        return np.asarray(predictions).T


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

    if dfg_series.shape[0] != config.base_intervals:
        raise RuntimeError(f"DFG时间步数错误: {dfg_series.shape[0]}")

    return loader, dfg_series


def split_train_test(
    dfg_series: np.ndarray,
    train_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    test_start = train_size
    test_end = train_size + config.test_intervals

    if test_end > len(dfg_series):
        raise ValueError(
            f"数据不足：训练{train_size}步后无法取得完整{config.test_intervals}步测试集"
        )

    train_data = dfg_series[:train_size].copy()
    test_data = dfg_series[test_start:test_end].copy()

    if len(test_data) != config.test_intervals:
        raise RuntimeError("测试集必须严格包含25个紧随训练区间的时间步")

    return train_data, test_data


def run_baseline_experiment(
    dataset_name: str,
    train_size: int,
    aggregation_type: str,
) -> Dict[str, Any]:
    print("\n" + "=" * 88)
    print(
        f"基线实验: {dataset_name} | {aggregation_type} | "
        f"训练1-{train_size} | 预测{train_size + 1}-{train_size + 25}"
    )
    print("=" * 88)

    loader, dfg_series = prepare_dfg_series(dataset_name, aggregation_type)
    train_data, test_data = split_train_test(dfg_series, train_size)

    flatten_dim = loader.num_activities ** 2
    train_flat = train_data.reshape(train_size, flatten_dim)

    methods = {
        "NAV": lambda: BaselineMethods.naive_forecast(
            train_flat, config.test_intervals
        ),
        "AR": lambda: BaselineMethods.ar_forecast(
            train_flat, config.test_intervals, lags=5
        ),
        "ARIMA": lambda: BaselineMethods.arima_forecast(
            train_flat, config.test_intervals, order=(1, 1, 1)
        ),
        "GARCH": lambda: BaselineMethods.garch_forecast(
            train_flat, config.test_intervals
        ),
        "HW": lambda: BaselineMethods.holt_winters_forecast(
            train_flat, config.test_intervals
        ),
    }

    method_results: Dict[str, Any] = {}

    for method_name, method in methods.items():
        print(f"  运行 {method_name}...")
        try:
            pred_flat = np.asarray(method(), dtype=np.float64)
            expected_shape = (config.test_intervals, flatten_dim)

            if pred_flat.shape != expected_shape:
                raise ValueError(
                    f"{method_name}预测形状错误: {pred_flat.shape}, "
                    f"应为 {expected_shape}"
                )

            pred_flat = np.maximum(
                np.nan_to_num(pred_flat, nan=0.0, posinf=0.0, neginf=0.0),
                0.0,
            )
            pred_series = pred_flat.reshape(
                config.test_intervals,
                loader.num_activities,
                loader.num_activities,
            )

            metrics = calculate_jmape_series(pred_series, test_data)
            method_results[method_name] = {
                # 旧输出兼容
                "newscore_stats": metrics["jmape_stats"],
                "newscore_list": metrics["jmape_list"],
                # 新增可解释分量
                **metrics,
            }

            print(
                f"    J-MAPE={metrics['jmape_stats']['mean']:.4f}, "
                f"Frequency-MAPE={metrics['frequency_mape_stats']['mean']:.4f}, "
                f"Jaccard={metrics['jaccard_stats']['mean']:.4f}"
            )
        except Exception as exc:
            print(f"    {method_name}失败: {exc}")
            method_results[method_name] = {
                "status": "failed",
                "error": str(exc),
            }

    return {
        "dataset": dataset_name,
        "aggregation_type": aggregation_type,
        "train_size": train_size,
        "test_start": train_size + 1,
        "test_end": train_size + config.test_intervals,
        "num_activities": loader.num_activities,
        "metric": {
            "name": "J-MAPE",
            "alpha": config.jmape_alpha,
            "edge_threshold": config.edge_threshold,
            "epsilon": config.mape_epsilon,
        },
        "results": method_results,
    }


def plot_baseline_comparison(
    result: Dict[str, Any],
    save_dir: Path,
) -> None:
    valid_methods = [
        name
        for name, value in result["results"].items()
        if "jmape_list" in value
    ]
    if not valid_methods:
        return

    plt.figure(figsize=(12, 7))
    for method in valid_methods:
        values = result["results"][method]["jmape_list"]
        plt.plot(
            range(1, len(values) + 1),
            values,
            marker="o",
            linewidth=1.5,
            markersize=4,
            label=method,
        )

    plt.xlabel("Forecast Horizon")
    plt.ylabel("J-MAPE (Lower Is Better)")
    plt.title(
        f"{result['dataset']} | {result['aggregation_type']} | "
        f"Train={result['train_size']}"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = save_dir / (
        f"{result['dataset']}_{result['aggregation_type']}_"
        f"Train{result['train_size']}_baseline_comparison.png"
    )
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def run_all_baseline_experiments(dataset_name: str) -> List[Dict[str, Any]]:
    all_results: List[Dict[str, Any]] = []
    dataset_dir = config.result_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for aggregation_type in config.aggregation_types:
        for train_size in config.train_sizes:
            try:
                result = run_baseline_experiment(
                    dataset_name,
                    train_size,
                    aggregation_type,
                )
                all_results.append(result)

                result_file = dataset_dir / (
                    f"{dataset_name}_{aggregation_type}_"
                    f"Train{train_size}_baseline_NewScore.json"
                )
                with result_file.open("w", encoding="utf-8") as file:
                    json.dump(result, file, indent=4, ensure_ascii=False)

                plot_baseline_comparison(result, dataset_dir)
                print(f"结果已保存: {result_file}")
            except Exception as exc:
                print(
                    f"实验失败: {dataset_name} | {aggregation_type} | "
                    f"Train={train_size}: {exc}"
                )

    summary_file = dataset_dir / f"{dataset_name}_baseline_summary.json"
    with summary_file.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset": dataset_name,
                "evaluation_metric": "J-MAPE (lower is better)",
                "results": all_results,
            },
            file,
            indent=4,
            ensure_ascii=False,
        )

    return all_results


def main() -> None:
    print("=" * 88)
    print("传统基线方法实验：NAV / AR / ARIMA / GARCH / Holt-Winters")
    print("实验切分：前25/50/75步训练，分别预测紧随其后的25步")
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
            run_all_baseline_experiments(dataset_name)
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
