import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score


def eval_load_forecasting(
    results: torch.Tensor,
    truths: torch.Tensor,
    eps: float = 1e-6,
    verbose: bool = True
):
    """
    负荷预测评估函数（回归）

    Args:
        results: (N,) or (N, 1) torch tensor, 模型预测值
        truths:  (N,) or (N, 1) torch tensor, 真实负荷
        eps:     防止除 0
        verbose: 是否打印结果

    Returns:
        metrics: dict
    """

    # -------- 1. Tensor -> numpy --------
    preds = results.view(-1).detach().cpu().numpy()
    trues = truths.view(-1).detach().cpu().numpy()

    # -------- 2. 基本回归指标 --------
    mae = np.mean(np.abs(preds - trues))
    rmse = np.sqrt(mean_squared_error(trues, preds))
    mape = np.mean(np.abs((preds - trues) / (trues + eps))) * 100.0
    r2 = r2_score(trues, preds)

    # Pearson correlation
    if np.std(preds) < eps or np.std(trues) < eps:
        corr = 0.0
    else:
        corr = np.corrcoef(preds, trues)[0, 1]

    # -------- 3. 结果汇总 --------
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE(%)": mape,
        "R2": r2,
        "Corr": corr
    }

    if verbose:
        print("Load Forecasting Evaluation")
        print("-" * 40)
        for k, v in metrics.items():
            print(f"{k:10s}: {v:.4f}")
        print("-" * 40)

    return metrics
