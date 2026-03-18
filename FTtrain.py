"""
FT-Transformer (tabular) + Heteroscedastic Lognormal head (mu, sigma)
Target: y = Ls_Area / Shape_Area  (ONLY Landslide==1),  z = log(y)
Loss: Negative log-likelihood of Normal on z with predicted mu(x), sigma(x)
Metrics: MAE/RMSE on z, NLL, 90% coverage, Tail MAE (top 10% y)

Dependencies:
  pip install torch pandas numpy scikit-learn
"""
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from data_yellow.data_GorkhaSize.FTTransformer import FTTransformer, TabDataset, encode_categoricals, evaluate
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ----------------------------
# Reproducibility
# ----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# Loss & metrics
# ----------------------------
def lognormal_nll_z(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    z: torch.Tensor,
    lam: float = 1e-3,   # 正则强度
) -> torch.Tensor:

    nll = (z - mu) ** 2 / (2 * sigma ** 2) + torch.log(sigma)

    # 轻微正则：防止 sigma 无限放大
    sigma_reg = lam * (sigma ** 2)

    return torch.mean(nll + sigma_reg)

# ----------------------------
# Train loop
# ----------------------------
def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    losses = []
    for x_num, x_cat, y_z in loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y_z = y_z.to(device)

        mu, sigma = model(x_num, x_cat)
        loss = lognormal_nll_z(mu, sigma, y_z)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def plot_regression_results(y_true_z, y_pred_z, title_suffix=""):
    """
    绘制对数空间(z)和原始空间(y)的对比散点图
    """
    # 计算原始空间的值
    y_true_raw = np.exp(y_true_z)
    y_pred_raw = np.exp(y_pred_z)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- 图1: 对数空间 (z = log(y_ratio)) ---
    sns.regplot(x=y_true_z, y=y_pred_z, ax=ax1,
                scatter_kws={'alpha': 0.4, 's': 10}, line_kws={'color': 'red'})
    ax1.set_title(f'Log-Space (z) Regression')
    ax1.set_xlabel('True log(y_ratio)')
    ax1.set_ylabel('Predicted mu')
    # 绘制1:1对角线
    ax1.plot([y_true_z.min(), y_true_z.max()], [y_true_z.min(), y_true_z.max()], 'k--', alpha=0.7)

    # --- 图2: 原始空间 (y = y_ratio) ---
    sns.scatterplot(x=y_true_raw, y=y_pred_raw, ax=ax2, alpha=0.4, s=15)
    ax2.set_title(f'Original-Space (y) Comparison')
    ax2.set_xlabel('True y_ratio')
    ax2.set_ylabel('Predicted exp(mu)')
    # 原始空间通常建议使用对数坐标轴观察，否则点会挤在角落
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.plot([y_true_raw.min(), y_true_raw.max()], [y_true_raw.min(), y_true_raw.max()], 'r--', alpha=0.8)

    plt.tight_layout()
    plt.show()

#############################################
##########################################
##########################################
if __name__ == "__main__":
    ## para
    batch_size = 32

    seed_everything(42)

    df0 = pd.read_csv("data_yellow/data_GorkhaSize/data_gorkha.csv")

    # 构建目标变量 (conditional size modeling)
    df = df0[df0["Landslide"] == 1].copy()
    # y = Ls_Area / Shape_Area
    df["y_ratio"] = df["Ls_Area"] / df["Shape_Area"]
    # Remove non-positive ratios (shouldn't exist, but safe)
    df = df[df["y_ratio"] > 0].copy()
    # z = log(y)
    df["z_logy"] = np.log(df["y_ratio"].values)


    # ---- Keep only needed columns,保留X和Y变量
    # Feature columns (edit these to match your dataset),包含土壤的深度值'Shape_Area',
    num_cols=[
        'Est_m', 'Est_s','Nrt_m', 'Nrt_s', 'HC_m', 'HC_s', 'VC_m', 'VC_s',
        'Ele_m', 'Ele_s', 'Slp_m', 'Slp_s', 'NDVI_m','NDVI_s', 'Prc_m', 'Prc_s',
        'Bdod_m', 'Bdod_s', 'Clay_m', 'Clay_s', 'Sand_m', 'Sand_s', 'Silt_m', 'Shape_Leng','Shape_Area',
         'PGV_Usgs'
    ]
    cat_cols= [
        "Lithology",
        # add more categorical features here...
    ]
    keep_cols = num_cols + cat_cols + ["y_ratio", "z_logy"]
    df = df[keep_cols].dropna().copy()


    # 训练测试集划分
    seed = 42
    df_trainval, df_test = train_test_split(df, test_size=0.2, random_state=seed)
    df_train, df_val = train_test_split(df_trainval, test_size=0.15, random_state=seed)
    df_test.to_csv("data_yellow/data_GorkhaSize/landslide_model_package/test_holdout.csv", index=False)

    # ---- Encode categoricals
    df_train, df_val, df_test, cat_maps = encode_categoricals(
        df_train, df_val, df_test, list(cat_cols)
    )
    cat_cardinalities = [cat_maps[c]["num_classes"] for c in cat_cols]

    # ---- Scale numeric using TRAIN only
    scaler = StandardScaler()
    Xtr_num = scaler.fit_transform(df_train[list(num_cols)].values.astype(np.float32))
    Xva_num = scaler.transform(df_val[list(num_cols)].values.astype(np.float32))
    Xte_num = scaler.transform(df_test[list(num_cols)].values.astype(np.float32))


    # ---- Categorical arrays
    Xtr_cat = df_train[list(cat_cols)].values.astype(np.int64) if cat_cols else np.zeros((len(df_train), 0), dtype=np.int64)
    Xva_cat = df_val[list(cat_cols)].values.astype(np.int64) if cat_cols else np.zeros((len(df_val), 0), dtype=np.int64)
    Xte_cat = df_test[list(cat_cols)].values.astype(np.int64) if cat_cols else np.zeros((len(df_test), 0), dtype=np.int64)

    # ---- Targets
    ytr = df_train["z_logy"].values.astype(np.float32)
    yva = df_val["z_logy"].values.astype(np.float32)
    yte = df_test["z_logy"].values.astype(np.float32)

    # ---- DataLoaders
    tr_ds = TabDataset(Xtr_num, Xtr_cat, ytr)
    va_ds = TabDataset(Xva_num, Xva_cat, yva)
    te_ds = TabDataset(Xte_num, Xte_cat, yte)

    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    va_loader = torch.utils.data.DataLoader(va_ds, batch_size=batch_size, shuffle=False)
    te_loader = torch.utils.data.DataLoader(te_ds, batch_size=batch_size, shuffle=False)



    # ---- Model
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    d_token = 32
    n_heads = 4
    n_layers = 3
    dropout = 0.15
    lr = 1e-3
    weight_decay = 1e-4
    early_stop_patience = 20

    model = FTTransformer(
        n_num=len(num_cols),
        cat_cardinalities=cat_cardinalities,
        d_token=d_token,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)

    best_val = float("inf")
    best_state = None
    patience = 0

    print(f"Device: {device}")
    print(f"Train n={len(tr_ds)}, Val n={len(va_ds)}, Test n={len(te_ds)}")

    # ---- Training with early stopping on Val NLL
    for epoch in range(1, 200 + 1):
        tr_loss = train_one_epoch(model, tr_loader, optimizer, device)
        val_metrics = evaluate(model, va_loader, device)

        val_nll = val_metrics["NLL"]
        improved = val_nll < best_val - 1e-5

        if improved:
            best_val = val_nll
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | TrainLoss={tr_loss:.4f} | "
                f"Val NLL={val_metrics['NLL']:.4f} | Val R_z={val_metrics['R_z']:.4f} | "
                f"Val Cov90={val_metrics['Coverage90_z']:.3f}"
            )

        if patience >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}. Best Val NLL={best_val:.4f}")
            break

    # ---- Load best and evaluate on Test
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, te_loader, device)
    print("\n=== Test Metrics (conditional on Landslide=1) ===")
    for k, v in test_metrics.items():
        print(f"{k:>12s}: {v:.6f}")

    ### PLOT The scatter
    model.eval()
    test_zs, test_mus = [], []
    with torch.no_grad():
        for x_num, x_cat, y_z in te_loader:
            mu, _ = model(x_num.to(device), x_cat.to(device))
            test_zs.append(y_z.numpy())
            test_mus.append(mu.cpu().numpy())

    y_true_test = np.vstack(test_zs).flatten()
    y_pred_test = np.vstack(test_mus).flatten()



    ###################################
    ## save the model
    #################
    import joblib
    import json
    import os
    # 创建保存文件夹
    save_dir = "data_yellow/data_GorkhaSize/landslide_model_package"
    os.makedirs(save_dir, exist_ok=True)

    # 1. 保存模型权重
    torch.save(best_state, os.path.join(save_dir, "model_weights.pt"))

    # 2. 保存数值标准化器 (StandardScaler)
    joblib.dump(scaler, os.path.join(save_dir, "num_scaler.joblib"))

    # 3. 保存模型配置参数 (用于重建网络结构)
    config_dict = {
        "num_cols": list(num_cols),
        "cat_cols": list(cat_cols),
        "cat_cardinalities": cat_cardinalities,
        "d_token": 32,
        "n_heads": 4,
        "n_layers": 3,
        "dropout": 0.15
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config_dict, f)

    # 4. 保存分类映射关系 (极其重要，确保分类索引不乱)
    joblib.dump(cat_maps, os.path.join(save_dir, "cat_maps.joblib"))

    print(f"所有模型组件已成功保存至: {save_dir}")


    # ---- 调用绘图 ----
    # plot_regression_results(y_true_test, y_pred_test)



    # print("\nNotes:")
    # print("- MAE_z / RMSE_z are errors on z=log(y_ratio).")
    # print("- MAE_y and TailMAE_y are on y_ratio (median prediction exp(mu)).")
    # print("- Coverage90_z should be ~0.90 if uncertainty is well-calibrated.")



    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # # 设置图形大小
    # plt.figure(figsize=(10, 6))
    # # 使用seaborn的kdeplot绘制密度曲线
    # sns.kdeplot(data=df, x="z_logy", fill=True, color="skyblue", alpha=0.6)
    # # 显示图形
    # plt.show()