import torch
import joblib
import json
import numpy as np
import pandas as pd
import os
from data_yellow.data_GorkhaSize.FTTransformer import FTTransformer, TabDataset, encode_categoricals
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device,
) -> Dict[str, float]:
    model.eval()
    zs, mus, sigmas = [], [], []
    nlls = []

    for x_num, x_cat, y_z in loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y_z = y_z.to(device)

        mu, sigma = model(x_num, x_cat)
        nll = (y_z - mu) ** 2 / (2 * sigma ** 2) + torch.log(sigma)

        zs.append(y_z.cpu().numpy())
        mus.append(mu.cpu().numpy())
        sigmas.append(sigma.cpu().numpy())
        nlls.append(nll.cpu().numpy())

    z = np.vstack(zs).reshape(-1)
    mu = np.vstack(mus).reshape(-1)
    sigma = np.vstack(sigmas).reshape(-1)
    nll = np.vstack(nlls).reshape(-1)

    mae_z = float(np.mean(np.abs(z - mu)))
    rmse_z = float(np.sqrt(np.mean((z - mu) ** 2)))
    r_z = pearsonr(np.array(z).ravel(), mu)[0]
    mean_nll = float(np.mean(nll))

    # 90% prediction interval coverage on z
    lower = mu - 1.645 * sigma
    upper = mu + 1.645 * sigma
    coverage_90 = float(np.mean((z >= lower) & (z <= upper)))

    # also compute MAE on original y_ratio (optional)
    y_true = np.exp(z)
    y_pred_med = np.exp(mu)  # median of lognormal
    mae_y = float(np.mean(np.abs(y_true - y_pred_med)))

    r_y = pearsonr(np.array(y_true).ravel(), y_pred_med)[0]

    # tail MAE on y (top 10% by true y)
    thr = np.quantile(y_true, 0.9)
    mask = y_true >= thr
    tail_mae_y = float(np.mean(np.abs(y_true[mask] - y_pred_med[mask]))) if np.any(mask) else float("nan")

    # 构造DataFrame
    results_df = pd.DataFrame({
        'trues': np.array(z).ravel(),
        'preds': mu,
        'type': 'FT-Transformer'

    })
    # 保存为CSV
    results_df.to_csv('./data_yellow/data_GorkhaSize/prediction_test_FTTransformer.csv', index=False, header=False, encoding='utf-8-sig')

    return {
        "MAE_z": mae_z,
        "RMSE_z": rmse_z,
        "NLL": mean_nll,
        "Coverage90_z": coverage_90,
        "MAE_y": mae_y,
        "TailMAE_y": tail_mae_y,
        "R_z": r_z,
        "R_y": r_y
    }

def run_inference(csv_path, model_dir="data_yellow/data_GorkhaSize/landslide_model_package"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1. 加载所有配置文件和工具 ---
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        cfg = json.load(f)

    scaler = joblib.load(os.path.join(model_dir, "num_scaler.joblib"))
    cat_maps = joblib.load(os.path.join(model_dir, "cat_maps.joblib"))

    # --- 2. 重建模型并加载权重 ---
    model = FTTransformer(
        n_num=len(cfg["num_cols"]),
        cat_cardinalities=cfg["cat_cardinalities"],
        d_token=cfg["d_token"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"]
    ).to(device)

    model.load_state_dict(torch.load(os.path.join(model_dir, "model_weights.pt"), map_location=device))
    model.eval()
    print("模型及权重加载成功。")

    # --- 3. 加载新数据并进行预处理 ---
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
    # Feature columns (edit these to match your dataset),包含土壤的深度值
    num_cols = [
        'Est_m', 'Est_s', 'Nrt_m', 'Nrt_s', 'HC_m', 'HC_s', 'VC_m', 'VC_s',
        'Ele_m', 'Ele_s', 'Slp_m', 'Slp_s', 'NDVI_m', 'NDVI_s', 'Prc_m', 'Prc_s',
        'Bdod_m', 'Bdod_s', 'Clay_m', 'Clay_s', 'Sand_m', 'Sand_s', 'Silt_m', 'Shape_Leng',
        'Shape_Area', 'PGV_Usgs'
    ]
    cat_cols = [
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
    Xtr_cat = df_train[list(cat_cols)].values.astype(np.int64) if cat_cols else np.zeros((len(df_train), 0),
                                                                                         dtype=np.int64)
    Xva_cat = df_val[list(cat_cols)].values.astype(np.int64) if cat_cols else np.zeros((len(df_val), 0), dtype=np.int64)
    Xte_cat = df_test[list(cat_cols)].values.astype(np.int64) if cat_cols else np.zeros((len(df_test), 0),
                                                                                        dtype=np.int64)

    # ---- Targets
    ytr = df_train["z_logy"].values.astype(np.float32)
    yva = df_val["z_logy"].values.astype(np.float32)
    yte = df_test["z_logy"].values.astype(np.float32)

    # ---- DataLoaders
    tr_ds = TabDataset(Xtr_num, Xtr_cat, ytr)
    va_ds = TabDataset(Xva_num, Xva_cat, yva)
    te_ds = TabDataset(Xte_num, Xte_cat, yte)
    batch_size = 32
    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    va_loader = torch.utils.data.DataLoader(va_ds, batch_size=batch_size, shuffle=False)
    te_loader = torch.utils.data.DataLoader(te_ds, batch_size=batch_size, shuffle=False)


    # 使用之前定义的 evaluate 函数
    results = evaluate(model, te_loader, device)

    print("\n--- 推理结果 ---")
    for k, v in results.items():
        print(f"{k}: {v:.6f}")

    return results


if __name__ == "__main__":
    # 使用你保存好的模型在任意数据(如测试集csv)上验证
    run_inference("data_yellow/data_GorkhaSize/landslide_model_package/test_holdout.csv")
    pass