#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import NearestCentroid

# -------------------- I/O & 预处理 --------------------
def load_expression(expr_path: Path) -> pd.DataFrame:
    """返回样本×基因矩阵；兼容长表与宽表两种输入。"""
    df_try = pd.read_csv(expr_path)
    if "biaoben" in df_try.columns:
        meta = {"Unnamed: 0", "biaoben", "merge_ct"}
        genes = [c for c in df_try.columns if c not in meta]
        df_try["biaoben"] = df_try["biaoben"].astype(str).str.strip()
        return df_try.groupby("biaoben")[genes].mean()
    df = pd.read_csv(expr_path, index_col=0)
    idx = pd.Series(df.index.astype(str), index=df.index)
    sample_root = idx.str.extract(r"(^[A-Za-z0-9\-]+)")[0]
    df.index = sample_root
    return df.groupby(df.index).mean()

def _drop_invalid_fusions(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    bad = s.str.contains(r"(?i)unevaluated|unknown|unassigned|not\s*available|^na$|^nan$", na=False)
    return s[~bad]

def select_dx198(X_full: pd.DataFrame, dx_path: Path):
    dx = pd.read_excel(dx_path)
    if "Biaoben" not in dx.columns or "Fusion gene subtyping" not in dx.columns:
        raise ValueError("Dx198.xlsx 需要包含 'Biaoben' 与 'Fusion gene subtyping'")
    dx["Biaoben"] = dx["Biaoben"].astype(str).str.strip()
    fusion_all = dx.set_index("Biaoben")["Fusion gene subtyping"]
    fusion17 = _drop_invalid_fusions(fusion_all)

    X = X_full.loc[X_full.index.isin(fusion17.index)].copy()
    X = X.groupby(X.index).mean()
    fusion17 = fusion17.loc[fusion17.index.intersection(X.index)].astype(str)
    X = X.loc[fusion17.index]
    return X, fusion17

def select_features(X: pd.DataFrame, method="MAD", top=1500):
    method = method.upper()
    if method == "MAD":
        scores = (X - X.mean()).abs().mean()
    elif method in ("VARIANCE","VAR"):
        scores = X.var()
    else:
        raise ValueError("method 必须为 MAD 或 Variance")
    keep = scores.sort_values(ascending=False).head(min(top, X.shape[1])).index
    return X[keep]

# -------------------- 全局聚类（OPTICS + DPGMM 回退） --------------------
def cluster_optics_global(X: pd.DataFrame, pca_dims=30, min_samples=5, min_cluster_size=8, xi=0.05, rs=2025):
    # 标准化 + PCA
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-8)
    p = max(2, min(pca_dims, X.shape[1], X.shape[0]-1))
    Xp = PCA(n_components=p, random_state=rs).fit_transform(Xz.values)

    # OPTICS 自动抽取簇；-1 为噪声
    optics = OPTICS(min_samples=min_samples, min_cluster_size=min_cluster_size,
                    metric='euclidean', cluster_method='xi', xi=xi)
    labels = optics.fit_predict(Xp)

    valid = [c for c in np.unique(labels) if c != -1]
    if len(valid) < 2:
        # 回退到 Dirichlet 过程高斯混合
        ncomp = min(20, max(2, Xp.shape[0]-1))
        bgm = BayesianGaussianMixture(
            n_components=ncomp,
            weight_concentration_prior_type='dirichlet_process',
            covariance_type='full',
            random_state=rs, n_init=5, max_iter=500
        )
        labels = bgm.fit_predict(Xp)

    # 将噪声样本分配到最近簇
    if np.any(labels == -1):
        nc = NearestCentroid()
        nc.fit(Xp[labels!=-1], labels[labels!=-1])
        nz = np.where(labels==-1)[0]
        if len(nz)>0:
            labels[nz] = nc.predict(Xp[nz])

    # 映射为 G1..Gn（按簇规模降序），并确保自然序排序
    order = pd.Series(labels).value_counts().index.tolist()
    lab2g = {lab: f"G{i+1}" for i, lab in enumerate(order)}
    groups = pd.Series([lab2g[l] for l in labels], index=X.index, name='Group')
    return groups, Xp

# -------------------- 绘图（纵向 alluvial） --------------------
def plot_alluvial(counts_df, out_png, out_pdf, g_order, f_order, label_y=-0.15, label_angle=60, title="Subgroups"):
    matplotlib.rcParams.update({
        "font.family":"serif",
        "font.serif":["Times New Roman","Times","Nimbus Roman No9 L","DejaVu Serif"],
    })
    df = counts_df.copy()
    df["Group"]  = pd.Categorical(df["Group"],  categories=g_order, ordered=True)
    df["Fusion"] = pd.Categorical(df["Fusion"], categories=f_order, ordered=True)
    df = df.sort_values(["Group","Fusion"]).reset_index(drop=True)

    N = df["Count"].sum()
    top_totals = df.groupby("Group")["Count"].sum().reindex(g_order).fillna(0)
    bot_totals = df.groupby("Fusion")["Count"].sum().reindex(f_order).fillna(0)

    gap=0.012; bar_h=0.06
    top_y0, top_y1 = 1.0-bar_h, 1.0
    bot_y0, bot_y1 = 0.0, bar_h
    left_margin=0.06; right_margin=0.06
    usable_width = 1.0 - left_margin - right_margin

    def spans_by_total(totals):
        widths = (totals/N) * (usable_width - gap*(len(totals)-1))
        x = left_margin; spans={}
        for name, w in zip(totals.index, widths):
            spans[name] = (x, x+w); x = x+w+gap
        return spans

    top_span = spans_by_total(top_totals)
    bot_span = spans_by_total(bot_totals)

    palette = ["#de7c6a","#79b5ad","#6ea0c7","#8f7b97","#6c4a3f","#6a5a8c","#e7a3c4","#e4c48a"]
    g_colors = {g: palette[i % len(palette)] for i,g in enumerate(g_order)}

    fig, ax = plt.subplots(figsize=(11,11))

    # 顶部条（G1..Gn）
    for g in g_order:
        x0,x1 = top_span[g]
        ax.add_patch(Rectangle((x0, top_y0), x1-x0, top_y1-top_y0, facecolor=g_colors[g], edgecolor="white"))
        ax.text((x0+x1)/2, top_y1+0.03, g, ha="center", va="bottom", fontsize=16)

    # 底部条（严格 17 类）
    for f in f_order:
        x0,x1 = bot_span[f]
        ax.add_patch(Rectangle((x0, bot_y0), x1-x0, bot_y1-bot_y0, facecolor="#bfbfbf", edgecolor="white"))

    # 流线
    top_off = {k: top_span[k][0] for k in top_span}
    bot_off = {k: bot_span[k][0] for k in bot_span}
    Nw = (usable_width - gap*(len(top_totals)-1))
    for _, row in df.iterrows():
        g = str(row["Group"]); f = str(row["Fusion"]); c = float(row["Count"])
        w = (c/N)*Nw
        sx0 = top_off[g]; sx1 = sx0 + w; top_off[g] = sx1
        tx0 = bot_off[f]; tx1 = tx0 + w; bot_off[f] = tx1
        verts = [(sx0,top_y0),
                 (sx0,top_y0-0.05),(tx0,bot_y1+0.05),(tx0,bot_y1),
                 (tx1,bot_y1),
                 (tx1,bot_y1+0.05),(sx1,top_y0-0.05),(sx1,top_y0),
                 (sx0,top_y0)]
        codes = [MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,
                 MplPath.LINETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MplPath(verts,codes), facecolor=g_colors[g], alpha=0.75, edgecolor="white", linewidth=0.7))

    ax.set_xlim(0,1); ax.set_ylim(-0.65,1.12); ax.axis("off")
    for f in f_order:
        x0,x1 = bot_span[f]
        ax.text((x0+x1)/2, label_y, f, rotation=label_angle, ha="center", va="top",
                rotation_mode="anchor", fontsize=14, clip_on=False)

    ax.text(0.5, 1.09, title, ha="center", va="bottom", fontsize=20)
    ax.text(0.5, -0.45, "WHO classification (Fusion gene subtyping)", ha="center", va="top", fontsize=18)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

# -------------------- 主流程 --------------------
def run_pipeline(expr, dx, outdir, method,
                 min_samples, min_cluster_size, xi, pca_dims,
                 novel_purity, min_novel_size,
                 label_y, label_angle, seed):

    outdir.mkdir(parents=True, exist_ok=True)
    X_full = load_expression(expr)
    X, fusion17 = select_dx198(X_full, dx)
    X_sel = select_features(X, method=method, top=1500)

    # —— 全局聚类
    groups, Xp = cluster_optics_global(X_sel, pca_dims=pca_dims,
                                       min_samples=min_samples, min_cluster_size=min_cluster_size,
                                       xi=xi, rs=seed)
    df_merge = pd.concat([groups.rename("Group"), fusion17.rename("Fusion")], axis=1).dropna()

    # 顶部顺序：自然序 G1..Gn；底部顺序：按样本数降序（严格 17 类）
    g_order = sorted(groups.unique().tolist(), key=lambda s: int(re.sub(r"\D+","",s)))
    f_order = fusion17.value_counts().index.tolist()

    # 计数与交叉表
    counts = df_merge.groupby(["Group","Fusion"]).size().reset_index(name="Count")
    xtab = counts.pivot(index="Group", columns="Fusion", values="Count").fillna(0).astype(int)

    # 摘要与 Novel 判定
    size = counts.groupby("Group")["Count"].sum().rename("size")
    top = counts.sort_values("Count", ascending=False).groupby("Group").first()[["Fusion","Count"]]
    top = top.rename(columns={"Fusion":"top_fusion","Count":"top_count"})
    second = counts.sort_values("Count", ascending=False).groupby("Group").nth(1)
    if second is None or second.empty:
        second = pd.DataFrame(columns=["Fusion","Count"])
    second = second[["Fusion","Count"]].rename(columns={"Fusion":"second_fusion","Count":"second_count"})
    summary = top.join(second, how="left").join(size)
    summary["purity"] = summary["top_count"] / summary["size"]
    summary["is_novel"] = (summary["purity"] < float(novel_purity)) & (summary["size"] >= int(min_novel_size))
    summary = summary.loc[g_order]  # 按自然序输出

    # —— 输出
    pref = f"Dx198_{method.upper()}_NOVEL_optics_ms{min_samples}_mcs{min_cluster_size}_xi{xi}"
    groups.to_frame("Group").to_csv(outdir / f"{pref}_groups.csv")                          # 1
    counts.to_csv(outdir / f"{pref}_group_fusion_counts.csv", index=False)                  # 2
    xtab.to_csv(outdir / f"{pref}_group_fusion_table.csv")                                  # 3
    summary.to_csv(outdir / f"{pref}_group_summary.csv")                                    # 4
    novel_groups = summary[summary["is_novel"]].index.tolist()
    novel_members = df_merge[df_merge["Group"].isin(novel_groups)].reset_index().rename(columns={"index":"Sample"})
    novel_members.to_csv(outdir / f"{pref}_novel_members.csv", index=False)                 # 5
    for g in novel_groups:                                                                  # 6
        sub = novel_members[novel_members["Group"] == g]
        sub.to_csv(outdir / f"{pref}_samples_{g}.csv", index=False)

    # —— 绘图
    plot_alluvial(counts,
                  out_png=outdir / f"{pref}_alluvial.png",
                  out_pdf =outdir / f"{pref}_alluvial.pdf",
                  g_order=g_order, f_order=f_order,
                  label_y=label_y, label_angle=label_angle, title="Subgroups")

# -------------------- CLI --------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dx198：OPTICS 全局聚类 + 纵向 Alluvial + 新类导出")
    ap.add_argument("--expr", required=True, type=Path)
    ap.add_argument("--dx", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--method", type=str, default="MAD", choices=["MAD","Variance"], help="特征选择方法")
    # OPTICS 参数
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--min-cluster-size", type=int, default=8)
    ap.add_argument("--xi", type=float, default=0.05, help="OPTICS 的 xi（越小越细）")
    ap.add_argument("--pca-dims", type=int, default=30)
    # Novel 判定
    ap.add_argument("--novel-purity", type=float, default=0.70, help="判定 Novel 的 purity 阈值")
    ap.add_argument("--min-novel-size", type=int, default=5, help="Novel 组的最小规模阈值")
    # 绘图
    ap.add_argument("--label-y", type=float, default=-0.15)
    ap.add_argument("--angle", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    run_pipeline(args.expr, args.dx, args.outdir, args.method,
                 min_samples=args.min_samples, min_cluster_size=args.min_cluster_size, xi=args.xi, pca_dims=args.pca_dims,
                 novel_purity=args.novel_purity, min_novel_size=args.min_novel_size,
                 label_y=args.label_y, label_angle=args.angle, seed=args.seed)
