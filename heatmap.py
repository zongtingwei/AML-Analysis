#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
module_heatmap_and_ranksum.py  (bi-clustered heatmaps + rank-sum tests)

功能概述
1) 读取：
   - --expr: 样本×特征表达矩阵（CSV；来自 Dx198_concat_tumor_only.csv）
   - --dx  : Dx198.v2.xlsx（至少含列 Biaoben, SubtypeMerge）
   - --module-score: 模块打分表（Excel/CSV；第一列为 Biaoben，其余为模块分数，如 HSC1/HSC2/...）
2) 数据规则：
   - -6.0 视为缺失：读入即转 NaN；聚类用表达矩阵在选完 top 基因后按“列中位数”补；整列仍缺用 -6.0 兜底
   - 先按样本对齐，再**剔除 SubtypeMerge 中样本数 < min_subtype_size（默认 3）** 的类别
3) 聚类（与前述一致）：
   - 表达矩阵 → 选 top 变异基因（MAD/Var） → 列中位数补缺 → PCA → UMAP → KMeans(k=7) 得到 G1..G7
4) 模块列选择：
   - 使用 7 个前缀家族：HSC, MPP, LMPP, GMP.Cycle, GMP.Mono, GMP.Neut, Early_GMP
   - 在模块表中以“前缀匹配（忽略大小写及非字母数字）”选列；样本与聚类样本取**交集**
5) 热图（全部模块合并为“一张图”）：
   A. **固定顺序**：
      - cluster-mean：行=G1..G7，列=所有选中模块；对每列按簇均值做 z-score
      - sample-level：行=样本（按簇排序），列=所有选中模块；对每列 z-score
   B. **行/列双向聚类（clustermap）**：
      - cluster-mean 与 sample-level 各一张
      - clustermap 前做 sanitize：丢全 NaN 列、按列中位数补残缺、替换 Inf/NaN→0，避免 SciPy 报错
      - 导出行/列聚类后的顺序 txt
6) **秩和检验**（Mann–Whitney U）：
   - 对每个簇 Gi：Gi vs 其余样本；对**所有选中模块**分别做检验
   - 输出 n_in, n_out, median_in/out, U, AUC=U/(n1*n2), p, q(FDR-BH)
   - 每簇一份 CSV，另有合并总表

可选加速：`pip install fastcluster`（Seaborn clustermap 会自动提速）
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from umap import UMAP
from scipy.stats import mannwhitneyu

# ----------------- Helpers -----------------
def norm_key(s: str) -> str:
    """lower + 去掉非字母数字，用于前缀匹配。"""
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())

def read_expr_csv(path: Path, missing_sentinel: float) -> pd.DataFrame:
    """读取表达矩阵 CSV；index=样本；-6.0→NaN；整表数值化。"""
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str).str.strip()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.mask(df == float(missing_sentinel))
    return df

def read_dx(path: Path) -> pd.DataFrame:
    """读取 Dx198.v2.xlsx；验证与规范化 Biaoben / SubtypeMerge。"""
    dx = pd.read_excel(path)
    assert 'Biaoben' in dx.columns and 'SubtypeMerge' in dx.columns, "Dx 需含 Biaoben, SubtypeMerge"
    dx['Biaoben'] = dx['Biaoben'].astype(str).str.strip()
    return dx.set_index('Biaoben')

def read_modules(path: Path, missing_sentinel: float) -> pd.DataFrame:
    """读取模块打分表（Excel/CSV）；第一列=Biaoben；-6.0→NaN；数值化。"""
    if str(path).lower().endswith(('.xlsx','.xls')):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    if 'Biaoben' not in df.columns:
        df.rename(columns={df.columns[0]:'Biaoben'}, inplace=True)
    df['Biaoben'] = df['Biaoben'].astype(str).str.strip()
    df.set_index('Biaoben', inplace=True)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.mask(df == float(missing_sentinel))
    return df

def select_family_columns(df: pd.DataFrame, families: List[str]) -> (List[str], Dict[str,str]):
    """在 df 中按家族前缀匹配列；返回列列表 + 列→家族映射（按家族顺序+列名排序）。"""
    fam_keys = [norm_key(f) for f in families]
    col2fam: Dict[str,str] = {}
    for c in df.columns:
        cn = norm_key(c)
        for f, fk in zip(families, fam_keys):
            if cn.startswith(fk):
                col2fam[c] = f
                break
    fam_pos = {f:i for i,f in enumerate(families)}
    cols = sorted(col2fam.keys(), key=lambda x: (fam_pos[col2fam[x]], x.lower()))
    return cols, col2fam

def impute_median(X: pd.DataFrame, fallback: float=-6.0) -> pd.DataFrame:
    """按列中位数补缺；整列仍缺用 fallback。"""
    med = X.median(axis=0, skipna=True)
    X2 = X.fillna(med)
    if X2.isna().any().any():
        X2 = X2.fillna(float(fallback))
    return X2

def top_genes(X: pd.DataFrame, method='MAD', n=700) -> pd.DataFrame:
    """MAD/Variance 选 top n 基因列。"""
    m = method.upper()
    if m == 'MAD':
        sc = (X - X.mean()).abs().mean()
    elif m in ('VAR', 'VARIANCE'):
        sc = X.var()
    else:
        raise ValueError("method 只支持 MAD/Variance")
    keep = sc.sort_values(ascending=False).head(min(n, X.shape[1])).index
    return X[keep]

def fit_umap(X: pd.DataFrame, pca_dims=20, n_neighbors=20, min_dist=0.1, metric='cosine', seed=2025):
    """标准化→PCA→UMAP。"""
    Z = (X - X.mean()) / (X.std() + 1e-8)
    p = max(2, min(pca_dims, Z.shape[1], Z.shape[0]-1))
    Xp = PCA(n_components=p, random_state=seed).fit_transform(Z.values)
    umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, n_components=2, random_state=seed)
    return umap.fit_transform(Xp)

def kmeans_groups(emb: np.ndarray, k: int, index: pd.Index, seed=2025) -> pd.Series:
    """UMAP 坐标上做 KMeans，按簇大小映射到 G1..Gk。"""
    km = KMeans(n_clusters=k, random_state=seed, n_init=20, max_iter=500)
    lab = km.fit_predict(emb)
    ser = pd.Series(lab, index=index)
    order = ser.value_counts().index.tolist()
    mp = {lab: f"G{i+1}" for i, lab in enumerate(order)}
    return ser.map(mp).rename("Group")

def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR。"""
    p = np.asarray(pvals, float)
    n = p.size
    if n == 0: return p
    order = np.argsort(p); ranks = np.arange(1, n+1)
    q = p[order] * n / ranks
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q); out[order] = np.clip(q, 0, 1)
    return out

def sanitize_for_clustermap(M: pd.DataFrame, outdir: Path, tag: str) -> pd.DataFrame:
    """clustermap 前清洗：丢全 NaN 列；剩余按列中位数补；Inf/NaN→0；记录丢弃列。"""
    M2 = M.copy()
    all_nan_cols = M2.columns[M2.isna().all(axis=0)]
    if len(all_nan_cols) > 0:
        (outdir / f"dropna_cols_{tag}.txt").write_text("\n".join(map(str, all_nan_cols)))
        M2 = M2.drop(columns=all_nan_cols)
    med = M2.median(axis=0, skipna=True)
    M2 = M2.fillna(med)
    M2 = M2.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return M2

def make_palettes(g_order: List[str], families: List[str]):
    """返回 组颜色映射 g2c 与 家族颜色映射 f2c。"""
    g_palette = sns.color_palette("tab20", n_colors=len(g_order))
    g2c = {g: g_palette[i] for i, g in enumerate(g_order)}
    f_palette = sns.color_palette("Set2", n_colors=len(families))
    f2c = {f: f_palette[i] for i, f in enumerate(families)}
    return g2c, f2c

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser(description="合并模块热图（行列双向聚类）+ 每簇秩和检验")
    ap.add_argument("--expr", required=True, type=Path)
    ap.add_argument("--dx", required=True, type=Path)
    ap.add_argument("--module-score", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)

    # 聚类配置
    ap.add_argument("--method", choices=["MAD","Variance"], default="MAD")
    ap.add_argument("--top", type=int, default=700)
    ap.add_argument("--pca-dims", type=int, default=20)
    ap.add_argument("--umap-n", type=int, default=20)
    ap.add_argument("--umap-min-dist", type=float, default=0.1)
    ap.add_argument("--umap-metric", type=str, default="cosine")
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--seed", type=int, default=2025)

    # 规则
    ap.add_argument("--missing-sentinel", type=float, default=-6.0)
    ap.add_argument("--min-subtype-size", type=int, default=3)

    # 家族列表
    ap.add_argument("--ct-families", type=str,
                    default="HSC,MPP,LMPP,GMP.Cycle,GMP.Mono,GMP.Neut,Early_GMP")

    # 是否输出 sample 级热图
    ap.add_argument("--no-sample-heatmap", action="store_true")

    # clustermap 参数
    ap.add_argument("--cm-metric", type=str, default="euclidean",
                    help="clustermap 距离度量（euclidean/correlation 等）")
    ap.add_argument("--cm-method", type=str, default="average",
                    help="clustermap 聚合方法（single/complete/average/ward 等）")

    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) 读取
    X_raw = read_expr_csv(args.expr, args.missing_sentinel)
    dx = read_dx(args.dx)
    modules_raw = read_modules(args.module_score, args.missing_sentinel)

    # 2) 对齐并去小体量 SubtypeMerge
    common = X_raw.index.intersection(dx.index)
    X_raw = X_raw.loc[common]; dx = dx.loc[common]
    cnt = dx['SubtypeMerge'].value_counts()
    keep_sub = cnt[cnt >= int(args.min_subtype_size)].index
    keep_mask = dx['SubtypeMerge'].isin(keep_sub)
    X_raw = X_raw.loc[keep_mask]; dx = dx.loc[keep_mask]

    # 3) 聚类得到 G1..Gk
    X_sel = top_genes(X_raw, method=args.method, n=args.top)
    X_for_clust = impute_median(X_sel, fallback=args.missing_sentinel)
    emb = fit_umap(X_for_clust, pca_dims=args.pca_dims,
                   n_neighbors=args.umap_n, min_dist=args.umap_min_dist,
                   metric=args.umap_metric, seed=args.seed)
    groups = kmeans_groups(emb, k=args.k, index=X_for_clust.index, seed=args.seed)
    g_order = sorted(groups.unique(), key=lambda s: int(s[1:]))
    groups.to_frame().to_csv(outdir / "groups_K7.csv")

    # 4) 模块列选择（七个家族）+ 数值化 + 与 groups 交集
    families = [f.strip() for f in args.ct_families.split(",") if f.strip()]
    modules_raw = modules_raw.loc[modules_raw.index.intersection(groups.index)]
    sel_cols, col2fam = select_family_columns(modules_raw, families)
    if len(sel_cols) == 0:
        raise RuntimeError("未匹配到任何模块列，请检查模块文件的列名与家族前缀。")
    pd.Series(col2fam).rename("family").to_csv(outdir / "col2family_mapping.csv")

    modules = modules_raw[sel_cols].copy()
    for c in modules.columns:
        modules[c] = pd.to_numeric(modules[c], errors='coerce')

    # 交集确保一致，避免 KeyError
    common_mg = modules.index.intersection(groups.index)
    if len(groups.index.difference(modules.index)) > 0:
        miss = list(groups.index.difference(modules.index))[:5]
        print(f"[warn] 有 {len(groups.index.difference(modules.index))} 个聚类样本缺少模块分数，例如: {miss} ...")
    modules = modules.loc[common_mg]
    groups  = groups.loc[common_mg]
    g_order = sorted(groups.unique(), key=lambda s: int(s[1:]))

    # 调色
    g2c, f2c = make_palettes(g_order, families)

    # 5A) 固定顺序：cluster-mean 单图热图
    fam_pos = {f:i for i,f in enumerate(families)}
    ordered_cols = sorted(sel_cols, key=lambda x: (fam_pos[col2fam[x]], x.lower()))
    df_plot = modules.join(groups)
    grp_mean = df_plot.groupby("Group")[ordered_cols].mean().reindex(g_order)
    grp_z = (grp_mean - grp_mean.mean(axis=0)) / (grp_mean.std(axis=0) + 1e-8)

    plt.figure(figsize=(max(12, 0.35*grp_z.shape[1]+4), 1.0*len(g_order)+3))
    sns.heatmap(grp_z, cmap="RdBu_r", center=0,
                cbar_kws={'label': 'Group mean (z-score per column)'})
    plt.title("All selected modules — cluster means (column z-scored)")
    plt.ylabel("Cluster (Group)")
    plt.xlabel("Modules (grouped by family)")
    # 家族分隔线及标签
    fam_ranges: Dict[str, List[int]] = {}
    for i, c in enumerate(ordered_cols):
        fam_ranges.setdefault(col2fam[c], []).append(i)
    for i, c in enumerate(ordered_cols[:-1]):
        if col2fam[c] != col2fam[ordered_cols[i+1]]:
            plt.axvline(i+1, color='black', lw=0.5)
    for f, idxs in fam_ranges.items():
        mid = (min(idxs) + max(idxs)) / 2.0 + 0.5
        plt.text(mid, -0.6, f, ha='center', va='top', fontsize=10)
    plt.tight_layout()
    plt.savefig(outdir / "heatmap_modules_all_cluster_mean.png", dpi=300, bbox_inches="tight")
    plt.savefig(outdir / "heatmap_modules_all_cluster_mean.pdf", bbox_inches="tight")
    plt.close()

    # 5B) 行列双向聚类：cluster-mean clustermap（sanitize）
    grp_z_cm = sanitize_for_clustermap(grp_z, outdir, tag="cluster_mean")
    row_colors_cm = pd.Series(index=grp_z_cm.index, data=[g2c[g] for g in grp_z_cm.index])
    cg = sns.clustermap(
        grp_z_cm,
        method=args.cm_method, metric=args.cm_metric,
        cmap="RdBu_r", center=0,
        row_cluster=True, col_cluster=True,
        row_colors=row_colors_cm,
        figsize=(max(12, 0.35*grp_z_cm.shape[1]+4), 1.0*len(grp_z_cm.index)+5),
        cbar_kws={'label': 'Group mean (z-score per column)'}
    )
    cg.ax_heatmap.set_title("Cluster means — bi-clustered")
    cg.ax_heatmap.set_xlabel("Modules"); cg.ax_heatmap.set_ylabel("Clusters")
    cg.savefig(outdir / "heatmap_modules_all_cluster_mean.clustermap.png", dpi=300, bbox_inches="tight")
    cg.savefig(outdir / "heatmap_modules_all_cluster_mean.clustermap.pdf", bbox_inches="tight")
    # 记录行/列顺序
    row_order = list(grp_z_cm.index[cg.dendrogram_row.reordered_ind])
    col_order = list(grp_z_cm.columns[cg.dendrogram_col.reordered_ind])
    Path(outdir / "clustermap_cluster_mean_row_order.txt").write_text("\n".join(row_order))
    Path(outdir / "clustermap_cluster_mean_col_order.txt").write_text("\n".join(col_order))
    plt.close(cg.fig)

    # 6A) 固定顺序：样本级单图热图（可关）
    if not args.no_sample_heatmap:
        samp = modules.reindex(groups.index)
        samp_z = (samp - samp.mean(axis=0)) / (samp.std(axis=0) + 1e-8)
        order_rows = groups.sort_values().index
        samp_z_fixed = samp_z.loc[order_rows, ordered_cols]
        plt.figure(figsize=(max(12, 0.35*samp_z_fixed.shape[1]+4), max(8, 0.05*samp_z_fixed.shape[0]+4)))
        sns.heatmap(samp_z_fixed, cmap="RdBu_r", center=0,
                    cbar_kws={'label': 'Sample value (z-score per column)'},
                    xticklabels=False, yticklabels=False)
        # 簇分隔线
        cum = 0
        for g in g_order:
            n = int((groups==g).sum())
            if cum > 0:
                plt.axhline(cum, color='black', lw=0.5)
            cum += n
        plt.title("All selected modules — samples (rows) ordered by cluster)")
        plt.xlabel("Modules (grouped by family)"); plt.ylabel("Samples (ordered by Group)")
        plt.tight_layout()
        plt.savefig(outdir / "heatmap_modules_all_samples.png", dpi=300, bbox_inches="tight")
        plt.savefig(outdir / "heatmap_modules_all_samples.pdf", bbox_inches="tight")
        plt.close()

    # 6B) 行列双向聚类：样本级 clustermap（sanitize）
    if not args.no_sample_heatmap:
        samp = modules.reindex(groups.index)
        samp_z = (samp - samp.mean(axis=0)) / (samp.std(axis=0) + 1e-8)
        samp_z_cm = sanitize_for_clustermap(samp_z, outdir, tag="samples")
        row_colors_samples = groups.loc[samp_z_cm.index].map(g2c)
        cg2 = sns.clustermap(
            samp_z_cm,
            method=args.cm_method, metric=args.cm_metric,
            cmap="RdBu_r", center=0,
            row_cluster=True, col_cluster=True,
            row_colors=row_colors_samples,
            xticklabels=False, yticklabels=False,
            figsize=(max(12, 0.35*samp_z_cm.shape[1]+4), max(8, 0.06*samp_z_cm.shape[0]+5)),
            cbar_kws={'label': 'Sample value (z-score per column)'}
        )
        cg2.ax_heatmap.set_title("Samples — bi-clustered")
        cg2.ax_heatmap.set_xlabel("Modules"); cg2.ax_heatmap.set_ylabel("Samples")
        cg2.savefig(outdir / "heatmap_modules_all_samples.clustermap.png", dpi=300, bbox_inches="tight")
        cg2.savefig(outdir / "heatmap_modules_all_samples.clustermap.pdf", bbox_inches="tight")
        # 记录行/列顺序
        row_order2 = list(samp_z_cm.index[cg2.dendrogram_row.reordered_ind])
        col_order2 = list(samp_z_cm.columns[cg2.dendrogram_col.reordered_ind])
        Path(outdir / "clustermap_samples_row_order.txt").write_text("\n".join(row_order2))
        Path(outdir / "clustermap_samples_col_order.txt").write_text("\n".join(col_order2))
        plt.close(cg2.fig)

    # 7) 秩和检验（每簇 vs 其余；列：所有选中模块）
    all_res = []
    ordered_cols_all = list(sel_cols)  # 检验覆盖所有选中列
    for g in g_order:
        in_mask = (groups == g)
        rows = []
        for col in ordered_cols_all:
            a = modules.loc[in_mask, col].dropna().values
            b = modules.loc[~in_mask, col].dropna().values
            n1, n2 = len(a), len(b)
            if n1 < 3 or n2 < 3:
                rows.append({'Group': g, 'feature': col, 'family': col2fam[col],
                             'n_in': n1, 'n_out': n2,
                             'median_in': np.nan if n1==0 else float(np.median(a)),
                             'median_out': np.nan if n2==0 else float(np.median(b)),
                             'AUC': np.nan, 'U': np.nan, 'p': np.nan, 'q': np.nan})
                continue
            try:
                U, p = mannwhitneyu(a, b, alternative='two-sided')
                auc = U / (n1 * n2)
            except Exception:
                U, p, auc = np.nan, np.nan, np.nan
            rows.append({'Group': g, 'feature': col, 'family': col2fam[col],
                         'n_in': n1, 'n_out': n2,
                         'median_in': float(np.median(a)),
                         'median_out': float(np.median(b)),
                         'AUC': float(auc) if auc==auc else np.nan,
                         'U': float(U) if U==U else np.nan,
                         'p': float(p) if p==p else np.nan})
        df_g = pd.DataFrame(rows)
        m = df_g['p'].notna()
        df_g.loc[m, 'q'] = fdr_bh(df_g.loc[m, 'p'].values)
        df_g.to_csv(outdir / f"ranksum_cluster_{g}.csv", index=False)
        all_res.append(df_g)

    pd.concat(all_res, ignore_index=True).to_csv(outdir / "ranksum_all_clusters.csv", index=False)

    print(f"[Done] 输出到：{outdir}")
    print(f"[Info] 使用样本数（模块∩聚类）：{len(groups)}；模块列数：{len(sel_cols)}；簇数：{len(g_order)}")

if __name__ == "__main__":
    main()




python module_heatmap_and_ranksum.py \
  --expr ./out_concat/Dx198_concat_tumor_only.csv \
  --dx ./Dx198.v2.xlsx \
  --module-score ./ALL.maglinant.module.score.forML.xlsx \
  --outdir ./out_tumoronly_K7_modules_allinone \
  --method MAD --top 700 \
  --pca-dims 20 \
  --umap-n 20 --umap-min-dist 0.1 --umap-metric cosine \
  --k 7 --seed 2025 \
  --missing-sentinel -6.0 \
  --min-subtype-size 3 \
  --ct-families 'HSC,MPP,LMPP,GMP.Cycle,GMP.Mono,GMP.Neut,Early_GMP' \
  --cm-metric euclidean --cm-method average

