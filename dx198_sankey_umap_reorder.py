#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dx198_sankey_umap_reorder.py

功能：
- 读取表达矩阵 (--expr) 与 Dx198.v2.xlsx (--dx, 至少含 Biaoben, SubtypeMerge，若含 NR/CR/Response 会自动统计 CR/NR 比例)
- 读取你已有的分组文件 (--groups)；若未提供，则在 UMAP 空间用 KMeans 做聚类（--k 指定 K）
- **将你点名的簇（--pin G3,G4,G6,G7）重命名并排到最前**：G3→G1, G4→G2, G6→G3, G7→G4；其余簇保持相对顺序，依次接到 G5...
- Sankey 底部 SubtypeMerge 的顺序也靠前放置：依次取这些优先簇的“主导 Subtype”并去重加到最前，剩余 subtype 按总体频数
- 输出：Sankey 图、UMAP 图（图例含 n 与 CR/NR 比例）、各种表格、marker/module 热图等

作者提醒：
- 表达矩阵支持两种格式：
  A) 行=样本、列=基因
  B) 含列 ['biaoben','merge_ct',<gene...>] —— 会先按 biaoben 聚合为样本×基因
"""

from __future__ import annotations
import argparse, warnings, re
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.feature_selection import f_classif

from umap import UMAP


# ---------------- I/O ----------------
def load_expression(expr_path: Path) -> pd.DataFrame:
    df_try = pd.read_csv(expr_path)
    if {'biaoben','merge_ct'}.issubset(df_try.columns):
        meta = {'Unnamed: 0','biaoben','merge_ct'}
        genes = [c for c in df_try.columns if c not in meta]
        df_try['biaoben'] = df_try['biaoben'].astype(str).str.strip()
        X = df_try.groupby('biaoben')[genes].mean()
        X.index.name = 'Sample'
        return X
    df = pd.read_csv(expr_path, index_col=0)
    idx = pd.Series(df.index.astype(str), index=df.index)
    sample_root = idx.str.extract(r'(^[A-Za-z0-9\-]+)')[0]
    df.index = sample_root
    X = df.groupby(df.index).mean()
    X.index.name = 'Sample'
    return X

def load_dx(dx_path: Path) -> Tuple[pd.Series, Optional[pd.Series]]:
    dx = pd.read_excel(dx_path)
    need = ['Biaoben','SubtypeMerge']
    for c in need:
        if c not in dx.columns:
            raise ValueError(f"{dx_path} 需包含列：{need}，缺少 {c}")
    dx['Biaoben'] = dx['Biaoben'].astype(str).str.strip()
    subtype = dx.set_index('Biaoben')['SubtypeMerge'].astype(str)

    # 识别 CR/NR
    resp = None
    for c in dx.columns:
        lc = c.lower()
        if any(k in lc for k in ['nr/cr','nrcr','response','remission','clinical_response']):
            r = dx.set_index('Biaoben')[c].astype(str).str.upper().str.strip()
            CR_like = r.str.contains(r'CR|CMR|MRD[- ]?NEG|COMPLETE', na=False)
            NR_like = r.str.contains(r'NR|PD|SD|NON[- ]?RESP', na=False)
            resp = pd.Series(index=r.index, dtype=object)
            resp[CR_like] = 'CR'
            resp[NR_like] = 'NR'
            resp = resp.dropna()
            break
    return subtype, resp

# ---------------- feature / UMAP / cluster ----------------
def top_genes(X: pd.DataFrame, method='MAD', n=700) -> pd.DataFrame:
    method = str(method).upper()
    if method == 'MAD':
        sc = (X - X.mean()).abs().mean()
    elif method in ('VAR','VARIANCE'):
        sc = X.var()
    else:
        raise ValueError("method 只支持 MAD / Variance")
    keep = sc.sort_values(ascending=False).head(min(n, X.shape[1])).index
    return X[keep]

def fit_umap(X: pd.DataFrame, pca_dims=20, n_neighbors=30, min_dist=0.2,
             metric='cosine', seed=2025) -> np.ndarray:
    Z = (X - X.mean()) / (X.std() + 1e-8)
    p = max(2, min(pca_dims, Z.shape[1], Z.shape[0]-1))
    Xp = PCA(n_components=p, random_state=seed).fit_transform(Z.values)
    umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric,
                n_components=2, random_state=seed)
    return umap.fit_transform(Xp)

def cluster_in_umap(emb: np.ndarray, k: int, seed=2025) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=seed, n_init=20, max_iter=500)
    return km.fit_predict(emb)

def labels_to_G(labels: np.ndarray, index: pd.Index) -> pd.Series:
    ser = pd.Series(labels, index=index)
    order = ser.value_counts().index.tolist()
    mp = {lab: f"G{i+1}" for i, lab in enumerate(order)}
    return ser.map(mp).rename('Group')

# ---------------- tables / Sankey ----------------
def build_counts(groups: pd.Series, subtype: pd.Series) -> pd.DataFrame:
    df = pd.concat([groups, subtype.rename('Subtype')], axis=1).dropna()
    return df.groupby(['Group','Subtype']).size().reset_index(name='Count')

def hungarian_order(counts: pd.DataFrame) -> Tuple[List[str], List[str]]:
    tab = counts.pivot_table(index='Group', columns='Subtype', values='Count', aggfunc='sum', fill_value=0)
    Gs, Fs = tab.index.tolist(), tab.columns.tolist()
    from scipy.optimize import linear_sum_assignment
    gi, fj = linear_sum_assignment(-tab.values.astype(float))
    g_order = [Gs[i] for i,_ in sorted(zip(gi,fj))]
    matched_fs = [Fs[j] for _,j in sorted(zip(gi,fj))]
    rest = [f for f in Fs if f not in matched_fs]
    rest = tab.sum(0).loc[rest].sort_values(ascending=False).index.tolist()
    f_order = matched_fs + rest
    return g_order, f_order

def relabel_with_priority(groups: pd.Series, g_order: List[str], priority_old: List[str]) -> Tuple[pd.Series, List[str]]:
    """把 priority_old 中的旧标签排到最前并重命名为 G1..；其他组按原 g_order 接到后面。"""
    pr = [g for g in priority_old if g in groups.unique()]
    if len(pr) < len(priority_old):
        miss = [g for g in priority_old if g not in groups.unique()]
        warnings.warn(f"这些优先组在当前分组中不存在，将忽略：{miss}")
    rest = [g for g in g_order if g not in pr]
    new_order = pr + rest
    mapping = {old: f"G{i+1}" for i, old in enumerate(new_order)}
    new_groups = groups.map(mapping)
    new_g_order = [mapping[g] for g in new_order]
    return new_groups, new_g_order, mapping

def bottom_order_with_priority(counts: pd.DataFrame, g_order: List[str], n_priority: int) -> List[str]:
    # 先取前 n_priority 个组的主导 subtype 依序放前，再接剩余
    top_sub = counts.sort_values("Count", ascending=False).groupby("Group").first()["Subtype"]
    front = []
    for g in g_order[:n_priority]:
        s = top_sub.get(g, None)
        if s is not None and s not in front:
            front.append(s)
    rest = [s for s in counts["Subtype"].unique().tolist() if s not in front]
    # 按总体频次排剩余
    rem_order = counts.groupby("Subtype")["Count"].sum().loc[rest].sort_values(ascending=False).index.tolist()
    return front + rem_order

def plot_sankey(counts: pd.DataFrame, out_png: Path, out_pdf: Path,
                g_order: List[str], f_order: List[str],
                title="Subgroups", label_y=-0.15, label_angle=60):
    matplotlib.rcParams.update({"font.family":"serif",
                                "font.serif":["Times New Roman","Times","Nimbus Roman No9 L","DejaVu Serif"]})
    df = counts.copy()
    df["Group"]  = pd.Categorical(df["Group"],  categories=g_order, ordered=True)
    df["Subtype"] = pd.Categorical(df["Subtype"], categories=f_order, ordered=True)
    df = df.sort_values(["Group","Subtype"]).reset_index(drop=True)

    N = df["Count"].sum()
    top_totals = df.groupby("Group")["Count"].sum().reindex(g_order).fillna(0)
    bot_totals = df.groupby("Subtype")["Count"].sum().reindex(f_order).fillna(0)

    gap=0.012; bar_h=0.06
    top_y0, top_y1 = 1.0-bar_h, 1.0
    bot_y0, bot_y1 = 0.0, bar_h
    left_margin=0.06; right_margin=0.06
    usable_width = 1.0 - left_margin - right_margin

    def spans(totals):
        widths = (totals/N) * (usable_width - gap*(len(totals)-1))
        x = left_margin; spans={}
        for name,w in zip(totals.index, widths):
            spans[name]=(x,x+w); x=x+w+gap
        return spans

    top_span = spans(top_totals)
    bot_span = spans(bot_totals)

    palette = ["#de7c6a","#79b5ad","#6ea0c7","#8f7b97","#6c4a3f",
               "#6a5a8c","#e7a3c4","#e4c48a","#7fb0d2","#b2d68b",
               "#d79ac9","#f2b38f","#c4c4c4","#9f8fbf","#a6dcef"]
    g_colors = {g: palette[i % len(palette)] for i,g in enumerate(g_order)}

    fig, ax = plt.subplots(figsize=(11,11))
    for g in g_order:
        x0,x1 = top_span[g]
        ax.add_patch(Rectangle((x0, top_y0), x1-x0, top_y1-top_y0,
                               facecolor=g_colors[g], edgecolor="white"))
        ax.text((x0+x1)/2, top_y1+0.03, g, ha="center", va="bottom", fontsize=16)

    for f in f_order:
        x0,x1 = bot_span[f]
        ax.add_patch(Rectangle((x0, bot_y0), x1-x0, bot_y1-bot_y0,
                               facecolor="#bfbfbf", edgecolor="white"))

    top_off = {k: top_span[k][0] for k in top_span}
    bot_off = {k: bot_span[k][0] for k in bot_span}
    Nw = (usable_width - gap*(len(top_totals)-1))
    for _, row in df.iterrows():
        g = str(row["Group"]); f = str(row["Subtype"]); c = float(row["Count"])
        w = (c/N)*Nw
        sx0 = top_off[g]; sx1 = sx0+w; top_off[g]=sx1
        tx0 = bot_off[f]; tx1 = tx0+w; bot_off[f]=tx1
        verts=[(sx0,top_y0),
               (sx0,top_y0-0.05),(tx0,bot_y1+0.05),(tx0,bot_y1),
               (tx1,bot_y1),
               (tx1,bot_y1+0.05),(sx1,top_y0-0.05),(sx1,top_y0),
               (sx0,top_y0)]
        codes=[MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,
               MplPath.LINETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MplPath(verts,codes),
                               facecolor=g_colors[g], alpha=0.75, edgecolor="white", linewidth=0.7))

    ax.set_xlim(0,1); ax.set_ylim(-0.65,1.12); ax.axis("off")
    for f in f_order:
        x0,x1 = bot_span[f]
        ax.text((x0+x1)/2, label_y, f, rotation=label_angle, ha="center", va="top",
                rotation_mode="anchor", fontsize=14, clip_on=False)
    ax.text(0.5, 1.09, title, ha="center", va="bottom", fontsize=20)
    ax.text(0.5, -0.45, "SubtypeMerge (Dx) classification", ha="center", va="top", fontsize=18)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

# ---------------- CR/NR & markers/modules ----------------
def summarize_response(groups: pd.Series, resp: Optional[pd.Series]) -> pd.DataFrame:
    df = pd.concat([groups, resp.rename('Resp')], axis=1).dropna()
    if df.empty:
        return pd.DataFrame(columns=['Group','CR','NR','Total','CR_rate','NR_rate'])
    tab = pd.crosstab(df['Group'], df['Resp']).rename_axis(None, axis=1)
    for c in ['CR','NR']:
        if c not in tab.columns: tab[c]=0
    tab = tab[['CR','NR']]
    tab['Total'] = tab.sum(1)
    tab['CR_rate'] = tab['CR']/tab['Total'].replace(0,np.nan)
    tab['NR_rate'] = tab['NR']/tab['Total'].replace(0,np.nan)
    return tab.reset_index().rename(columns={'index':'Group'})

def pick_markers_per_group(X: pd.DataFrame, groups: pd.Series, topN=40) -> pd.DataFrame:
    Z = (X - X.mean()) / (X.std()+1e-8)
    y = groups.loc[Z.index].astype('category').cat.codes.values
    F, _ = f_classif(Z.values, y)
    keep = pd.Series(F, index=Z.columns).sort_values(ascending=False).head(min(2000, Z.shape[1])).index
    Zk = Z[keep]
    out_rows = []
    for g in groups.unique():
        idx_in  = groups[groups==g].index
        idx_out = groups[groups!=g].index
        mu_in  = Zk.loc[idx_in].mean(0)
        mu_out = Zk.loc[idx_out].mean(0)
        sd_in  = Zk.loc[idx_in].std(0) + 1e-6
        sd_out = Zk.loc[idx_out].std(0) + 1e-6
        score = (mu_in - mu_out) / (sd_in + sd_out)
        top = score.sort_values(ascending=False).head(topN)
        for gene, s in top.items():
            out_rows.append({'Group':g, 'gene':gene, 'score':float(s)})
    return pd.DataFrame(out_rows)

def modules_from_markers(X: pd.DataFrame, markers: pd.DataFrame, groups: pd.Series, n_modules=6):
    Z = (X - X.mean()) / (X.std()+1e-8)
    genes = pd.Index(markers['gene'].unique()).intersection(Z.columns)
    if len(genes) == 0:
        return (pd.DataFrame(columns=['gene','module']),
                pd.DataFrame(index=Z.index),
                pd.DataFrame())
    Zm = Z[genes]
    C = np.corrcoef(Zm.T); C = np.nan_to_num(C, nan=0.0)
    D = 1 - C; D[D<0] = 0.0
    lab = AgglomerativeClustering(n_clusters=n_modules, metric='precomputed', linkage='average').fit_predict(D)
    module = pd.Series(lab, index=genes).map(lambda x: f"M{x+1}")
    module.name = 'module'
    scores_sample = pd.concat([Zm[list(v.index)].mean(1).rename(k)
                               for k,v in module.groupby(module)], axis=1)
    scores_group = scores_sample.groupby(groups).mean()
    return module.reset_index().rename(columns={'index':'gene'}), scores_sample, scores_group

# ---------------- optional composition ----------------
def load_composition(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None: return None
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    if {'biaoben','celltype'}.issubset(cols):
        d = {c.lower():c for c in df.columns}
        val_col = 'fraction' if 'fraction' in cols else ('count' if 'count' in cols else None)
        if val_col is None: raise ValueError("composition long 需包含 fraction 或 count")
        T = df.rename(columns=d).pivot_table(index='biaoben', columns='celltype',
                                             values=val_col, aggfunc='mean').fillna(0)
        T.index = T.index.astype(str)
        return T
    df.rename(columns={df.columns[0]:'biaoben'}, inplace=True)
    df['biaoben'] = df['biaoben'].astype(str)
    return df.set_index('biaoben')

def plot_composition_stacked(avg: pd.DataFrame, out_png: Path, out_pdf: Path):
    if avg.empty: return
    fig, ax = plt.subplots(figsize=(10,5))
    bottom = np.zeros(avg.shape[0]); x = np.arange(avg.shape[0])
    for ct in avg.columns:
        ax.bar(x, avg[ct].values, bottom=bottom, label=ct)
        bottom += avg[ct].values
    ax.set_xticks(x); ax.set_xticklabels(avg.index, rotation=0)
    ax.set_ylabel("Average fraction")
    ax.legend(bbox_to_anchor=(1.02,1), loc="upper left", frameon=False)
    ax.set_title("Cell-type composition by Group")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

# ---------------- main ----------------
def run(expr: Path, dx: Path, outdir: Path,
        method: str, top:int,
        pca_dims:int, umap_n:int, umap_min_dist:float, umap_metric:str,
        groups_path: Optional[Path], k: Optional[int],
        pin: List[str], label_y: float, angle: float,
        topN_markers:int, n_modules:int,
        composition_path: Optional[Path],
        seed:int):

    outdir.mkdir(parents=True, exist_ok=True)
    X_full = load_expression(expr)
    subtype, resp = load_dx(dx)

    common = X_full.index.intersection(subtype.index)
    X_full = X_full.loc[common]
    subtype = subtype.loc[common]
    if resp is not None:
        resp = resp.reindex(common).dropna()

    X = top_genes(X_full, method=method, n=top)
    emb = fit_umap(X, pca_dims=pca_dims, n_neighbors=umap_n,
                   min_dist=umap_min_dist, metric=umap_metric, seed=seed)

    # 使用现有分组或在 UMAP 上聚类
    if groups_path and Path(groups_path).exists():
        G0 = pd.read_csv(groups_path, index_col=0).iloc[:,0]
        G0.index = G0.index.astype(str)
        G0 = G0.reindex(X.index)
    else:
        if k is None:
            # 小范围自动选
            best = None
            for kk in [5,6,7]:
                lab = cluster_in_umap(emb, kk, seed=seed)
                sc = silhouette_score(emb, lab, metric='euclidean')
                if (best is None) or (sc>best[0]): best=(sc,kk)
            k = best[1]
        lab = cluster_in_umap(emb, k, seed=seed)
        G0 = labels_to_G(lab, X.index)

    # 初始 order（匈牙利法减少交叉）
    counts0 = build_counts(G0, subtype)
    g_order0, _ = hungarian_order(counts0)

    # ---- 关键步骤：按优先组重命名并靠前 ----
    G, g_order, mapping = relabel_with_priority(G0, g_order0, pin)

    # 重新汇总与底部顺序（优先组的主导 subtype 放前）
    counts = build_counts(G, subtype)
    f_order = bottom_order_with_priority(counts, g_order, n_priority=len([g for g in pin if g in mapping]))

    # 统计 CR/NR
    NRCR = summarize_response(G, resp)

    # 概要
    size = counts.groupby("Group")["Count"].sum().rename("size")
    topf = counts.sort_values("Count", ascending=False).groupby("Group").first()[["Subtype","Count"]]
    topf = topf.rename(columns={"Subtype":"top_subtype","Count":"top_count"})
    summary = topf.join(size)
    summary["purity"] = summary["top_count"]/summary["size"]
    if not NRCR.empty:
        summary = summary.join(NRCR.set_index("Group")[["CR_rate","NR_rate"]], how="left")
    summary = summary.loc[g_order]

    prefix = f"DX198_REORDER_nn{umap_n}_md{umap_min_dist}_{umap_metric}_{method}{top}"

    # 导出
    G.to_frame("Group").to_csv(outdir / f"{prefix}_groups.csv")
    pd.DataFrame(emb, index=X.index, columns=["UMAP1","UMAP2"]).to_csv(outdir / f"{prefix}_umap_coords.csv")
    counts.to_csv(outdir / f"{prefix}_group_fusion_counts.csv", index=False)
    counts.pivot(index="Group", columns="Subtype", values="Count").fillna(0).astype(int)\
          .to_csv(outdir / f"{prefix}_group_fusion_table.csv")
    summary.to_csv(outdir / f"{prefix}_group_summary.csv")
    if not NRCR.empty:
        NRCR.to_csv(outdir / f"{prefix}_NRCR_counts.csv", index=False)

    # Sankey
    plot_sankey(counts, out_png=outdir / f"{prefix}_sankey.png",
                out_pdf=outdir / f"{prefix}_sankey.pdf",
                g_order=g_order, f_order=f_order,
                title="Subgroups", label_y=label_y, label_angle=angle)

    # UMAP（图例带 n 与 CR/NR）
    colors = sns.color_palette("tab20", n_colors=len(g_order))
    g2c = {g:colors[i] for i,g in enumerate(g_order)}
    plt.figure(figsize=(9,7))
    for g in g_order:
        idx = (G==g)
        plt.scatter(emb[idx,0], emb[idx,1], s=40, color=g2c[g], label=g, alpha=0.9)
    legend_labels=[]
    for g in g_order:
        n = int((G==g).sum())
        if not NRCR.empty and g in set(NRCR['Group']):
            row = NRCR.set_index("Group").loc[g]
            cr = 0.0 if pd.isna(row['CR_rate']) else row['CR_rate']*100
            nr = 0.0 if pd.isna(row['NR_rate']) else row['NR_rate']*100
            legend_labels.append(f"{g} (n={n}, CR {cr:.0f}%, NR {nr:.0f}%)")
        else:
            legend_labels.append(f"{g} (n={n})")
    handles = [plt.Line2D([0],[0], marker='o', linestyle='', color=g2c[g], markersize=8) for g in g_order]
    plt.legend(handles, legend_labels, bbox_to_anchor=(1.02,1), loc="upper left", frameon=False)
    plt.xlabel("UMAP1"); plt.ylabel("UMAP2")
    plt.title(f"UMAP colored by Groups (K={len(g_order)})")
    plt.tight_layout()
    plt.savefig(outdir / f"{prefix}_umap.png", dpi=300, bbox_inches="tight")
    plt.close()

    # markers & modules
    markers = pick_markers_per_group(X, G, topN=topN_markers)
    markers.to_csv(outdir / f"{prefix}_marker_genes_per_group.csv", index=False)
    module_mem, ms_sample, ms_group = modules_from_markers(X, markers, G, n_modules=n_modules)
    module_mem.to_csv(outdir / f"{prefix}_module_membership.csv", index=False)
    ms_sample.to_csv(outdir / f"{prefix}_module_scores_sample.csv")
    ms_group.to_csv(outdir / f"{prefix}_module_scores_group.csv")
    if not ms_group.empty:
        plt.figure(figsize=(1.2*ms_group.shape[1]+3, 0.5*ms_group.shape[0]+3))
        sns.heatmap(ms_group.loc[g_order], cmap='RdBu_r', center=0, annot=True, fmt=".2f",
                    cbar_kws={'label':'Module score (z-mean)'})
        plt.title("Module scores (group mean)")
        plt.tight_layout()
        plt.savefig(outdir / f"{prefix}_module_heatmap.png", dpi=300)
        plt.savefig(outdir / f"{prefix}_module_heatmap.pdf")
        plt.close()

    # 可选细胞组成
    comp = load_composition(composition_path)
    if comp is not None:
        comp = comp.loc[comp.index.intersection(G.index)]
        comp = comp.div(comp.sum(1).replace(0,np.nan), axis=0).fillna(0)
        comp_avg = comp.join(G.rename('Group')).groupby('Group').mean().loc[g_order]
        comp_avg.to_csv(outdir / f"{prefix}_composition_group_avg.csv")
        plot_composition_stacked(comp_avg,
                                 out_png=outdir / f"{prefix}_composition_stackedbar.png",
                                 out_pdf=outdir / f"{prefix}_composition_stackedbar.pdf")

    print(f"[Done] 输出目录：{outdir}")
    print(f"[Info] 组顺序：{', '.join(g_order)}")
    print(f"[Info] 底部 Subtype 顺序（前几项来自优先组的主导 subtype）：{', '.join(f_order)}")

def parse_pin(s: str) -> List[str]:
    if s is None or len(s.strip())==0:
        return []
    return [x.strip() for x in s.split(',') if x.strip()]

def main():
    ap = argparse.ArgumentParser(description="Sankey/UMAP 报告（支持把指定簇重命名并排到最前）")
    ap.add_argument("--expr", required=True, type=Path)
    ap.add_argument("--dx", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)

    ap.add_argument("--method", choices=["MAD","Variance"], default="MAD")
    ap.add_argument("--top", type=int, default=700)
    ap.add_argument("--pca-dims", type=int, default=20)
    ap.add_argument("--umap-n", type=int, default=30)
    ap.add_argument("--umap-min-dist", type=float, default=0.2)
    ap.add_argument("--umap-metric", type=str, default="cosine")

    ap.add_argument("--groups", type=Path, default=None, help="已有分组文件(两列：Sample,Group)。若不给则根据 --k 在UMAP聚类")
    ap.add_argument("--k", type=int, default=None, help="UMAP空间KMeans的K（没传 groups 时才用）")

    ap.add_argument("--pin", type=str, default="G3,G4,G6,G7",
                    help="要靠前并重命名的旧组标签列表，逗号分隔（例如 G3,G4,G6,G7）")

    ap.add_argument("--label-y", type=float, default=-0.15)
    ap.add_argument("--angle", type=float, default=60.0)
    ap.add_argument("--topN-markers", type=int, default=40)
    ap.add_argument("--n-modules", type=int, default=6)
    ap.add_argument("--composition", type=Path, default=None, help="可选：样本×细胞类型表（宽表或 long 表）")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    run(args.expr, args.dx, args.outdir,
        method=args.method, top=args.top,
        pca_dims=args.pca_dims, umap_n=args.umap_n, umap_min_dist=args.umap_min_dist, umap_metric=args.umap_metric,
        groups_path=args.groups, k=args.k,
        pin=parse_pin(args.pin), label_y=args.label_y, angle=args.angle,
        topN_markers=args.topN_markers, n_modules=args.n_modules,
        composition_path=args.composition, seed=args.seed)

if __name__ == "__main__":
    main()

python dx198_sankey_umap_reorder.py \
  --expr AML.by_samle_celltype_mean_exp.csv \
  --dx Dx198.v2.xlsx \
  --outdir out_reorder_G3467 \
  --method MAD --top 700 \
  --pca-dims 20 \
  --umap-n 20 --umap-min-dist 0.1 --umap-metric cosine \
  --k 7 \
  --pin G3,G4,G6,G7 \
  --topN-markers 40 --n-modules 6 \
  --label-y -0.15 --angle 60

