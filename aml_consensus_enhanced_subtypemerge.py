#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced consensus clustering (paper-style) adapted to Dx198.v2.xlsx:
 - Bottom categories use **SubtypeMerge** (NOT Fusion-17)
 - Automatically **drops SubtypeMerge classes with < min-bottom-size** (default 3)
 - 20-method consensus (5 feature selectors × 4 clusterers) + PAC curve
 - Chooses K* by lowest PAC, but if multiple K within a tolerance of the best PAC,
   picks the one **closest to --target-k** (default 10) to meet "about 10 clusters"
 - Vertical Sankey: top = G1..Gn; bottom = SubtypeMerge (filtered)
 - Optional scRNA signature scoring (set,gene)

Outputs (under --outdir, prefix Dx198_CONSENSUS_k{K*}_SUBTM):
  1) *_groups.csv
  2) *_group_fusion_counts.csv           # long table (Group × SubtypeMerge)
  3) *_group_fusion_table.csv            # wide cross-tab
  4) *_group_summary.csv                 # size/top_subtype/purity/is_novel
  5) *_novel_members.csv
  6) *_samples_G*.csv
  7) *_consensus_matrix.csv
  8) *_pac_curve.png/pdf
  9) *_consensus_heatmap.pdf
 10) *_signature_scores.csv              # if --signatures provided
"""

import argparse, re, warnings
from pathlib import Path
from typing import Dict, Set, Optional, Tuple, List

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize


# -------------------- I/O --------------------
def load_expression(expr_path: Path) -> pd.DataFrame:
    """Return samples × genes matrix; support two formats."""
    df_try = pd.read_csv(expr_path)
    if 'biaoben' in df_try.columns:
        meta = {'Unnamed: 0','biaoben','merge_ct'}
        genes = [c for c in df_try.columns if c not in meta]
        df_try['biaoben'] = df_try['biaoben'].astype(str).str.strip()
        return df_try.groupby('biaoben')[genes].mean()
    df = pd.read_csv(expr_path, index_col=0)
    idx = pd.Series(df.index.astype(str), index=df.index)
    sample_root = idx.str.extract(r'(^[A-Za-z0-9\-]+)')[0]
    df.index = sample_root
    return df.groupby(df.index).mean()

def _clean_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def _drop_invalid_labels(s: pd.Series) -> pd.Series:
    s = _clean_text(s)
    bad = s.str.contains(r'(?i)unevaluated|unknown|unassigned|not\s*available|^na$|^nan$', na=False)
    return s[~bad]

def select_dx198_subtypemerge(X_full: pd.DataFrame, dx_path: Path, min_bottom_size: int=3) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Read Dx198.v2.xlsx and use column 'SubtypeMerge' as the bottom categories.
    - Aligns X to samples in Dx file.
    - Drops SubtypeMerge classes with < min_bottom_size samples.
    """
    dx = pd.read_excel(dx_path)
    need = ['Biaoben','SubtypeMerge']
    for col in need:
        if col not in dx.columns:
            raise ValueError(f"{dx_path} 需要包含列: {need}，缺少 {col}")
    dx['Biaoben'] = _clean_text(dx['Biaoben'])
    subtype_all = dx.set_index('Biaoben')['SubtypeMerge']
    subtype_all = _drop_invalid_labels(subtype_all)

    # Align to expression matrix
    X = X_full.loc[X_full.index.isin(subtype_all.index)].copy()
    X = X.groupby(X.index).mean()
    subtype = subtype_all.loc[subtype_all.index.intersection(X.index)].astype(str)
    X = X.loc[subtype.index]

    # Drop SubtypeMerge classes with < min_bottom_size
    counts = subtype.value_counts()
    keep_cats = counts[counts >= int(min_bottom_size)].index
    dropped = [c for c in counts.index if c not in keep_cats]
    if len(dropped) > 0:
        warnings.warn(f"以下 SubtypeMerge 类别样本数 < {min_bottom_size}，将被移除：{dropped}")
        mask = subtype.isin(keep_cats)
        X = X.loc[mask]
        subtype = subtype.loc[mask]

    return X, subtype


# -------------------- Feature selection --------------------
def fs_scores(X: pd.DataFrame, method: str) -> pd.Series:
    method = method.upper()
    if method == 'VAR':
        return X.var()
    if method == 'MAD':
        return (X - X.mean()).abs().mean()
    if method == 'CV':
        mu = X.mean().replace(0, np.nan)
        return (X.std() / mu).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if method == 'FANO':
        mu = X.mean().replace(0, np.nan)
        return (X.var() / mu).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if method == 'DISP':
        return X.var() - X.mean()
    raise ValueError(f"Unknown feature selector: {method}")

def select_features(X: pd.DataFrame, methods: Tuple[str, ...]=('VAR','MAD','CV','FANO','DISP'), top: int=1500) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for m in methods:
        sc = fs_scores(X, m).sort_values(ascending=False)
        out[m] = X[sc.head(min(top, X.shape[1])).index]
    return out


# -------------------- Clusterers --------------------
def cluster_kmeans(Xp: np.ndarray, k: int, rs: int) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=rs, n_init=10, max_iter=300)
    return km.fit_predict(Xp)

def cluster_spherical_kmeans(Xp: np.ndarray, k: int, rs: int) -> np.ndarray:
    Xn = normalize(Xp, norm='l2')
    km = KMeans(n_clusters=k, random_state=rs, n_init=10, max_iter=300)
    return km.fit_predict(Xn)

def cluster_agg_cosine(Xp: np.ndarray, k: int) -> np.ndarray:
    Xn = normalize(Xp, norm='l2')
    D = 1 - (Xn @ Xn.T)
    D[D<0] = 0.0
    lab = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average').fit_predict(D)
    return lab

def cluster_gmm(Xp: np.ndarray, k: int, rs: int) -> np.ndarray:
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=rs, max_iter=500, n_init=3)
    return gmm.fit_predict(Xp)


# -------------------- Consensus machinery --------------------
def consensus_one_combo(X: pd.DataFrame, k: int, repeats: int, subsample_samples: float, subsample_genes: float,
                        pca_dims: int, clusterer: str, rs: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(rs)
    n = X.shape[0]
    C = np.zeros((n,n), dtype=float)
    P = np.zeros((n,n), dtype=float)
    idx_all = np.arange(n)
    genes_all = np.arange(X.shape[1])

    for r in range(repeats):
        ss = max(2, int(round(subsample_samples * n)))
        sg = max(5, int(round(subsample_genes * X.shape[1])))
        idx = rng.choice(idx_all, size=ss, replace=False)
        gsel = rng.choice(genes_all, size=sg, replace=False)
        X_sub = X.iloc[idx, gsel]

        Xz = (X_sub - X_sub.mean(0)) / (X_sub.std(0) + 1e-8)
        p = max(2, min(pca_dims, Xz.shape[1], Xz.shape[0]-1))
        Xp = PCA(n_components=p, random_state=rs+r).fit_transform(Xz.values)

        if clusterer == 'kmeans':
            labels = cluster_kmeans(Xp, k, rs=rs+r)
        elif clusterer == 'skm':
            labels = cluster_spherical_kmeans(Xp, k, rs=rs+r)
        elif clusterer == 'aggcos':
            labels = cluster_agg_cosine(Xp, k)
        elif clusterer == 'gmm':
            labels = cluster_gmm(Xp, k, rs=rs+r)
        else:
            raise ValueError(f"Unknown clusterer {clusterer}")

        for lab in np.unique(labels):
            mem = idx[labels==lab]
            if len(mem) >= 2:
                C[np.ix_(mem,mem)] += 1.0
        P[np.ix_(idx,idx)] += 1.0

    return C, P

def pac_score(C: np.ndarray, P: np.ndarray, lower: float=0.1, upper: float=0.9) -> float:
    with np.errstate(divide='ignore', invalid='ignore'):
        M = np.divide(C, P, out=np.zeros_like(C), where=P>0)
    tri = M[np.triu_indices_from(M, k=1)]
    if tri.size == 0:
        return 1.0
    amb = ((tri > lower) & (tri < upper)).sum()
    return float(amb) / float(tri.size)


# -------------------- Utilities --------------------
def map_groups_by_size(labels: np.ndarray, index: pd.Index) -> pd.Series:
    ser = pd.Series(labels, index=index, name='cluster')
    order = ser.value_counts().index.tolist()
    ren = {lab: f"G{i+1}" for i, lab in enumerate(order)}
    return ser.map(ren).rename('Group')

def build_counts(groups: pd.Series, subtype: pd.Series) -> pd.DataFrame:
    df = pd.concat([groups, subtype.rename('SubtypeMerge')], axis=1).dropna()
    return df.groupby(['Group','SubtypeMerge']).size().reset_index(name='Count')

def natural_g_order(g_list: List[str]) -> List[str]:
    def num(s):
        m = re.findall(r'\d+', s)
        return int(m[0]) if m else 1
    return sorted(g_list, key=num)


# -------------------- Plot: PAC curve, heatmap, alluvial --------------------
def plot_pac_curve(ks, pac_mean, pac_sem, out_png, out_pdf):
    plt.figure(figsize=(6,4.2))
    plt.plot(ks, pac_mean, marker='o')
    if pac_sem is not None:
        plt.fill_between(ks, pac_mean-pac_sem, pac_mean+pac_sem, alpha=0.2)
    plt.xlabel('K'); plt.ylabel('PAC (lower=better)'); plt.title('PAC vs K')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()

def plot_consensus_heatmap(Cbar, labels, out_pdf, title="Consensus matrix (K*)"):
    D = 1 - Cbar
    lab = AgglomerativeClustering(n_clusters=len(np.unique(labels)), metric='precomputed', linkage='average').fit_predict(D)
    order = np.argsort(lab)
    plt.figure(figsize=(6,6))
    plt.imshow(Cbar[np.ix_(order,order)], vmin=0, vmax=1, cmap='viridis')
    plt.colorbar(label='Consensus')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()

def plot_alluvial(counts_df, title, out_png, out_pdf, g_order, f_order, label_y=-0.15, label_angle=60):
    matplotlib.rcParams.update({
        "font.family":"serif",
        "font.serif":["Times New Roman","Times","Nimbus Roman No9 L","DejaVu Serif"],
    })
    df = counts_df.copy()
    df["Group"]  = pd.Categorical(df["Group"],  categories=g_order, ordered=True)
    df["SubtypeMerge"] = pd.Categorical(df["SubtypeMerge"], categories=f_order, ordered=True)
    df = df.sort_values(["Group","SubtypeMerge"]).reset_index(drop=True)

    N = df["Count"].sum()
    top_totals = df.groupby("Group")["Count"].sum().reindex(g_order).fillna(0)
    bot_totals = df.groupby("SubtypeMerge")["Count"].sum().reindex(f_order).fillna(0)

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

    palette = ["#de7c6a","#79b5ad","#6ea0c7","#8f7b97","#6c4a3f","#6a5a8c","#e7a3c4","#e4c48a","#9dc3e6","#a8d08d"]
    g_colors = {g: palette[i%len(palette)] for i,g in enumerate(g_order)}

    fig, ax = plt.subplots(figsize=(11,11))

    # 顶部条：G1..Gn
    for g in g_order:
        x0,x1 = top_span[g]
        ax.add_patch(Rectangle((x0, top_y0), x1-x0, top_y1-top_y0, facecolor=g_colors[g], edgecolor="white"))
        ax.text((x0+x1)/2, top_y1+0.03, g, ha="center", va="bottom", fontsize=16)

    # 底部条（严格 SubtypeMerge）
    for f in f_order:
        x0,x1 = bot_span[f]
        ax.add_patch(Rectangle((x0, bot_y0), x1-x0, bot_y1-bot_y0, facecolor="#bfbfbf", edgecolor="white"))

    # 流线
    top_off = {k: top_span[k][0] for k in top_span}
    bot_off = {k: bot_span[k][0] for k in bot_span}
    Nw = (usable_width - gap*(len(top_totals)-1))
    for _, row in df.iterrows():
        g = str(row["Group"]); f = str(row["SubtypeMerge"]); c = float(row["Count"])
        w = (c/N)*Nw
        sx0 = top_off[g]; sx1 = sx0+w; top_off[g]=sx1
        tx0 = bot_off[f]; tx1 = tx0+w; bot_off[f]=tx1
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

    ax.text(0.5, 1.09, "Subgroups", ha="center", va="bottom", fontsize=20)
    ax.text(0.5, -0.45, "SubtypeMerge (Dx) classification", ha="center", va="top", fontsize=18)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# -------------------- Signature scoring --------------------
def load_signatures(path: Optional[Path]) -> Dict[str, Set[str]]:
    if path is None:
        return {}
    df = pd.read_csv(path, sep=None, engine='python')
    cols = {c.lower():c for c in df.columns}
    if 'set' not in cols or 'gene' not in cols:
        raise ValueError("Signature file needs columns: set,gene")
    out: Dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        s = str(row[cols['set']]).strip()
        g = str(row[cols['gene']]).strip()
        if s and g:
            out.setdefault(s, set()).add(g)
    return out

def score_signatures(X: pd.DataFrame, sigs: Dict[str, Set[str]]) -> pd.DataFrame:
    if not sigs:
        return pd.DataFrame(index=X.index)
    Z = (X - X.mean()) / (X.std() + 1e-8)
    scores = {}
    for s, genes in sigs.items():
        genes = [g for g in genes if g in Z.columns]
        if len(genes)==0:
            scores[s] = pd.Series(0.0, index=Z.index)
        else:
            scores[s] = Z[genes].mean(axis=1)
    return pd.DataFrame(scores)


# -------------------- Exports --------------------
def export_six_tables(df_merge: pd.DataFrame, counts: pd.DataFrame, g_order: List[str], f_order: List[str],
                      outdir: Path, pref: str, novel_purity: float, min_novel_size: int) -> None:
    # 1) groups
    groups = df_merge['Group']
    groups.index.name = "Sample"
    groups.to_frame("Group").to_csv(outdir / f"{pref}_groups.csv")
    # 2) long table
    counts.to_csv(outdir / f"{pref}_group_fusion_counts.csv", index=False)
    # 3) wide table
    xtab = counts.pivot(index="Group", columns="SubtypeMerge", values="Count").fillna(0).astype(int)
    xtab.to_csv(outdir / f"{pref}_group_fusion_table.csv")
    # 4) summary + Novel
    size = counts.groupby("Group")["Count"].sum().rename("size")
    top = counts.sort_values("Count", ascending=False).groupby("Group").first()[["SubtypeMerge","Count"]]
    top = top.rename(columns={"SubtypeMerge":"top_subtype","Count":"top_count"})
    second = counts.sort_values("Count", ascending=False).groupby("Group").nth(1)
    if second is None or second.empty:
        second = pd.DataFrame(columns=["SubtypeMerge","Count"])
    second = second[["SubtypeMerge","Count"]].rename(columns={"SubtypeMerge":"second_subtype","Count":"second_count"})
    summary = top.join(second, how="left").join(size)
    summary["purity"] = summary["top_count"] / summary["size"]
    summary["is_novel"] = (summary["purity"] < float(novel_purity)) & (summary["size"] >= int(min_novel_size))
    summary = summary.loc[natural_g_order(summary.index.tolist())]
    summary.to_csv(outdir / f"{pref}_group_summary.csv")
    # 5) novel members + 6) per-novel group
    novel_groups = summary[summary["is_novel"]].index.tolist()
    novel_members = df_merge[df_merge["Group"].isin(novel_groups)].reset_index().rename(columns={"index":"Sample"})
    novel_members.to_csv(outdir / f"{pref}_novel_members.csv", index=False)
    for g in novel_groups:
        sub = novel_members[novel_members["Group"] == g]
        sub.to_csv(outdir / f"{pref}_samples_{g}.csv", index=False)


# -------------------- K* chooser with target bias --------------------
def choose_k_star(ks: List[int], pac_mean: np.ndarray, target_k: int, pac_tol: float) -> int:
    """Pick K* by lowest PAC; if multiple within tolerance (relative), choose K closest to target_k."""
    pac_mean = np.asarray(pac_mean)
    best = pac_mean.min()
    # within relative tolerance OR absolute tolerance (0.01) safeguard
    candidates = [k for k, p in zip(ks, pac_mean) if (p <= best*(1.0+pac_tol)) or (p <= best + 0.01)]
    if candidates:
        return min(candidates, key=lambda k: (abs(k - target_k), k))  # tie-breaker: smaller K
    return ks[int(pac_mean.argmin())]


# -------------------- Main pipeline --------------------
def run_pipeline(expr: Path, dx: Path, outdir: Path,
                 kmin: int, kmax: int, repeats: int,
                 subsample_samples: float, subsample_genes: float,
                 feature_top: int, pca_dims: int,
                 pac_lower: float, pac_upper: float,
                 novel_purity: float, min_novel_size: int,
                 label_y: float, label_angle: float,
                 signatures_path: Optional[Path],
                 seed: int,
                 min_bottom_size: int,
                 target_k: int,
                 pac_tol: float) -> int:

    outdir.mkdir(parents=True, exist_ok=True)
    # load
    X_full = load_expression(expr)
    X, subtype = select_dx198_subtypemerge(X_full, dx, min_bottom_size=min_bottom_size)

    # feature sets
    fs_methods = ('VAR','MAD','CV','FANO','DISP')
    X_fs = select_features(X, methods=fs_methods, top=feature_top)

    # clustering algos
    clusterers = ('kmeans','skm','aggcos','gmm')

    # Per-K consensus across all 20 combos
    n = X.shape[0]
    pac_per_k = []
    Cbar_per_k = {}

    for k in range(kmin, kmax+1):
        C_accum = np.zeros((n,n)); P_accum = np.zeros((n,n))
        pac_this_k = []

        for fs_m in fs_methods:
            Xsel = X_fs[fs_m]
            for cl in clusterers:
                C, P = consensus_one_combo(Xsel, k, repeats=repeats,
                                           subsample_samples=subsample_samples,
                                           subsample_genes=subsample_genes,
                                           pca_dims=pca_dims,
                                           clusterer=cl, rs=seed + hash((k, fs_m, cl)) % (10**6))
                pac = pac_score(C, P, lower=pac_lower, upper=pac_upper)
                pac_this_k.append(pac)
                C_accum += C; P_accum += P

        with np.errstate(divide='ignore', invalid='ignore'):
            Cbar = np.divide(C_accum, P_accum, out=np.zeros_like(C_accum), where=P_accum>0)
        Cbar_per_k[k] = Cbar
        pac_mean = float(np.mean(pac_this_k)) if pac_this_k else 1.0
        pac_sem  = float(np.std(pac_this_k)/np.sqrt(len(pac_this_k))) if pac_this_k else 0.0
        pac_per_k.append((k, pac_mean, pac_sem))

    ks = [x[0] for x in pac_per_k]
    pac_mean = np.array([x[1] for x in pac_per_k])
    pac_sem  = np.array([x[2] for x in pac_per_k])

    # choose K* with target bias
    Kstar = choose_k_star(ks, pac_mean, target_k=target_k, pac_tol=pac_tol)

    # final labels from consensus at K*
    Cstar = Cbar_per_k[Kstar]
    Dstar = 1 - Cstar
    Dstar = np.nan_to_num(Dstar, nan=1.0, posinf=1.0, neginf=1.0)
    final_labels = AgglomerativeClustering(n_clusters=Kstar, metric='precomputed', linkage='average').fit_predict(Dstar)
    groups = map_groups_by_size(final_labels, X.index)
    g_order = natural_g_order(groups.unique().tolist())
    # bottom order strictly by SubtypeMerge counts (already filtered by min_bottom_size)
    f_order = subtype.value_counts().index.tolist()

    # export PAC plots
    pref = f"Dx198_CONSENSUS_k{Kstar}_SUBTM"
    pd.DataFrame(Cstar, index=X.index, columns=X.index).to_csv(outdir / f"{pref}_consensus_matrix.csv")
    plot_pac_curve(ks, pac_mean, pac_sem, out_png=outdir / f"{pref}_pac_curve.png", out_pdf=outdir / f"{pref}_pac_curve.pdf")
    plot_consensus_heatmap(Cstar, final_labels, out_pdf=outdir / f"{pref}_consensus_heatmap.pdf")

    # tables + alluvial
    df_merge = pd.concat([groups.rename('Group'), subtype.rename('SubtypeMerge')], axis=1).dropna()
    counts = build_counts(groups, subtype)
    export_six_tables(df_merge, counts, g_order, f_order, outdir, pref, novel_purity, min_novel_size)
    plot_alluvial(counts, "Subgroups", outdir / f"{pref}_alluvial.png", outdir / f"{pref}_alluvial.pdf",
                  g_order=g_order, f_order=f_order, label_y=label_y, label_angle=label_angle)

    # optional scRNA signature scoring
    try:
        sigs = load_signatures(signatures_path) if signatures_path else {}
        if sigs:
            scores = score_signatures(X, sigs)
            scores.to_csv(outdir / f"{pref}_signature_scores.csv")
    except Exception as e:
        warnings.warn(f"Signature scoring skipped: {e}")

    print(f"[Done] K* = {Kstar}. Outputs under: {outdir}")
    return Kstar


# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="Enhanced consensus clustering using SubtypeMerge bottom classes")
    ap.add_argument("--expr", required=True, type=Path)
    ap.add_argument("--dx", required=True, type=Path, help="Dx198.v2.xlsx (must contain Biaoben, SubtypeMerge)")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--kmin", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=14)
    ap.add_argument("--repeats", type=int, default=60, help="per combo bootstrap repeats")
    ap.add_argument("--subsample-samples", type=float, default=0.8)
    ap.add_argument("--subsample-genes", type=float, default=0.8)
    ap.add_argument("--feature-top", type=int, default=1500)
    ap.add_argument("--pca-dims", type=int, default=30)
    ap.add_argument("--pac-lower", type=float, default=0.1)
    ap.add_argument("--pac-upper", type=float, default=0.9)
    ap.add_argument("--novel-purity", type=float, default=0.70)
    ap.add_argument("--min-novel-size", type=int, default=5)
    ap.add_argument("--label-y", type=float, default=-0.15)
    ap.add_argument("--angle", type=float, default=60.0)
    ap.add_argument("--signatures", type=Path, default=None, help="TSV/CSV with columns: set,gene (optional)")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--min-bottom-size", type=int, default=3, help="drop SubtypeMerge classes with < this size")
    ap.add_argument("--target-k", type=int, default=10, help="prefer K near this if PAC ties")
    ap.add_argument("--pac-tol", type=float, default=0.05, help="relative tolerance for PAC tie region (e.g., 0.05=5%)")
    args = ap.parse_args()

    run_pipeline(args.expr, args.dx, args.outdir,
                 kmin=args.kmin, kmax=args.kmax, repeats=args.repeats,
                 subsample_samples=args.subsample_samples, subsample_genes=args.subsample_genes,
                 feature_top=args.feature_top, pca_dims=args.pca_dims,
                 pac_lower=args.pac_lower, pac_upper=args.pac_upper,
                 novel_purity=args.novel_purity, min_novel_size=args.min_novel_size,
                 label_y=args.label_y, label_angle=args.angle,
                 signatures_path=args.signatures, seed=args.seed,
                 min_bottom_size=args.min_bottom_size,
                 target_k=args.target_k, pac_tol=args.pac_tol)

if __name__ == "__main__":
    main()




python aml_consensus_enhanced_subtypemerge.py \
  --expr AML.by_samle_celltype_mean_exp.csv \
  --dx Dx198.v2.xlsx \
  --outdir out_consensus_subtm \
  --kmin 3 --kmax 14 \
  --repeats 80 \
  --subsample-samples 0.8 \
  --subsample-genes 0.8 \
  --feature-top 1000 \
  --pca-dims 30 \
  --label-y -0.15 --angle 60 \
  --novel-purity 0.7 --min-novel-size 5 \
  --min-bottom-size 3 \
  --target-k 10 --pac-tol 0.05
