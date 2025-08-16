#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment

# ================= 基础工具 =================
def load_expression(expr_path: Path) -> pd.DataFrame:
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

def _drop_invalid_fusions(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    bad = s.str.contains(r'(?i)unevaluated|unknown|unassigned|not\s*available|^na$|^nan$', na=False)
    return s[~bad]

def select_dx198(X_full: pd.DataFrame, dx_path: Path):
    dx = pd.read_excel(dx_path)
    if 'Biaoben' not in dx.columns or 'Fusion gene subtyping' not in dx.columns:
        raise ValueError("Dx198.xlsx 需要包含 'Biaoben' 与 'Fusion gene subtyping' 两列")
    dx['Biaoben'] = dx['Biaoben'].astype(str).str.strip()
    fusion_all = dx.set_index('Biaoben')['Fusion gene subtyping']
    fusion = _drop_invalid_fusions(fusion_all)

    X = X_full.loc[X_full.index.isin(fusion.index)].copy()
    X = X.groupby(X.index).mean()
    fusion = fusion.loc[fusion.index.intersection(X.index)].astype(str)
    X = X.loc[fusion.index]
    return X, fusion

def top_features(X: pd.DataFrame, method='MAD', n=1500):
    method = method.upper()
    if method=='MAD':
        scores = (X - X.mean()).abs().mean()
    elif method in ('VARIANCE','VAR'):
        scores = X.var()
    else:
        raise ValueError("method 必须是 'MAD' 或 'Variance'")
    return X[scores.sort_values(ascending=False).head(min(n, X.shape[1])).index]

def choose_k_by_silhouette(X, kmin=3, kmax=10, pca_dims=15, rs=0):
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-8)
    p = min(pca_dims, X.shape[1], X.shape[0])
    Xp = PCA(n_components=p, random_state=rs).fit_transform(Xz.values)
    best = None
    for k in range(kmin, kmax+1):
        km = MiniBatchKMeans(n_clusters=k, random_state=rs+k, batch_size=128, n_init=5, max_iter=200)
        lbl = km.fit_predict(Xp)
        sc = silhouette_score(Xp, lbl, metric='euclidean')
        if (best is None) or (sc > best[0]): best = (sc, k)
    return best[1]

def consensus_labels(X, k, iters=50, subsample=0.6, pca_dims=15, rs=0):
    rng = np.random.default_rng(rs)
    n = X.shape[0]
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-8)
    p = min(pca_dims, X.shape[1], X.shape[0])
    Xp = PCA(n_components=p, random_state=rs).fit_transform(Xz.values)
    idx_all = np.arange(n)
    C = np.zeros((n,n)); P = np.zeros((n,n))
    for t in range(iters):
        idx = rng.choice(idx_all, size=max(2, int(subsample*n)), replace=False)
        km = MiniBatchKMeans(n_clusters=k, random_state=rs+t, batch_size=128, n_init=5, max_iter=200)
        lab = km.fit_predict(Xp[idx])
        for lab_id in range(k):
            mem = idx[lab==lab_id]
            if len(mem)>=2:
                C[np.ix_(mem,mem)] += 1.0
        P[np.ix_(idx,idx)] += 1.0
    with np.errstate(divide='ignore', invalid='ignore'):
        C = np.divide(C, P, out=np.zeros_like(C), where=P>0); C[np.isnan(C)] = 0.0
    D = 1.0 - C; D[np.isnan(D)] = 1.0
    # 新版 sklearn 用 metric 参数
    lbl = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average').fit_predict(D)
    return lbl, Xp

def map_groups(labels, index):
    ser = pd.Series(labels, index=index, name='cluster')
    order = ser.value_counts().index.tolist()
    mp = {lab:f"G{i+1}" for i,lab in enumerate(order)}
    return ser.map(mp).rename('Group')

def build_counts(groups, fusion):
    df = pd.concat([groups, fusion.rename('Fusion')], axis=1).dropna()
    return df.groupby(['Group','Fusion']).size().reset_index(name='Count')

def natural_g_order(g_list):
    return sorted(g_list, key=lambda s: int(re.findall(r'\d+', s)[0]) if re.findall(r'\d+', s) else 1)

# ================= 方案A：匈牙利对齐 =================
def hungarian_align(merge_df):
    tab = merge_df.groupby(['Group','Fusion']).size().unstack(fill_value=0)
    Gs, Fs = tab.index.tolist(), tab.columns.tolist()
    cost = -tab.values.astype(float)
    gi, fj = linear_sum_assignment(cost)
    mapping = {Gs[i]: Fs[j] for i,j in zip(gi,fj)}
    purity = []
    for g in Gs:
        total = tab.loc[g].sum()
        mf = mapping.get(g)
        hit = tab.loc[g,mf] if mf in tab.columns else 0
        purity.append({'Group':g,'matched_fusion':mf,'purity': (hit/total if total>0 else 0.0),'n':int(total)})
    g_order = [Gs[i] for i,_ in sorted(zip(gi,fj))]
    matched_fs = [mapping[g] for g in g_order]
    remaining = [f for f in Fs if f not in matched_fs]
    remaining = (tab.sum(0).loc[remaining].sort_values(ascending=False).index.tolist())
    f_order = matched_fs + remaining
    return mapping, pd.DataFrame(purity), g_order, f_order

# ================= 方案B：微量重分配 =================
def micro_reassign(groups, fusion, mapping, Xp, distance_margin=0.05, max_iter=1):
    idx = groups.index
    moved = []
    for _ in range(max_iter):
        centroids = {g: Xp[(groups.values==g)].mean(axis=0) for g in groups.unique()}
        inv = {v:k for k,v in mapping.items()}
        changed = False
        for i,s in enumerate(idx):
            g_cur = groups.loc[s]; f = fusion.loc[s]; g_tar = inv.get(f)
            if g_tar is None or g_tar==g_cur: continue
            d_cur = np.linalg.norm(Xp[i]-centroids[g_cur])
            d_tar = np.linalg.norm(Xp[i]-centroids[g_tar])
            if d_tar < d_cur*(1-distance_margin):
                groups.loc[s] = g_tar
                moved.append({'sample':s,'from':g_cur,'to':g_tar,'fusion':f,'d_cur':float(d_cur),'d_tar':float(d_tar)})
                changed = True
        if not changed: break
        mapping, _, _, _ = hungarian_align(pd.concat([groups.rename('Group'), fusion.rename('Fusion')],axis=1))
    return groups, mapping, pd.DataFrame(moved)

# ================= 方案C：引导聚类（你之前的 guided） =================
MACRO_MAP = {
    'PML-RARA':'PML::RARA',
    'CBFB-MYH11':'CBFB::MYH11',
    'RUNX1-RUNX1T1':'RUNX1::RUNX1T1',
    'CEBPA bZip':'CEBPA',
    'MLL rearragement':'KMT2A fusions',
    'NUP98 rearragement':'NUP98 fusions',
    'Mutated NPM1 with FLT3-ITD':'NPM1',
    'Mutated NPM1 without FLT3-ITD':'NPM1',
    'AML-MR':'Myelodysplasia-related',
    'TP53 mutated':'Myelodysplasia-related',
    'GATA2-MECOM':'Differentiation entities',
    'DEK-CAN':'Differentiation entities',
    'EVI1 rearragement':'Differentiation entities',
    'TLS-ERG':'Differentiation entities',
    'Wild NPM1 with FLT3-ITD':'Differentiation entities',
    'Negative':'Differentiation entities',
    'Unevaluated':'Differentiation entities',
}

def guided_kmeans_groups(X, fusion_17, pca_dims=15, rs=0):
    macro = fusion_17.map(lambda x: MACRO_MAP.get(x, 'Differentiation entities'))
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-8)
    p = min(pca_dims, X.shape[1], X.shape[0])
    Xp = PCA(n_components=p, random_state=rs).fit_transform(Xz.values)

    macro_levels = [m for m in pd.Index(macro.unique()) if (macro==m).sum()>1]
    if len(macro_levels) < 2:
        warnings.warn("宏类别太少，无法引导聚类；退回普通 KMeans")
        km = KMeans(n_clusters=3, random_state=rs, n_init=10).fit(Xp)
        groups = map_groups(km.labels_, X.index)
        return groups, Xp, macro

    centroids = np.vstack([Xp[(macro.values==m)].mean(axis=0) for m in macro_levels])
    k = len(macro_levels)
    km = KMeans(n_clusters=k, init=centroids, n_init=1, max_iter=300, random_state=rs).fit(Xp)
    order = pd.Series(km.labels_).value_counts().index.tolist()
    groups = pd.Series([f"G{order.get(l,0)+1}" for l in km.labels_], index=X.index, name='Group')
    # 用统一规则重编号（按簇大小 G1..Gn）
    ser = pd.Series(km.labels_, index=X.index)
    size_order = ser.value_counts().index.tolist()
    ren = {lab: f"G{i+1}" for i, lab in enumerate(size_order)}
    groups = ser.map(ren).rename("Group")
    return groups, Xp, macro

# ================= 方案D：半监督（17 类锚定）=================
def _kmeanspp_extra_centers(Xp: np.ndarray, centers: np.ndarray, extra_k: int, rs: int):
    # 简单的“最远点”近似：每次选对现有中心距离最大的点
    rng = np.random.default_rng(rs)
    if extra_k <= 0:
        return centers
    current = centers.copy()
    for _ in range(extra_k):
        d2 = ((Xp[:,None,:] - current[None,:,:])**2).sum(axis=2).min(axis=1)
        idx = int(np.argmax(d2))
        current = np.vstack([current, Xp[idx]])
    return current

def semisup17_groups(X: pd.DataFrame, fusion_17: pd.Series, pca_dims=30, extra_k=0, rs=0):
    """以 17 类为锚定的半监督 KMeans；支持加 extra_k 个自由中心捕获潜在新类。"""
    # PCA 嵌入
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-8)
    p = min(pca_dims, X.shape[1], max(2, X.shape[0]-1))
    Xp = PCA(n_components=p, random_state=rs).fit_transform(Xz.values)

    # 以每个 17 类的样本均值作为初始质心
    f_levels = fusion_17.unique().tolist()
    seeds = []
    for f in f_levels:
        mask = (fusion_17.values==f)
        if mask.sum() == 0: continue
        seeds.append(Xp[mask].mean(axis=0))
    seeds = np.vstack(seeds)
    # 可选：加 extra_k 个自由种子
    centers = _kmeanspp_extra_centers(Xp, seeds, extra_k=extra_k, rs=rs)

    k = centers.shape[0]
    km = KMeans(n_clusters=k, init=centers, n_init=1, max_iter=500, random_state=rs).fit(Xp)

    # 按簇大小映射成 G1..Gn
    ser = pd.Series(km.labels_, index=X.index)
    size_order = ser.value_counts().index.tolist()
    ren = {lab: f"G{i+1}" for i, lab in enumerate(size_order)}
    groups = ser.map(ren).rename("Group")
    return groups, Xp

# ================= 绘图（纵向 alluvial） =================
def plot_alluvial(counts_df, title, out_png, out_pdf, g_order, f_order, label_y=-0.15, label_angle=60):
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
    g_colors = {g: palette[i%len(palette)] for i,g in enumerate(g_order)}

    fig, ax = plt.subplots(figsize=(11,11))

    # 顶部条：G1..Gn
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
    ax.text(0.5, -0.45, "WHO classification (Fusion gene subtyping)", ha="center", va="top", fontsize=18)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

# ================= 导出六张表（统一函数） =================
def export_all(df_merge, counts, g_order, f_order, outdir: Path, pref: str, novel_purity: float, min_novel_size: int):
    # 1) groups
    groups = df_merge['Group']
    groups.index.name = "Sample"
    groups.to_frame("Group").to_csv(outdir / f"{pref}_groups.csv")
    # 2) 长表
    counts.to_csv(outdir / f"{pref}_group_fusion_counts.csv", index=False)
    # 3) 宽表
    xtab = counts.pivot(index="Group", columns="Fusion", values="Count").fillna(0).astype(int)
    xtab.to_csv(outdir / f"{pref}_group_fusion_table.csv")
    # 4) 摘要 + Novel 判定
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
    summary = summary.loc[natural_g_order(summary.index.tolist())]
    summary.to_csv(outdir / f"{pref}_group_summary.csv")
    # 5) novel_members + 6) 每个 Novel 组
    novel_groups = summary[summary["is_novel"]].index.tolist()
    novel_members = df_merge[df_merge["Group"].isin(novel_groups)].reset_index().rename(columns={"index":"Sample"})
    novel_members.to_csv(outdir / f"{pref}_novel_members.csv", index=False)
    for g in novel_groups:
        sub = novel_members[novel_members["Group"] == g]
        sub.to_csv(outdir / f"{pref}_samples_{g}.csv", index=False)

# ================= 主流程 =================
def run_pipeline(expr, dx, outdir,
                 kmin, kmax, iters,
                 label_y, label_angle, seed,
                 strategy,
                 method, pca_dims,
                 novel_purity, min_novel_size,
                 extra_k):

    outdir.mkdir(parents=True, exist_ok=True)
    X_full = load_expression(expr)
    X, fusion17 = select_dx198(X_full, dx)
    X_sel = top_features(X, method=method, n=1500)

    # 底部严格 17 类顺序（按样本数降序）
    f_order = fusion17.value_counts().index.tolist()

    if strategy.lower() == 'semisup17':
        # —— 半监督：17 类为锚定 + 可选 extra_k
        groups, Xp = semisup17_groups(X_sel, fusion17, pca_dims=pca_dims, extra_k=extra_k, rs=seed)
        df_merge = pd.concat([groups.rename('Group'), fusion17.rename('Fusion')], axis=1).dropna()
        counts = build_counts(groups, fusion17)
        g_order = natural_g_order(groups.unique().tolist())
        pref = f"Dx198_{method.upper()}_SEMISUP17_pca{pca_dims}_extra{extra_k}"
        export_all(df_merge, counts, g_order, f_order, outdir, pref, novel_purity, min_novel_size)
        plot_alluvial(counts, "Subgroups", outdir / f"{pref}_alluvial.png", outdir / f"{pref}_alluvial.pdf",
                      g_order=g_order, f_order=f_order, label_y=label_y, label_angle=label_angle)

    elif strategy.lower() == 'guided':
        # —— 你原本的 guided（宏类锚定）
        groups, Xp, macro = guided_kmeans_groups(X_sel, fusion17, pca_dims=min(15, pca_dims), rs=seed)
        df_merge = pd.concat([groups.rename('Group'), fusion17.rename('Fusion')], axis=1).dropna()
        counts = build_counts(groups, fusion17)
        g_order = natural_g_order(groups.unique().tolist())
        pref = f"Dx198_{method.upper()}_guided"
        export_all(df_merge, counts, g_order, f_order, outdir, pref, novel_purity, min_novel_size)
        macro.rename("MacroFusion").to_frame().to_csv(outdir / f"{pref}_macro_labels.csv")
        plot_alluvial(counts, "Subgroups", outdir / f"{pref}_alluvial.png", outdir / f"{pref}_alluvial.pdf",
                      g_order=g_order, f_order=f_order, label_y=label_y, label_angle=label_angle)

    else:
        # —— 你的 AB（无监督 + 对齐 + 微调）
        k = choose_k_by_silhouette(X_sel, kmin=kmin, kmax=kmax, pca_dims=min(15, pca_dims), rs=seed)
        labels, Xp = consensus_labels(X_sel, k=k, iters=iters, subsample=0.6, pca_dims=min(15, pca_dims), rs=seed+11)
        groups = map_groups(labels, X_sel.index)
        df_merge = pd.concat([groups.rename("Group"), fusion17.rename("Fusion")], axis=1).dropna()
        mappingA, purityA, gA, fA = hungarian_align(df_merge)
        countsA = build_counts(groups, fusion17)
        prefA = f"Dx198_{method.UPPER()}_k{k}_Aalign"
        export_all(df_merge, countsA, gA, fA, outdir, prefA, novel_purity, min_novel_size)
        plot_alluvial(countsA, "Subgroups", outdir / f"{prefA}_alluvial.png", outdir / f"{prefA}_alluvial.pdf",
                      g_order=gA, f_order=fA, label_y=label_y, label_angle=label_angle)

        # 微调 B
        groupsB, mappingB, moved = micro_reassign(groups.copy(), fusion17, mappingA, Xp,
                                                   distance_margin=0.05, max_iter=1)
        mergeB = pd.concat([groupsB.rename("Group"), fusion17.rename("Fusion")], axis=1).dropna()
        mappingB2, purityB, gB, fB = hungarian_align(mergeB)
        countsB = build_counts(groupsB, fusion17)
        prefB = f"Dx198_{method.UPPER()}_k{k}_AB"
        export_all(mergeB, countsB, gB, fB, outdir, prefB, novel_purity, min_novel_size)
        moved.to_csv(outdir / f"{prefB}_moved_samples.csv", index=False)
        plot_alluvial(countsB, "Subgroups", outdir / f"{prefB}_alluvial.png", outdir / f"{prefB}_alluvial.pdf",
                      g_order=gB, f_order=fB, label_y=label_y, label_angle=label_angle)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dx198：半监督(17类锚定)/引导/无监督(A+B) + 纵向Alluvial + 六表导出")
    parser.add_argument("--expr", required=True, type=Path)
    parser.add_argument("--dx", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    # 策略
    parser.add_argument("--strategy", type=str, default="semisup17", choices=["semisup17","guided","AB"],
                        help="semisup17=17类锚定半监督；guided=宏类锚定；AB=无监督对齐+微调")
    # 特征 & PCA
    parser.add_argument("--method", type=str, default="MAD", choices=["MAD","Variance"], help="特征选择方法")
    parser.add_argument("--pca-dims", type=int, default=30, help="PCA 维数（semisup17 默认 30）")
    # AB 需要的参数
    parser.add_argument("--kmin", type=int, default=3)
    parser.add_argument("--kmax", type=int, default=12)
    parser.add_argument("--iters", type=int, default=60)
    # Novel 判定阈值
    parser.add_argument("--novel-purity", type=float, default=0.70, help="G 的 top_fusion 占比阈值")
    parser.add_argument("--min-novel-size", type=int, default=5, help="判定 Novel 的最小组规模")
    # semisup17 额外自由中心（用于吸纳潜在新类）
    parser.add_argument("--extra-k", type=int, default=0, help="在17个锚定之外额外添加的自由质心个数")
    # 绘图
    parser.add_argument("--label-y", type=float, default=-0.15)
    parser.add_argument("--angle", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    run_pipeline(args.expr, args.dx, args.outdir,
                 kmin=args.kmin, kmax=args.kmax, iters=args.iters,
                 label_y=args.label_y, label_angle=args.angle, seed=args.seed,
                 strategy=args.strategy,
                 method=args.method, pca_dims=args.pca_dims,
                 novel_purity=args.novel_purity, min_novel_size=args.min_novel_size,
                 extra_k=args.extra_k)
