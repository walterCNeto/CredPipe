"""Figuras do relatório (matplotlib, sem dependência do PyCaret)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ESCALA = 0.80   # tamanho relativo das figuras
plt.rcParams.update({"figure.dpi": 100, "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})
_subplots = plt.subplots
def _sub(*a, figsize=(6.4, 4.8), **k):
    return _subplots(*a, figsize=(figsize[0] * ESCALA, figsize[1] * ESCALA), **k)
plt.subplots = _sub
from sklearn.metrics import roc_curve
from .grades import matriz_gh_tempo, tab_gh
from .metrics import janela_de


def _save(fig, out: Path, nome: str) -> str:
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{nome}.png"; fig.tight_layout(); fig.savefig(p, dpi=220); plt.close(fig)
    return p.name


def roc(y_by, p_by, out) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for nome, (y, p) in zip(y_by, zip(y_by.values(), p_by.values())):
        fpr, tpr, _ = roc_curve(y, p); ax.plot(fpr, tpr, label=nome)
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("Curva ROC"); ax.legend()
    return _save(fig, out, "roc")


def calibracao(y, p, out, n=10) -> str:
    t = pd.DataFrame({"y": y, "p": p}); t["b"] = pd.qcut(t["p"].rank(method="first"), n, labels=False)
    g = t.groupby("b").agg(pm=("p", "mean"), ob=("y", "mean"))
    fig, ax = plt.subplots(figsize=(5, 5)); ax.plot(g["pm"], g["ob"], "o-"); lim = [0, max(g.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1); ax.set_xlabel("PD média prevista"); ax.set_ylabel("taxa observada"); ax.set_title("Calibração (decis)")
    return _save(fig, out, "calibracao")


def decis(d: pd.DataFrame, out) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(d["decil"], d["taxa_default"], label="taxa observada"); ax.plot(d["decil"], d["pd_media"], "ko-", label="PD média")
    ax.set_xlabel("decil (1 = maior risco)"); ax.set_title("Tabela de decis (OOT)"); ax.legend()
    return _save(fig, out, "decis")


def score_dist(score, y, lo, hi, out) -> str:
    s = pd.Series(np.asarray(score)); faixas = pd.cut(s, bins=range(lo, hi + 1, 100), include_lowest=True)
    tx = pd.Series(np.asarray(y)).groupby(faixas, observed=True).mean()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    s.hist(bins=40, ax=ax[0]); ax[0].set_title(f"Distribuição do score ({lo}–{hi})")
    tx.plot(kind="bar", ax=ax[1]); ax[1].set_title("Taxa de default por faixa de 100 pontos"); ax[1].tick_params(axis="x", rotation=45)
    return _save(fig, out, "score_dist")


def por_safra(ps: pd.DataFrame, safras_oot, out) -> str:
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    x = np.arange(len(ps)); lab = ps["safra"].astype(str)
    ax[0].plot(x, ps["taxa_default"], "o-", label="observado"); ax[0].plot(x, ps["pd_media"], "s--", label="PD média"); ax[0].legend()
    ax[0].set_title("Observado vs. previsto por safra")
    ax[1].plot(x, ps["AUC"], "o-", label="AUC"); ax[1].plot(x, ps["KS"], "s-", label="KS"); ax[1].plot(x, ps["PSI_score"], "^-", label="PSI"); ax[1].legend()
    ax[1].set_title("Discriminação e PSI por safra")
    i0 = int(np.where(ps["safra"].isin(safras_oot))[0].min()) - 0.5
    for a in ax:
        a.axvspan(i0, len(ps) - 0.5, alpha=0.12, color="grey"); a.set_xticks(x); a.set_xticklabels(lab, rotation=90, fontsize=7)
    return _save(fig, out, "por_safra")


def betas(b: pd.DataFrame, out) -> str:
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * b.shape[1]), 0.45 * b.shape[0] + 1.5))
    im = ax.imshow(b.values, cmap="RdYlGn_r", aspect="auto", vmin=min(-2, np.nanmin(b.values)), vmax=max(0.5, np.nanmax(b.values)))
    ax.set_xticks(range(b.shape[1])); ax.set_xticklabels(b.columns, rotation=45, ha="right"); ax.set_yticks(range(b.shape[0])); ax.set_yticklabels(b.index)
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            ax.text(j, i, f"{b.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax); ax.set_title("Betas sobre WoE por janela (esperado < 0)")
    return _save(fig, out, "betas_janela")


def painel_gh(s_all, y_all, saf_all, edges, R, mapa, s_oos, y_oos, safras_oot, out) -> str:
    taxa, n = matriz_gh_tempo(s_all, y_all, saf_all, edges, R["janela"])
    taxa = taxa.where(n >= R["min_n_celula"]); pop = n.div(n.sum(axis=0), axis=1)
    oot_j = sorted(set(janela_de(np.asarray(safras_oot), R["janela"])))
    i0 = [c in oot_j for c in taxa.columns].index(True) - 0.5
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    taxa.T.plot(ax=ax[0, 0], marker="o"); ax[0, 0].axvspan(i0, taxa.shape[1] - 0.5, alpha=0.12, color="grey")
    ax[0, 0].set_title(f"Taxa de default por GH × janela ({R['janela']}) — sombreado = OOT"); ax[0, 0].legend(title="GH", ncol=2, fontsize=7)
    pop.T.plot(kind="bar", stacked=True, ax=ax[0, 1], width=0.85); ax[0, 1].axvspan(i0, pop.shape[1] - 0.5, alpha=0.12, color="grey")
    ax[0, 1].set_title("Distribuição da população por GH"); ax[0, 1].legend(title="GH", ncol=2, fontsize=7)
    x = np.arange(len(mapa))
    ax[1, 0].bar(x - 0.2, mapa["PD_modelo"], 0.4, label="PD modelo/escala (OOT)")
    ax[1, 0].bar(x + 0.2, mapa["PD_calibrada"], 0.4, label="PD calibrada = taxa OOT")
    ax[1, 0].errorbar(x + 0.2, mapa["PD_calibrada"], yerr=[mapa["PD_calibrada"] - mapa["IC95_inf"], mapa["IC95_sup"] - mapa["PD_calibrada"]],
                      fmt="none", ecolor="black", capsize=3)
    for xi, pv in zip(x, mapa["p_binomial"]):
        if pv == pv and pv < 0.05:
            ax[1, 0].annotate("*", (xi - 0.2, mapa["PD_modelo"].iloc[xi]), ha="center", va="bottom", fontsize=14)
    ax[1, 0].set_xticks(x); ax[1, 0].set_xticklabels([f"GH{g}" for g in mapa.index]); ax[1, 0].legend()
    ax[1, 0].set_title("Calibração por GH (* = binomial rejeita PD do modelo a 5%)")
    comp = pd.DataFrame({"dev": mapa["taxa_dev"], "OOS": mapa["taxa_oos"], "OOT": mapa["PD_calibrada"]})
    comp.plot(kind="bar", ax=ax[1, 1], width=0.8); ax[1, 1].set_xticklabels([f"GH{g}" for g in comp.index], rotation=0)
    ax[1, 1].set_title("Taxa observada por GH e amostra")
    return _save(fig, out, "painel_gh")


def calib_oos_vs_oot(mapa, out) -> str:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(mapa["PD_calibrada"], mapa["taxa_oos"], s=np.clip(mapa["n_oot"].values / 5, 10, 400))
    for g, r in mapa.iterrows():
        ax.annotate(f"GH{g}", (r["PD_calibrada"], r["taxa_oos"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    lim = [0, float(max(mapa["PD_calibrada"].max(), mapa["taxa_oos"].max())) * 1.1]; ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("PD calibrada (OOT)"); ax.set_ylabel("taxa observada no OOS"); ax.set_title("Validação cruzada da calibração por GH")
    return _save(fig, out, "calib_oos_oot")


def desafiantes(res: pd.DataFrame, lr_oos, lr_oot, sc_oos, sc_oot, out) -> str:
    t = pd.concat([pd.DataFrame([{"modelo": "Regressão logística", "AUC_OOS": lr_oos, "AUC_OOT": lr_oot},
                                 {"modelo": "Scorecard refinado", "AUC_OOS": sc_oos, "AUC_OOT": sc_oot}]), res[["modelo", "AUC_OOS", "AUC_OOT"]]])
    fig, ax = plt.subplots(figsize=(8, 4)); t.set_index("modelo").plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylim(0.5, max(0.8, t[["AUC_OOS", "AUC_OOT"]].max().max() + 0.02)); ax.set_title("AUC: LR, scorecard e desafiantes"); ax.tick_params(axis="x", rotation=20)
    return _save(fig, out, "desafiantes")
