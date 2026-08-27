"""AUC/Gini/KS, decis, PSI e utilidades comuns."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def metr(y, s) -> dict:
    """s = escore de risco (maior = mais risco). Para score 100-900 passe -score."""
    y, s = np.asarray(y).astype(int), np.asarray(s, float)
    if len(np.unique(y)) < 2:
        return {"AUC": np.nan, "Gini": np.nan, "KS": np.nan}
    auc = roc_auc_score(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    return {"AUC": float(auc), "Gini": float(2 * auc - 1), "KS": float((tpr - fpr).max())}


def decis(y, p, n=10) -> pd.DataFrame:
    t = pd.DataFrame({"y": np.asarray(y).astype(int), "p": np.asarray(p, float)})
    t["decil"] = pd.qcut(t["p"].rank(method="first"), n, labels=False) + 1
    t["decil"] = t["decil"].max() + 1 - t["decil"]
    d = (t.groupby("decil").agg(n=("y", "size"), defaults=("y", "sum"), pd_media=("p", "mean"), taxa_default=("y", "mean"))
         .reset_index())
    d["pct_defaults_acum"] = d["defaults"].cumsum() / d["defaults"].sum()
    return d


def psi(ref, cur, bins) -> float:
    r = np.histogram(ref, bins)[0] / max(len(ref), 1)
    c = np.histogram(cur, bins)[0] / max(len(cur), 1)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(((c - r) * np.log(c / r)).sum())


def psi_disc(r, c) -> float:
    r, c = np.clip(np.asarray(r, float), 1e-6, None), np.clip(np.asarray(c, float), 1e-6, None)
    return float(((c - r) * np.log(c / r)).sum())


def janela_de(safra, tipo: str):
    s = pd.Series(safra).astype(int)
    a, m = (s // 100).astype(str), (s % 100)
    if tipo == "M":
        return (a + m.astype(str).str.zfill(2)).values
    if tipo == "B":
        return (a + "B" + ((m - 1) // 2 + 1).astype(str)).values
    if tipo == "T":
        return (a + "T" + ((m - 1) // 3 + 1).astype(str)).values
    if tipo == "S":
        return (a + "S" + np.where(m <= 6, "1", "2")).values
    raise ValueError(tipo)


def cortes_custo(y, p, custo_fn: float, custo_fp: float, grid=None):
    from sklearn.metrics import confusion_matrix, f1_score
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    grid = np.arange(0.01, 0.90, 0.01) if grid is None else grid
    custos, f1s = [], []
    for t in grid:
        yhat = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        custos.append(fn * custo_fn + fp * custo_fp)
        f1s.append(f1_score(y, yhat, zero_division=0))
    i_c, i_f = int(np.argmin(custos)), int(np.argmax(f1s))
    return {"corte_custo": float(grid[i_c]), "custo_min": float(custos[i_c]), "pct_aprovacao_custo": float((p < grid[i_c]).mean()),
            "corte_f1": float(grid[i_f]), "f1_max": float(f1s[i_f]), "pct_aprovacao_f1": float((p < grid[i_f]).mean())}
