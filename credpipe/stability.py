"""Estabilidade temporal: métricas e PSI por safra, CSI/rho/inversões por variável."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import scorecard as SC
from .metrics import metr, psi, psi_disc


def por_safra(df_val: pd.DataFrame, cfg: dict, col_p: str = "p", col_score: str = "score", ref_mask=None) -> pd.DataFrame:
    """df_val precisa de: target, safra, amostra ('OOS'|'OOT'), p (PD) e score."""
    tgt, saf = cfg["target"], cfg["safra"]
    ref = df_val.loc[ref_mask if ref_mask is not None else (df_val["amostra"] == "OOS"), col_score]
    bins = np.quantile(ref, np.linspace(0, 1, 11)); bins[0], bins[-1] = -np.inf, np.inf
    linhas = []
    for s, g in df_val.groupby(saf):
        linhas.append({"safra": int(s), "amostra": g["amostra"].iloc[0], "n": len(g), "taxa_default": g[tgt].mean(),
                       "pd_media": g[col_p].mean(), **metr(g[tgt], g[col_p]), "PSI_score": psi(ref, g[col_score], bins)})
    return pd.DataFrame(linhas)


def por_variavel(spec: dict, X_dev, y_dev, X_oot, y_oot) -> tuple:
    """Devolve (comparativo por faixa, resumo por variável com rho/inversões/CSI)."""
    C_dev, C_oot = SC.binar(X_dev, spec), SC.binar(X_oot, spec)
    comp_l, res_l = [], []
    for v in spec:
        td = SC.tab_woe(C_dev[v], y_dev); to = SC.tab_woe(C_oot[v], y_oot).reindex(td.index)
        for f in td.index:
            comp_l.append({"variavel": v, "faixa": str(f), "taxa_dev": td.loc[f, "taxa"], "taxa_oot": to.loc[f, "taxa"],
                           "woe_dev": td.loc[f, "woe"], "woe_oot": to.loc[f, "woe"], "pct_dev": td.loc[f, "pct"], "pct_oot": to.loc[f, "pct"],
                           "inversao_sinal": bool(td.loc[f, "woe"] * to.loc[f, "woe"] < 0)})
        res_l.append({"variavel": v, "rho_ordenacao": SC.rho_ordenacao(td, to),
                      "faixas_com_inversao": int(((td["woe"] * to["woe"]) < 0).sum()),
                      "CSI": psi_disc(td["pct"].values, to["pct"].values), "IV_dev": float(td["iv"].sum()), "IV_oot": float(to["iv"].sum())})
    res = pd.DataFrame(res_l).set_index("variavel").sort_values("CSI", ascending=False)
    return pd.DataFrame(comp_l), res
