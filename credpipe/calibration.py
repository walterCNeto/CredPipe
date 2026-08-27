"""PD por GH calibrada na base OOT observada, IC de Jeffreys, teste binomial, HHI, de-para e escoragem."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist, binomtest
from .grades import tab_gh, gh_de


def _faixa(a, b):
    return f"{'-inf' if np.isinf(a) else round(a)} a {'+inf' if np.isinf(b) else round(b)}"


def mapa_gh(edges, s_dev, y_dev, s_oos, y_oos, s_oot, y_oot, p_modelo_oot) -> pd.DataFrame:
    t_d, t_s, t_o = tab_gh(s_dev, y_dev, edges), tab_gh(s_oos, y_oos, edges), tab_gh(s_oot, y_oot, edges)
    p_mod = pd.DataFrame({"gh": gh_de(s_oot, edges), "p": np.asarray(p_modelo_oot, float)}).groupby("gh")["p"].mean().reindex(t_o.index)
    m = pd.DataFrame({"faixa_score": [_faixa(a, b) for a, b in zip(edges[:-1], edges[1:])],
                      "n_dev": t_d["n"].values, "taxa_dev": t_d["taxa"].values,
                      "n_oos": t_s["n"].values, "taxa_oos": t_s["taxa"].values,
                      "n_oot": t_o["n"].values, "defaults_oot": t_o["defaults"].values,
                      "PD_modelo": p_mod.values, "PD_calibrada": t_o["taxa"].values}, index=t_d.index)
    m.index.name = "GH"
    k, n = m["defaults_oot"], m["n_oot"]
    m["IC95_inf"] = beta_dist.ppf(0.025, k + 0.5, n - k + 0.5)
    m["IC95_sup"] = beta_dist.ppf(0.975, k + 0.5, n - k + 0.5)
    m["p_binomial"] = [binomtest(int(kk), int(nn), float(np.clip(pp, 1e-9, 1 - 1e-9))).pvalue if nn > 0 and pp == pp else np.nan
                       for kk, nn, pp in zip(k, n, m["PD_modelo"])]
    m["pct_carteira_oot"] = m["n_oot"] / m["n_oot"].sum()
    return m


def hhi(m: pd.DataFrame) -> float:
    return float((m["pct_carteira_oot"] ** 2).sum())


def de_para(m: pd.DataFrame, edges, moc, lo=100, hi=900) -> pd.DataFrame:
    d = pd.DataFrame({
        "score_min": [lo if np.isinf(a) else int(np.floor(a)) + 1 for a in edges[:-1]],
        "score_max": [hi if np.isinf(b) else int(np.floor(b)) for b in edges[1:]],
        "n_oot": m["n_oot"].values, "pct_carteira": m["pct_carteira_oot"].values,
        "PD_12m": m["PD_calibrada"].values, "PD_IC95_inf": m["IC95_inf"].values, "PD_IC95_sup": m["IC95_sup"].values,
    }, index=m.index)
    if moc == "IC95_sup":
        d["PD_12m_conservadora"] = d["PD_IC95_sup"]
    elif isinstance(moc, (int, float)) and moc:
        d["PD_12m_conservadora"] = (d["PD_12m"] * float(moc)).clip(upper=1)
    else:
        d["PD_12m_conservadora"] = d["PD_12m"]
    d["odds_bom"] = (1 - d["PD_12m"]) / d["PD_12m"].replace(0, np.nan)
    return d


class Escorador:
    """Base bruta → score, GH, PD (artefato de produção)."""

    def __init__(self, scorecard, edges, de_para_tab: pd.DataFrame):
        self.sc, self.edges, self.de_para = scorecard, np.asarray(edges, float), de_para_tab

    def __call__(self, X: pd.DataFrame) -> pd.DataFrame:
        s = self.sc.score(X)
        gh = gh_de(s, self.edges)
        return pd.DataFrame({"score": s.values, "GH": gh,
                             "PD_12m": self.de_para.loc[gh, "PD_12m"].values,
                             "PD_12m_conservadora": self.de_para.loc[gh, "PD_12m_conservadora"].values}, index=X.index)
