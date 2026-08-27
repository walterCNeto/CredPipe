"""Categorização (faixas), WoE/IV, regressão sobre WoE, pontos por faixa e escoragem."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from . import config as C
from .metrics import janela_de

MISSING = "MISSING"


# ------------------------------------------------------------------ spec de faixas
def spec_inicial(X: pd.DataFrame, cfg: dict) -> dict:
    """{var: {'tipo':'num','edges':[...]} | {'tipo':'cat','grupos':{nivel:grupo}}} a partir do layout + quantis do dev."""
    nq, raro = cfg["binagem"]["n_quantis"], cfg["binagem"]["nivel_raro_cat"]
    spec = {}
    for v in C.variaveis(cfg):
        s = cfg["variaveis"][v]
        if s["tipo"] == "num":
            if s.get("bins"):
                edges = list(s["bins"])
            else:
                x = X[v].dropna()
                e = np.unique(np.quantile(x, np.linspace(0, 1, nq + 1))) if len(x) else np.array([0.0])
                edges = [-np.inf] + [float(t) for t in e[1:-1]] + [np.inf]
            spec[v] = {"tipo": "num", "edges": edges}
        else:
            freq = X[v].astype(str).value_counts(normalize=True)
            grupos = {k: (k if f >= raro else "OUTROS") for k, f in freq.items()}
            spec[v] = {"tipo": "cat", "grupos": grupos}
    return spec


def binar(X: pd.DataFrame, spec: dict) -> pd.DataFrame:
    out = {}
    for v, s in spec.items():
        if s["tipo"] == "num":
            x = pd.to_numeric(X[v], errors="coerce")
            c = pd.cut(x, bins=s["edges"], include_lowest=True)
            if x.isna().any():
                c = c.cat.add_categories([MISSING]).fillna(MISSING)
            out[v] = c
        else:
            g = X[v].astype(str).map(s["grupos"]).fillna("OUTROS")
            cats = sorted(set(s["grupos"].values()) | ({"OUTROS"} if (g == "OUTROS").any() else set()))
            out[v] = pd.Categorical(g, categories=cats)
    return pd.DataFrame(out, index=X.index)


# ------------------------------------------------------------------ WoE / IV
def tab_woe(c: pd.Series, y, eps: float = 0.5) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    tb, tm = max((y == 0).sum(), 1), max((y == 1).sum(), 1)
    d = pd.DataFrame({"c": c.values, "y": y})
    g = d.groupby("c", observed=False)["y"].agg(n="size", defaults="sum")
    g = g.reindex(c.cat.categories, fill_value=0)
    b, m = (g["n"] - g["defaults"]) + eps, g["defaults"] + eps
    woe = np.log((b / tb) / (m / tm))
    return pd.DataFrame({"n": g["n"].astype(int), "pct": g["n"] / max(len(y), 1),
                         "taxa": g["defaults"] / g["n"].replace(0, np.nan), "defaults": g["defaults"].astype(int),
                         "woe": woe, "iv": ((b / tb) - (m / tm)) * woe})


def tabela_completa(Cb: pd.DataFrame, y) -> pd.DataFrame:
    linhas = []
    for v in Cb.columns:
        t = tab_woe(Cb[v], y)
        for faixa, r in t.iterrows():
            linhas.append({"variavel": v, "faixa": str(faixa), **r.to_dict()})
    out = pd.DataFrame(linhas)
    for c in ("n", "defaults"):
        out[c] = out[c].astype(int)
    return out


def woe_maps(spec: dict, Cb: pd.DataFrame, y) -> dict:
    return {v: {str(k): float(w) for k, w in tab_woe(Cb[v], y)["woe"].items()} for v in spec}


def transformar(X: pd.DataFrame, spec: dict, maps: dict) -> pd.DataFrame:
    Cb = binar(X, spec)
    return pd.DataFrame({v: Cb[v].astype(str).map(maps[v]).astype(float) for v in spec}, index=X.index).fillna(0.0)


def fit_lr_woe(W: pd.DataFrame, y) -> LogisticRegression:
    return LogisticRegression(C=1e6, max_iter=2000).fit(W, np.asarray(y).astype(int))


# ------------------------------------------------------------------ scorecard final
class Scorecard:
    def __init__(self, spec, maps, beta: pd.Series, alpha: float, escala, tabela: pd.DataFrame):
        self.spec, self.maps, self.beta, self.alpha, self.escala, self.tabela = spec, maps, beta, alpha, escala, tabela
        self.pts_map = {v: dict(zip(g["faixa"], g["pontos"])) for v, g in tabela.groupby("variavel")}
        self.pts_neutro = {v: int(round(np.mean(list(m.values())))) for v, m in self.pts_map.items()}

    @classmethod
    def ajustar(cls, X_dev, y_dev, spec, escala):
        Cb = binar(X_dev, spec)
        maps = woe_maps(spec, Cb, y_dev)
        W = transformar(X_dev, spec, maps)
        lr = fit_lr_woe(W, y_dev)
        beta = pd.Series(lr.coef_[0], index=list(spec)); alpha = float(lr.intercept_[0])
        K, F, O = len(spec), escala.factor, escala.offset
        tab = tabela_completa(Cb, y_dev)
        tab["pontos"] = [int(round(-F * beta[r.variavel] * r.woe + (O - F * alpha) / K)) for r in tab.itertuples()]
        tab = tab.rename(columns={"taxa": "taxa_default"})[["variavel", "faixa", "n", "pct", "taxa_default", "woe", "iv", "pontos"]]
        return cls(spec, maps, beta, alpha, escala, tab)

    def pontos(self, X: pd.DataFrame) -> pd.DataFrame:
        Cb = binar(X, self.spec)
        pts = pd.DataFrame({v: Cb[v].astype(str).map(self.pts_map[v]) for v in self.spec}, index=X.index)
        for v in self.spec:
            pts[v] = pts[v].fillna(self.pts_neutro[v]).astype(int)
        pts["score"] = pts.sum(axis=1).clip(self.escala.lo, self.escala.hi).astype(int)
        return pts

    def score(self, X: pd.DataFrame) -> pd.Series:
        return self.pontos(X)["score"]

    def iv_por_variavel(self) -> pd.Series:
        return self.tabela.groupby("variavel")["iv"].sum().sort_values(ascending=False)


# ------------------------------------------------------------------ betas por janela (teste de sinal)
def betas_por_janela(spec, maps, X_all, y_all, safra_all, tipo_janela):
    W = transformar(X_all, spec, maps)
    out = {}
    if tipo_janela:
        j = janela_de(safra_all, tipo_janela)
        for jan in sorted(set(j)):
            idx = np.where(j == jan)[0]
            yy = np.asarray(y_all)[idx]
            if len(np.unique(yy)) < 2 or len(idx) < 50:
                continue
            out[jan] = fit_lr_woe(W.iloc[idx], yy).coef_[0]
    return pd.DataFrame(out, index=list(spec))


def rho_ordenacao(t_dev: pd.DataFrame, t_oot: pd.DataFrame) -> float:
    if len(t_dev) < 3:
        return np.nan
    r = spearmanr(t_dev["woe"].values, t_oot["woe"].values).correlation
    return float(r) if r == r else np.nan
