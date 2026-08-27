"""Malha de grupos homogêneos (GH) recursiva: monotônica no dev, massa mínima e sem cruzamento em nenhuma janela válida."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .metrics import janela_de


def gh_de(score, edges):
    return pd.cut(np.asarray(score, float), bins=edges, include_lowest=True, labels=False) + 1   # 1 = menor score = maior risco


def tab_gh(s, y, edges) -> pd.DataFrame:
    t = pd.DataFrame({"gh": gh_de(s, edges), "y": np.asarray(y).astype(int)}).groupby("gh").agg(n=("y", "size"), defaults=("y", "sum"))
    t = t.reindex(range(1, len(edges)), fill_value=0)
    t["taxa"] = t["defaults"] / t["n"].replace(0, np.nan)
    return t


def matriz_gh_tempo(s, y, safra, edges, tipo):
    d = pd.DataFrame({"gh": gh_de(s, edges), "y": np.asarray(y).astype(int), "j": janela_de(safra, tipo)})
    taxa = d.pivot_table(index="gh", columns="j", values="y", aggfunc="mean")
    n = d.pivot_table(index="gh", columns="j", values="y", aggfunc="size").fillna(0)
    return taxa.reindex(range(1, len(edges))), n.reindex(range(1, len(edges))).fillna(0)


def celulas_validas(s, y, safra, edges, R):
    taxa, n = matriz_gh_tempo(s, y, safra, edges, R["janela"])
    ok = n >= R["min_n_celula"]
    return taxa.where(ok), n.where(ok)


def frac_cruzamentos(taxa: pd.DataFrame, n: pd.DataFrame | None = None, alpha: float | None = None) -> pd.Series:
    """Fração de janelas em que o par adjacente (gh−1, gh) inverte a ordem.
    Com alpha, só conta inversão significativa (teste z unilateral de duas proporções)."""
    inv = taxa.diff(axis=0) > 0                       # linha gh − linha gh−1 > 0 ⇒ par invertido
    valid = taxa.notna() & taxa.shift(1).notna()
    if alpha is not None and n is not None:
        from scipy.stats import norm
        p1, p0 = taxa, taxa.shift(1)
        n1, n0 = n, n.shift(1)
        pool = (p1 * n1 + p0 * n0) / (n1 + n0)
        se = np.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n0))
        z = (p1 - p0) / se.replace(0, np.nan)
        inv = inv & (z > norm.ppf(1 - alpha))
    f = inv.sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)
    return f.iloc[1:].rename(lambda g: f"GH{g-1}+GH{g}")


def diagnosticar(edges, R, s_dev, y_dev, saf_dev, s_oot, y_oot, saf_oot):
    t = tab_gh(s_dev, y_dev, edges)
    inv = np.where(np.diff(t["taxa"].values) > 0)[0]
    if len(inv):
        return int(inv[0]), "taxa não monotônica no dev"
    peq = np.where(t["defaults"].values < R["min_def_gh"])[0]
    if len(peq):
        i = int(peq[0]); return min(i, len(t) - 2), f"< {R['min_def_gh']} defaults no dev"
    blocos = [("OOT", s_oot, y_oot, saf_oot)] + ([("dev", s_dev, y_dev, saf_dev)] if R["testar_dev"] else [])
    pior = None
    for nome, s, y, saf in blocos:
        taxa, n = celulas_validas(s, y, saf, edges, R)
        f = frac_cruzamentos(taxa, n, R.get("alpha_cruz")).dropna()
        if len(f) and f.max() > R["max_frac_cruz"]:
            i = int(f.idxmax().split("+")[0][2:])
            if pior is None or f.max() > pior[2]:
                pior = (i, nome, float(f.max()))
    if pior:
        return pior[0] - 1, f"cruzamento no {pior[1]} em {pior[2]:.0%} das janelas"
    return None


def malha_recursiva(s_dev, y_dev, saf_dev, s_oot, y_oot, saf_oot, R: dict):
    """Devolve (edges, log, status). edges em pontos de score, com -inf/inf nas pontas."""
    e = np.unique(np.quantile(np.asarray(s_dev, float), np.linspace(0, 1, R["k_inicial"] + 1)))
    e = e.astype(float); e[0], e[-1] = -np.inf, np.inf
    log = []
    while True:
        d = diagnosticar(e, R, s_dev, y_dev, saf_dev, s_oot, y_oot, saf_oot)
        if d is None:
            status = "convergiu"; break
        if len(e) - 1 <= R["min_gh"]:
            status = f"parou em min_gh={R['min_gh']} com violação pendente: {d[1]}"; break
        i, motivo = d
        log.append({"n_gh": len(e) - 1, "acao": f"mescla GH{i+1}+GH{i+2}", "motivo": motivo})
        e = np.delete(e, i + 1)
    return e, pd.DataFrame(log), status


def verificar_ordem(s, y, safra, edges, R) -> pd.Series:
    """Por janela válida: True se não há cruzamento (significativo, se alpha_cruz) entre GHs adjacentes."""
    taxa, n = celulas_validas(s, y, safra, edges, R)
    f = frac_cruzamentos(taxa, n, R.get("alpha_cruz"))
    inv = (taxa.diff(axis=0) > 0)
    if R.get("alpha_cruz") is not None:
        from scipy.stats import norm
        p1, p0, n1, n0 = taxa, taxa.shift(1), n, n.shift(1)
        pool = (p1 * n1 + p0 * n0) / (n1 + n0); se = np.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n0))
        inv = inv & (((p1 - p0) / se.replace(0, np.nan)) > norm.ppf(1 - R["alpha_cruz"]))
    return ~inv.iloc[1:].any(axis=0)
