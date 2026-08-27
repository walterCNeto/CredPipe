"""Refino recursivo de faixas e variáveis sob restrições de monotonicidade, estabilidade e sinal (Algoritmo H)."""
from __future__ import annotations
import copy
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from . import scorecard as SC


def _vizinho(t, i, num):
    cands = [k for k in (i - 1, i + 1) if 0 <= k < len(t)] if num else [k for k in range(len(t)) if k != i]
    return min(cands, key=lambda k: abs(t["woe"].iloc[k] - t["woe"].iloc[i]))


def _mesclar(spec, v, i, j, t):
    s = spec[v]
    if s["tipo"] == "num":
        faixas = list(t.index)
        # só mescla faixas de intervalo (MISSING fica separada)
        if faixas[i] == SC.MISSING or faixas[j] == SC.MISSING:
            return False
        s["edges"].pop(max(i, j))
    else:
        a, b = str(t.index[i]), str(t.index[j])
        novo = f"{a}+{b}"
        s["grupos"] = {c: (novo if g in (a, b) else g) for c, g in s["grupos"].items()}
    return True


def _monotonico(w):
    if len(w) < 3:
        return True, None
    rho = spearmanr(np.arange(len(w)), w).correlation
    sinal = -1 if (rho == rho and rho < 0) else 1
    bad = np.where(np.sign(np.diff(w)) == -sinal)[0]
    return (len(bad) == 0), (int(bad[0]) if len(bad) else None)


def diagnosticar(v, spec, C_dev, y_dev, C_oot, y_oot, R):
    num = spec[v]["tipo"] == "num"
    td = SC.tab_woe(C_dev[v], y_dev)
    to = SC.tab_woe(C_oot[v], y_oot).reindex(td.index)
    # MISSING não entra na monotonicidade nem no vizinho — separa
    mask = np.array([str(f) != SC.MISSING for f in td.index])
    tdn, ton = td[mask], to[mask]
    wd, wo = tdn["woe"].values, ton["woe"].values
    if len(tdn) < 2:
        return ("drop", "restou 1 faixa")
    if td["iv"].sum() < R["iv_min"]:
        return ("drop", f"IV={td['iv'].sum():.3f}<{R['iv_min']}")
    i = int(np.argmin(tdn["pct"].values))
    if tdn["pct"].iloc[i] < R["n_min_faixa"]:
        return ("merge", i, _vizinho(tdn, i, num), f"faixa {tdn.index[i]} com {tdn['pct'].iloc[i]:.1%}")
    if num and R["monotonico"]:
        ok, i = _monotonico(wd)
        if not ok:
            return ("merge", i, i + 1, "WoE não monotônico")
    inv = np.where(wd * wo < 0)[0]
    if len(inv):
        i = int(inv[np.argmax(np.abs(wd[inv]))])
        return ("merge", i, _vizinho(tdn, i, num), "inversão de sinal no OOT")
    if len(tdn) > 2:
        rho = spearmanr(wd, wo).correlation
        if rho == rho and rho < R["rho_min"]:
            i = int(np.argmax(np.abs(tdn["woe"].rank().values - ton["woe"].rank().values)))
            return ("merge", i, _vizinho(tdn, i, num), f"rho={rho:.2f}<{R['rho_min']}")
    r, c = np.clip(tdn["pct"].values, 1e-6, None), np.clip(ton["pct"].values, 1e-6, None)
    contrib = (c - r) * np.log(c / r); csi = float(contrib.sum())
    if csi > R["csi_max"]:
        if len(tdn) == 2:
            return ("drop", f"CSI={csi:.3f}")
        i = int(np.argmax(contrib))
        return ("merge", i, _vizinho(tdn, i, num), f"CSI={csi:.3f}")
    return None


def refinar(spec0: dict, X_dev, y_dev, X_oot, y_oot, X_all, y_all, safra_all, cfg) -> tuple:
    """Devolve (spec_final, log DataFrame, betas por janela, status)."""
    R = cfg["restricoes"]
    sinais = {v: cfg["variaveis"][v].get("sinal_esperado") for v in spec0}
    spec = copy.deepcopy(spec0)
    log, status = [], "max_iter atingido"
    for it in range(1, R["max_iter"] + 1):
        C_dev, C_oot = SC.binar(X_dev, spec), SC.binar(X_oot, spec)
        agiu = False
        for v in list(spec):
            d = diagnosticar(v, spec, C_dev, y_dev, C_oot, y_oot, R)
            if d is None:
                continue
            if d[0] == "drop":
                log.append({"iter": it, "variavel": v, "acao": "descarta", "motivo": d[1]}); spec.pop(v); agiu = True
            else:
                td = SC.tab_woe(C_dev[v], y_dev)
                td = td[[str(f) != SC.MISSING for f in td.index]]
                if _mesclar(spec, v, d[1], d[2], td):
                    log.append({"iter": it, "variavel": v, "acao": f"mescla {td.index[d[1]]} + {td.index[d[2]]}", "motivo": d[3]}); agiu = True
        if agiu:
            continue
        if not spec:
            status = "todas as variáveis descartadas"; break
        maps = SC.woe_maps(spec, C_dev, y_dev)
        W_dev = SC.transformar(X_dev, spec, maps)
        lr = SC.fit_lr_woe(W_dev, y_dev)
        betas = pd.DataFrame({"dev": lr.coef_[0]}, index=list(spec))
        bj = SC.betas_por_janela(spec, maps, X_all, y_all, safra_all, R["janela"])
        betas = pd.concat([betas, bj], axis=1)
        pior = betas.max(axis=1)                      # WoE = ln(bons/maus) → beta esperado < 0
        if pior.max() > R["beta_tol"]:
            v = pior.idxmax(); jan = betas.loc[v].idxmax()
            log.append({"iter": it, "variavel": v, "acao": "descarta", "motivo": f"beta={pior[v]:+.3f} em {jan} (contra a lógica WoE)"})
            spec.pop(v); continue
        # sinal econômico esperado: coeficiente da LR contínua é o teste natural; aqui verificamos pelo WoE da 1ª vs última faixa
        contra = []
        for v in spec:
            if spec[v]["tipo"] == "num" and sinais.get(v):
                t = SC.tab_woe(C_dev[v], y_dev); w = t[[str(f) != SC.MISSING for f in t.index]]["woe"].values
                if len(w) >= 2:
                    tendencia = "-" if w[-1] > w[0] else "+"   # WoE cresce com x → risco cai → sinal '-'
                    if tendencia != sinais[v]:
                        contra.append(v)
        if contra:
            v = contra[0]
            log.append({"iter": it, "variavel": v, "acao": "descarta", "motivo": f"tendência contra sinal_esperado='{sinais[v]}'"})
            spec.pop(v); continue
        status = f"convergiu na iteração {it}"
        return spec, pd.DataFrame(log), betas, status
    return spec, pd.DataFrame(log), pd.DataFrame(), status
