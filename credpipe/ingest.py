"""Validação da base contra o layout: id único, target 0/1, safra YYYYMM, maturação, variáveis presentes e tipadas."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config as C


def ler(path: str) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    if p.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path, sep=None, engine="python")


def validar(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    erros = []
    for col in (cfg["id"], cfg["safra"], cfg["target"]):
        if col not in df.columns:
            erros.append(f"coluna obrigatória ausente: '{col}'")
    faltam = [v for v in C.variaveis(cfg) if v not in df.columns]
    if faltam:
        erros.append(f"variáveis do layout ausentes na base: {faltam}")
    if erros:
        raise ValueError("; ".join(erros))

    out = df.copy()
    if out[cfg["id"]].duplicated().any():
        raise ValueError(f"'{cfg['id']}' tem {int(out[cfg['id']].duplicated().sum())} duplicatas")
    y = out[cfg["target"]]
    if y.isna().any():
        raise ValueError(f"target '{cfg['target']}' tem {int(y.isna().sum())} nulos")
    vals = set(pd.unique(y))
    if not vals <= {0, 1, 0.0, 1.0, True, False}:
        raise ValueError(f"target deve ser 0/1; valores encontrados: {sorted(map(str, vals))[:10]}")
    out[cfg["target"]] = y.astype(int)

    s = pd.to_numeric(out[cfg["safra"]].astype(str).str.replace("-", "", regex=False).str[:6], errors="coerce")
    if s.isna().any() or not ((s // 100 >= 1990) & (s % 100 >= 1) & (s % 100 <= 12)).all():
        raise ValueError(f"safra '{cfg['safra']}' deve ser YYYYMM; exemplos inválidos: {out[cfg['safra']][s.isna()].head(3).tolist()}")
    out[cfg["safra"]] = s.astype(int)
    smax = cfg["maturacao"].get("safra_max_valida")
    if smax:
        out = out[out[cfg["safra"]] <= int(smax)].copy()
        if out.empty:
            raise ValueError("nenhuma safra <= safra_max_valida")

    for v in C.numericas(cfg):
        out[v] = pd.to_numeric(out[v], errors="coerce")
    for v in C.categoricas(cfg):
        out[v] = out[v].astype("string").fillna("MISSING").astype(str)
    if out[cfg["target"]].nunique() < 2:
        raise ValueError("target constante")
    return out


def sumario(df: pd.DataFrame, cfg: dict) -> dict:
    y, s = df[cfg["target"]], df[cfg["safra"]]
    por_safra = df.groupby(s)[cfg["target"]].agg(n="size", taxa_default="mean").reset_index()
    nulos = {v: float(df[v].isna().mean()) for v in C.variaveis(cfg)}
    return {"n": int(len(df)), "taxa_default": float(y.mean()), "safras": sorted(map(int, s.unique())),
            "por_safra": por_safra, "nulos": nulos}
