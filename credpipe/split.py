"""dev_train / OOS / OOT pela safra."""
from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split


def dividir(df: pd.DataFrame, cfg: dict):
    saf, tgt = cfg["safra"], cfg["target"]
    safras = sorted(df[saf].unique())
    if "a_partir_de" in cfg["oot"] and cfg["oot"]["a_partir_de"]:
        oot_saf = [s for s in safras if s >= int(cfg["oot"]["a_partir_de"])]
    else:
        oot_saf = safras[-int(cfg["oot"]["n_safras"]):]
    if not oot_saf or len(oot_saf) >= len(safras):
        raise ValueError("OOT vazio ou igual à base inteira — revise 'oot' no layout")
    oot = df[df[saf].isin(oot_saf)].copy()
    dev = df[~df[saf].isin(oot_saf)].copy()
    dev_tr, oos = train_test_split(dev, test_size=cfg["oos_frac"], random_state=cfg["seed"], stratify=dev[tgt])
    return {"dev_train": dev_tr, "oos": oos, "oot": oot, "dev": dev, "safras_oot": [int(s) for s in oot_saf]}
