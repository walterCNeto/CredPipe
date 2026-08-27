"""Escala de pontos (Eq. PDO): score = Offset + Factor * ln(odds_bom)."""
from __future__ import annotations
import numpy as np


class Escala:
    def __init__(self, factor: float, offset: float, lo: int = 100, hi: int = 900):
        self.factor, self.offset, self.lo, self.hi = float(factor), float(offset), int(lo), int(hi)
        self.pdo = float(self.factor * np.log(2))

    @classmethod
    def ajustar(cls, p, cfg_escala: dict):
        lo, hi = cfg_escala["lo"], cfg_escala["hi"]
        if cfg_escala["modo"] == "pdo":
            factor = cfg_escala["pdo"] / np.log(2)
            offset = cfg_escala["score0"] - factor * np.log(cfg_escala["odds0"])
        else:
            p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
            p_best, p_worst = np.quantile(p, [0.005, 0.995])
            lo_best, lo_worst = np.log((1 - p_best) / p_best), np.log((1 - p_worst) / p_worst)
            factor = (hi - lo) / (lo_best - lo_worst)
            offset = lo - factor * lo_worst
        return cls(factor, offset, lo, hi)

    def prob_para_score(self, p):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        return np.clip(self.offset + self.factor * np.log((1 - p) / p), self.lo, self.hi).round(0).astype(int)

    def score_para_prob(self, s):
        return 1 / (1 + np.exp((np.asarray(s, float) - self.offset) / self.factor))

    def to_dict(self):
        return {"factor": self.factor, "offset": self.offset, "pdo": self.pdo, "lo": self.lo, "hi": self.hi}
