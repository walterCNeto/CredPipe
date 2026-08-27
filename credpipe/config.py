"""Leitura e validação do layout.yaml."""
from __future__ import annotations
from pathlib import Path
import math
import yaml

DEFAULTS = {
    "seed": 42, "oos_frac": 0.20, "oot": {"n_safras": 6}, "maturacao": {"safra_max_valida": None}, "ignorar": [],
    "lr": {"train_size": 0.70, "fold": 5, "tune_iter": 20, "backend": "auto"},
    "escala": {"modo": "faixa_cheia", "pdo": 40, "score0": 600, "odds0": 20, "lo": 100, "hi": 900},
    "binagem": {"n_quantis": 5, "nivel_raro_cat": 0.02},
    "restricoes": {"iv_min": 0.02, "n_min_faixa": 0.05, "monotonico": True, "rho_min": 0.80, "csi_max": 0.10,
                   "beta_tol": 0.0, "janela": "S", "max_iter": 40},
    "gh": {"k_inicial": 10, "min_gh": 4, "min_def_gh": 20, "janela": "B", "min_n_celula": 30, "max_frac_cruz": 0.0,
           "alpha_cruz": 0.10, "testar_dev": True},
    "calibracao": {"moc": "IC95_sup"},
    "desafiantes": {"ativo": True, "top_k": 3, "tune_iter": 20},
    "custo_fn_fp": [10, 1],
    "formato": {"casas": 3, "sep_decimal": ","},
    "relatorio": {"titulo": "Modelo de PD — Scorecard", "autor": "", "pdf": "auto"},
}


def _merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for k, v in (extra or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def _inf(x):
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("inf", "+inf", ".inf"):
            return math.inf
        if s in ("-inf", "-.inf"):
            return -math.inf
        return float(x)
    return float(x)


def load_layout(path: str | Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return validar(raw)


def validar(raw: dict) -> dict:
    for k in ("id", "safra", "target", "variaveis"):
        if k not in raw:
            raise ValueError(f"layout.yaml: chave obrigatória ausente: '{k}'")
    cfg = _merge(DEFAULTS, raw)
    if not isinstance(cfg["variaveis"], dict) or not cfg["variaveis"]:
        raise ValueError("layout.yaml: 'variaveis' deve ser um mapa nome -> {tipo, ...} não vazio")
    for v, spec in list(cfg["variaveis"].items()):
        spec = dict(spec or {})
        tipo = spec.get("tipo", "num")
        if tipo not in ("num", "cat"):
            raise ValueError(f"variável '{v}': tipo deve ser 'num' ou 'cat' (recebido '{tipo}')")
        if "bins" in spec:
            if tipo != "num":
                raise ValueError(f"variável '{v}': 'bins' só se aplica a tipo num")
            spec["bins"] = sorted(_inf(b) for b in spec["bins"])
            if spec["bins"][0] != -math.inf or spec["bins"][-1] != math.inf:
                raise ValueError(f"variável '{v}': bins devem começar em -inf e terminar em inf")
        if spec.get("sinal_esperado") not in (None, "+", "-"):
            raise ValueError(f"variável '{v}': sinal_esperado deve ser '+' ou '-'")
        spec["tipo"] = tipo
        cfg["variaveis"][v] = spec
    if not (0 < cfg["oos_frac"] < 1):
        raise ValueError("oos_frac deve estar em (0, 1)")
    if cfg["escala"]["modo"] not in ("faixa_cheia", "pdo"):
        raise ValueError("escala.modo deve ser 'faixa_cheia' ou 'pdo'")
    if cfg["restricoes"]["janela"] not in ("S", "T", None) or cfg["gh"]["janela"] not in ("M", "B", "T"):
        raise ValueError("restricoes.janela ∈ {S,T,null}; gh.janela ∈ {M,B,T}")
    return cfg


def variaveis(cfg: dict) -> list:
    return [v for v in cfg["variaveis"] if v not in cfg["ignorar"]]


def numericas(cfg: dict) -> list:
    return [v for v in variaveis(cfg) if cfg["variaveis"][v]["tipo"] == "num"]


def categoricas(cfg: dict) -> list:
    return [v for v in variaveis(cfg) if cfg["variaveis"][v]["tipo"] == "cat"]
