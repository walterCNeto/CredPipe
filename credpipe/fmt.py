"""Formatação global: N casas decimais em tudo, percentuais nas colunas de taxa/PD, inteiros onde couber."""
from __future__ import annotations
import numpy as np
import pandas as pd

CASAS = 3
SEP = ","
COLS_PCT = {"taxa", "taxa_default", "taxa_dev", "taxa_oos", "taxa_oot", "pd_media", "pct", "pct_dev", "pct_oot",
            "pct_carteira", "pct_carteira_oot", "pct_defaults_acum", "PD_modelo", "PD_calibrada", "PD_12m",
            "PD_12m_conservadora", "PD_IC95_inf", "PD_IC95_sup", "IC95_inf", "IC95_sup", "taxa_default_dev",
            "taxa_default_oot", "frac_janelas_invertidas", "pct_aprovacao"}
COLS_INT = {"n", "defaults", "defaults_oot", "n_dev", "n_oos", "n_oot", "pontos", "score", "score_min", "score_max",
            "decil", "GH", "gh", "iter", "n_gh", "n_faixas", "n_variaveis", "k_gh"}


def configurar(casas: int = 3, sep_decimal: str = ",") -> None:
    global CASAS, SEP
    CASAS, SEP = int(casas), sep_decimal
    pd.set_option("display.float_format", lambda x: f3(x))
    np.set_printoptions(precision=CASAS, suppress=True)


def _nan(x) -> bool:
    try:
        return x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x)
    except Exception:
        return False


def f3(x) -> str:
    return "—" if _nan(x) else f"{float(x):.{CASAS}f}".replace(".", SEP)


def p3(x) -> str:
    return "—" if _nan(x) else f"{float(x) * 100:.{CASAS}f}%".replace(".", SEP)


def i0(x) -> str:
    return "—" if _nan(x) else f"{int(round(float(x)))}"


def formatar_df(df: pd.DataFrame) -> pd.DataFrame:
    """Cópia com numéricos como strings formatadas (relatório)."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]) and not pd.api.types.is_bool_dtype(out[c]):
            f = p3 if c in COLS_PCT else i0 if c in COLS_INT else f3
            out[c] = out[c].map(f)
    return out


def arredondar_df(df: pd.DataFrame) -> pd.DataFrame:
    """Arredonda floats a CASAS mantendo tipo numérico (JSON/Excel)."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(CASAS)
    return out


def r3(x):
    return None if _nan(x) else round(float(x), CASAS)
