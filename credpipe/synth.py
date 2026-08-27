"""Base simulada com DGP conhecido (célula 4 + safra) — testes e exemplo."""
from __future__ import annotations
import numpy as np
import pandas as pd


def gerar(n: int = 10_000, seed: int = 42, safra_ini: str = "2023-01", n_safras: int = 24, drift: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idade = rng.integers(21, 70, n)
    renda = rng.lognormal(8.2, 0.5, n).round(0)
    tempo_emprego = rng.exponential(5, n).round(1)
    utilizacao = rng.beta(2, 3, n).round(3)
    n_atrasos_12m = rng.poisson(0.4, n)
    dti = rng.beta(2, 5, n).round(3)
    consultas_6m = rng.poisson(1.2, n)
    tipo_ocupacao = rng.choice(["CLT", "autonomo", "servidor", "aposentado"], n, p=[.50, .25, .15, .10])
    possui_imovel = rng.integers(0, 2, n)
    efeito = pd.Series(tipo_ocupacao).map({"CLT": 0.0, "autonomo": 0.4, "servidor": -0.5, "aposentado": -0.3}).values
    logit = (-3.0 - 0.02 * (idade - 40) - 0.40 * np.log(renda / 3600) - 0.05 * tempo_emprego
             + 2.00 * utilizacao + 0.60 * n_atrasos_12m + 2.50 * dti + 0.25 * consultas_6m
             - 0.50 * possui_imovel + efeito + rng.normal(0, 0.3, n))
    default = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    safras = pd.period_range(safra_ini, periods=n_safras, freq="M").strftime("%Y%m")
    df = pd.DataFrame({
        "id": np.arange(1, n + 1), "safra": rng.choice(safras, size=n).astype(int),
        "idade": idade, "renda": renda, "tempo_emprego": tempo_emprego, "utilizacao": utilizacao,
        "n_atrasos_12m": n_atrasos_12m, "dti": dti, "consultas_6m": consultas_6m,
        "tipo_ocupacao": tipo_ocupacao, "possui_imovel": possui_imovel, "default": default,
    })
    if drift:
        m = df["safra"] >= int(safras[len(safras) * 3 // 4])
        df.loc[m, "utilizacao"] = (df.loc[m, "utilizacao"] * 1.25).clip(upper=1)
    return df
