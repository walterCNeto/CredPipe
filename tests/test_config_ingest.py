import pytest
import pandas as pd
from credpipe import config, ingest, synth, scale


def test_layout_faltando_chave():
    with pytest.raises(ValueError, match="obrigatória"):
        config.validar({"id": "id", "safra": "safra", "target": "y"})


def test_bins_invalidos():
    with pytest.raises(ValueError, match="bins"):
        config.validar({"id": "id", "safra": "s", "target": "y", "variaveis": {"x": {"tipo": "num", "bins": [0, 1, 2]}}})


def test_ingest_id_duplicado(cfg):
    df = synth.gerar(200); df.loc[1, "id"] = df.loc[0, "id"]
    with pytest.raises(ValueError, match="duplicatas"):
        ingest.validar(df, cfg)


def test_ingest_target_nao_binario(cfg):
    df = synth.gerar(200); df.loc[0, "default"] = 2
    with pytest.raises(ValueError, match="0/1"):
        ingest.validar(df, cfg)


def test_ingest_safra_invalida(cfg):
    df = synth.gerar(200); df["safra"] = df["safra"].astype(str); df.loc[0, "safra"] = "2023-13"
    with pytest.raises(ValueError, match="YYYYMM"):
        ingest.validar(df, cfg)


def test_split_oot(base, cfg):
    df, sp = base
    assert len(sp["safras_oot"]) == cfg["oot"]["n_safras"]
    assert set(sp["oot"][cfg["safra"]]) == set(sp["safras_oot"])
    assert len(sp["dev_train"]) + len(sp["oos"]) + len(sp["oot"]) == len(df)


def test_escala_roundtrip():
    import numpy as np
    e = scale.Escala.ajustar(None, {"modo": "pdo", "pdo": 40, "score0": 600, "odds0": 20, "lo": 100, "hi": 900})
    assert e.prob_para_score([1 / 21])[0] == 600
    p = np.array([0.05, 0.2, 0.5]); s = e.offset + e.factor * np.log((1 - p) / p)
    assert np.allclose(e.score_para_prob(s), p)
