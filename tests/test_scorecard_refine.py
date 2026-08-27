import numpy as np
from credpipe import config, scale, scorecard as SC, refine, stability
from credpipe.metrics import metr


def _prep(cfg, base):
    df, sp = base
    V, tgt, saf = config.variaveis(cfg), cfg["target"], cfg["safra"]
    return df, sp, V, tgt, saf, sp["dev_train"][V], sp["dev_train"][tgt].values, sp["oot"][V], sp["oot"][tgt].values


def test_refino_converge_e_monotonico(cfg, base):
    df, sp, V, tgt, saf, Xd, yd, Xo, yo = _prep(cfg, base)
    spec, log, betas, status = refine.refinar(SC.spec_inicial(Xd, cfg), Xd, yd, Xo, yo, df[V], df[tgt].values, df[saf].values, cfg)
    assert status.startswith("convergiu")
    assert len(spec) >= 7                                   # base estacionária: não descarta
    esc = scale.Escala.ajustar(None, cfg["escala"] | {"modo": "pdo"})
    sc = SC.Scorecard.ajustar(Xd, yd, spec, esc)
    for v, g in sc.tabela.groupby("variavel"):
        if spec[v]["tipo"] == "num":
            w = g[g["faixa"] != SC.MISSING]["woe"].values
            assert np.all(np.diff(w) >= 0) or np.all(np.diff(w) <= 0)
    assert (betas.drop(columns="dev", errors="ignore") <= cfg["restricoes"]["beta_tol"]).all().all()
    m = metr(yo, -sc.score(sp["oot"]))
    assert m["AUC"] > 0.65


def test_drift_acusado(cfg, base_drift):
    df, sp, V, tgt, saf, Xd, yd, Xo, yo = _prep(cfg, base_drift)
    comp, res = stability.por_variavel(SC.spec_inicial(Xd, cfg), Xd, yd, Xo, yo)
    assert res["CSI"].idxmax() == "utilizacao" and res.loc["utilizacao", "CSI"] > cfg["restricoes"]["csi_max"]
    spec, log, betas, status = refine.refinar(SC.spec_inicial(Xd, cfg), Xd, yd, Xo, yo, df[V], df[tgt].values, df[saf].values, cfg)
    assert "utilizacao" in set(log["variavel"])              # refino age sobre a variável com drift


def test_missing_vira_faixa(cfg, base):
    df, sp, V, tgt, saf, Xd, yd, Xo, yo = _prep(cfg, base)
    Xd2 = Xd.copy(); Xd2.loc[Xd2.index[:300], "renda"] = np.nan
    spec = SC.spec_inicial(Xd2, cfg); Cb = SC.binar(Xd2, spec)
    assert SC.MISSING in list(Cb["renda"].cat.categories)
