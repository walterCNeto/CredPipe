import numpy as np
from credpipe import config, scale, scorecard as SC, refine, grades, calibration


def test_malha_e_de_para(cfg, base):
    df, sp = base
    V, tgt, saf = config.variaveis(cfg), cfg["target"], cfg["safra"]
    Xd, yd, Xo, yo = sp["dev_train"][V], sp["dev_train"][tgt].values, sp["oot"][V], sp["oot"][tgt].values
    spec, *_ = refine.refinar(SC.spec_inicial(Xd, cfg), Xd, yd, Xo, yo, df[V], df[tgt].values, df[saf].values, cfg)
    esc = scale.Escala.ajustar(None, cfg["escala"] | {"modo": "pdo"})
    sc = SC.Scorecard.ajustar(Xd, yd, spec, esc)
    sd, so, ss = sc.score(sp["dev_train"]), sc.score(sp["oot"]), sc.score(sp["oos"])
    R = cfg["gh"]
    edges, log, status = grades.malha_recursiva(sd, yd, sp["dev_train"][saf].values, so, yo, sp["oot"][saf].values, R)
    assert status == "convergiu"
    t = grades.tab_gh(sd, yd, edges)
    assert np.all(np.diff(t["taxa"].values) <= 0) and (t["defaults"] >= R["min_def_gh"]).all()
    assert grades.verificar_ordem(so, yo, sp["oot"][saf].values, edges, R).all()
    m = calibration.mapa_gh(edges, sd, yd, ss, sp["oos"][tgt].values, so, yo, esc.score_para_prob(so))
    assert abs((m["PD_calibrada"] * m["pct_carteira_oot"]).sum() - yo.mean()) < 1e-9
    dp = calibration.de_para(m, edges, "IC95_sup")
    assert dp["score_min"].iloc[0] == 100 and dp["score_max"].iloc[-1] == 900
    assert (dp["score_min"].values[1:] == dp["score_max"].values[:-1] + 1).all()     # cortes contíguos
    esc_ = calibration.Escorador(sc, edges, dp)
    out = esc_(sp["oot"].head(50))
    assert set(out.columns) == {"score", "GH", "PD_12m", "PD_12m_conservadora"} and out["GH"].between(1, len(edges) - 1).all()


def test_estrito_colapsa_menos_que_min_gh_nao_quebra(cfg, base):
    df, sp = base
    V, tgt, saf = config.variaveis(cfg), cfg["target"], cfg["safra"]
    esc = scale.Escala.ajustar(None, cfg["escala"] | {"modo": "pdo"})
    sc = SC.Scorecard.ajustar(sp["dev_train"][V], sp["dev_train"][tgt].values, SC.spec_inicial(sp["dev_train"][V], cfg), esc)
    R = dict(cfg["gh"], alpha_cruz=None, janela="M", min_gh=3)
    edges, log, status = grades.malha_recursiva(sc.score(sp["dev_train"]), sp["dev_train"][tgt].values, sp["dev_train"][saf].values,
                                                sc.score(sp["oot"]), sp["oot"][tgt].values, sp["oot"][saf].values, R)
    assert len(edges) - 1 >= 3
