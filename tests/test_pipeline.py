import json
from pathlib import Path
import yaml
from credpipe import synth
from credpipe.pipeline import run


def test_run_completo(tmp_path):
    base = tmp_path / "base.csv"; synth.gerar(n=6000, seed=3).to_csv(base, index=False)
    lay = yaml.safe_load(Path("examples/layout.yaml").read_text(encoding="utf-8"))
    lay["desafiantes"] = {"ativo": True, "top_k": 1, "tune_iter": 2}; lay["lr"]["tune_iter"] = 3
    layout = tmp_path / "layout.yaml"; layout.write_text(yaml.safe_dump(lay, allow_unicode=True), encoding="utf-8")
    out = tmp_path / "run"
    res = run(str(base), str(layout), str(out), verbose=False)
    for f in ("relatorio.html", "relatorio.md", "modelo_scorecard_final.xlsx", "de_para_score_GH_PD.csv", "scorecard_final.csv", "escorador.pkl", "resultados.json", "figs/painel_gh.png"):
        assert (out / f).exists(), f
    r = json.loads((out / "resultados.json").read_text(encoding="utf-8"))
    assert r["metricas_scorecard"]["oot"]["AUC"] > 0.6 and r["k_gh"] >= 3
    html = (out / "relatorio.html").read_text(encoding="utf-8")
    assert "De-para" in html and "Referências" in html
