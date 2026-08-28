"""Orquestração: base + layout → LR → escala → scorecard refinado → estabilidade → GH → calibração → desafiantes → relatório."""
from __future__ import annotations
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
from . import config as C, fmt, ingest, split, lr as LR, scale as SCALE, scorecard as SC, refine, stability, grades, calibration, challengers, plots, report
from .metrics import metr


def _log(msg, verbose):
    if verbose:
        print(f"[credpipe] {msg}", flush=True)


def run(data_path: str, layout_path: str, out_dir: str, verbose: bool = True) -> dict:
    t0 = time.time()
    out = Path(out_dir); figs = out / "figs"; out.mkdir(parents=True, exist_ok=True)
    cfg = C.load_layout(layout_path)
    fmt.configurar(cfg["formato"]["casas"], cfg["formato"]["sep_decimal"])
    tgt, saf = cfg["target"], cfg["safra"]
    V = C.variaveis(cfg)

    # 1) ingest + split
    df = ingest.validar(ingest.ler(data_path), cfg)
    sumario = ingest.sumario(df, cfg)
    sp = split.dividir(df, cfg)
    dtr, oos, oot = sp["dev_train"], sp["oos"], sp["oot"]
    _log(f"base {len(df)} obs | dev_train {len(dtr)} | OOS {len(oos)} | OOT {len(oot)} (safras {sp['safras_oot'][0]}–{sp['safras_oot'][-1]})", verbose)

    # 2) LR
    lr_res = LR.ajustar(dtr, oos, oot, cfg)
    _log(f"LR ({lr_res['backend']}): AUC OOS {lr_res['metricas']['oos']['AUC']:.3f} | OOT {lr_res['metricas']['oot']['AUC']:.3f}", verbose)

    # 3) escala
    esc = SCALE.Escala.ajustar(lr_res["p"]["dev_train"], cfg["escala"])

    # 4) scorecard + refino
    Xd, yd, Xo, yo = dtr[V], dtr[tgt].values, oot[V], oot[tgt].values
    spec0 = SC.spec_inicial(Xd, cfg)
    sc0 = SC.Scorecard.ajustar(Xd, yd, spec0, esc)
    spec, log_ref, betas, status_ref = refine.refinar(spec0, Xd, yd, Xo, yo, df[V], df[tgt].values, df[saf].values, cfg)
    if not spec:
        raise RuntimeError(f"refino descartou todas as variáveis: {status_ref}")
    sc = SC.Scorecard.ajustar(Xd, yd, spec, esc)
    s = {k: sc.score(d) for k, d in (("dev_train", dtr), ("oos", oos), ("oot", oot))}
    met_sc = {k: metr(d[tgt], -s[k]) for k, d in (("dev_train", dtr), ("oos", oos), ("oot", oot))}
    met_sc0 = {k: metr(d[tgt], -sc0.score(d)) for k, d in (("oos", oos), ("oot", oot))}
    _log(f"scorecard: {status_ref}; {len(spec)} variáveis, {len(sc.tabela)} faixas; AUC OOS {met_sc['oos']['AUC']:.3f} | OOT {met_sc['oot']['AUC']:.3f}", verbose)

    # 5) estabilidade
    df_val = pd.concat([oos.assign(amostra="OOS", p=lr_res["p"]["oos"], score=s["oos"].values),
                        oot.assign(amostra="OOT", p=lr_res["p"]["oot"], score=s["oot"].values)])
    ps = stability.por_safra(df_val, cfg)
    comp_var, res_var = stability.por_variavel(spec, Xd, yd, Xo, yo)

    # 6) GH + calibração
    edges, log_gh, status_gh = grades.malha_recursiva(s["dev_train"], yd, dtr[saf].values, s["oot"], yo, oot[saf].values, cfg["gh"])
    ordem = {nome: grades.verificar_ordem(s[k], d[tgt].values, d[saf].values, edges, cfg["gh"])
             for nome, k, d in (("dev", "dev_train", dtr), ("OOT", "oot", oot))}
    mapa = calibration.mapa_gh(edges, s["dev_train"], yd, s["oos"], oos[tgt].values, s["oot"], yo, esc.score_para_prob(s["oot"]))
    dp = calibration.de_para(mapa, edges, cfg["calibracao"]["moc"], esc.lo, esc.hi)
    escorador = calibration.Escorador(sc, edges, dp)
    _log(f"GH: {status_gh}; {len(edges)-1} grupos; HHI {calibration.hhi(mapa):.3f}", verbose)

    # 7) desafiantes
    ch = None
    if cfg["desafiantes"]["ativo"]:
        ch = challengers.rodar(lr_res, dtr, oos, oot, cfg)
        _log("desafiantes: " + ", ".join(f"{r.modelo} AUC OOT {r.AUC_OOT:.3f}" for r in ch["resultados"].itertuples()), verbose)

    # 8) figuras
    F = {}
    F["roc"] = plots.roc({"LR OOS": oos[tgt].values, "LR OOT": yo, "Scorecard OOT": yo},
                         {"a": lr_res["p"]["oos"], "b": lr_res["p"]["oot"], "c": -s["oot"].values}, figs)
    F["calibracao"] = plots.calibracao(yo, lr_res["p"]["oot"], figs)
    F["decis"] = plots.decis(lr_res["decis_oot"], figs)
    F["score_dist"] = plots.score_dist(s["oot"], yo, esc.lo, esc.hi, figs)
    F["por_safra"] = plots.por_safra(ps, sp["safras_oot"], figs)
    if not betas.empty:
        F["betas"] = plots.betas(betas, figs)
    s_all = sc.score(df)
    F["painel_gh"] = plots.painel_gh(s_all, df[tgt].values, df[saf].values, edges, cfg["gh"], mapa, s["oos"], oos[tgt].values, sp["safras_oot"], figs)
    F["calib_oos_oot"] = plots.calib_oos_vs_oot(mapa, figs)
    if ch is not None and len(ch["resultados"]):
        F["desafiantes"] = plots.desafiantes(ch["resultados"], lr_res["metricas"]["oos"]["AUC"], lr_res["metricas"]["oot"]["AUC"],
                                             met_sc["oos"]["AUC"], met_sc["oot"]["AUC"], figs)

    # 9) resumo + artefatos
    resumo = {
        "n_base": int(len(df)), "taxa_default_base": float(df[tgt].mean()),
        "n_dev_train": int(len(dtr)), "n_oos": int(len(oos)), "n_oot": int(len(oot)), "safras_oot": sp["safras_oot"],
        "backend_lr": lr_res["backend"], "tuning": lr_res["tuning"], "escala": esc.to_dict(),
        "n_variaveis": len(spec), "variaveis": list(spec), "n_faixas": int(len(sc.tabela)), "n_faixas_inicial": int(len(sc0.tabela)),
        "status_refino": status_ref, "k_gh": int(len(edges) - 1), "cortes_gh": [float(e) for e in edges[1:-1]], "status_gh": status_gh,
        "hhi": calibration.hhi(mapa), "csi_max": float(res_var["CSI"].max()),
        "ordem_gh_ok": {k: bool(v.all()) for k, v in ordem.items()},
        "metricas_lr": lr_res["metricas"], "metricas_scorecard": met_sc, "metricas_scorecard_inicial": met_sc0,
        "cortes": lr_res["cortes"], "tempo_s": round(time.time() - t0, 1),
    }
    res = {"cfg": cfg, "sumario": sumario, "split": sp, "lr": lr_res, "escala": esc, "spec0": spec0, "scorecard_inicial": sc0,
           "spec": spec, "log_refino": log_ref, "betas": betas, "scorecard": sc, "scores": s, "metricas_scorecard": met_sc,
           "por_safra": ps, "comp_var": comp_var, "res_var": res_var, "edges": edges, "log_gh": log_gh, "ordem_gh": ordem,
           "mapa_gh": mapa, "de_para": dp, "escorador": escorador, "desafiantes": ch, "figs": F, "resumo": resumo}
    _salvar(res, out)
    report.gerar(res, out)
    _log(f"concluído em {resumo['tempo_s']} s → {out}", verbose)
    return res


def _salvar(res, out: Path):
    r = res["resumo"]; cfg = res["cfg"]
    (out / "resultados.json").write_text(json.dumps(_jsonable(r), ensure_ascii=False, indent=2), encoding="utf-8")
    res["scorecard"].tabela.pipe(fmt.arredondar_df).to_csv(out / "scorecard_final.csv", index=False)
    res["de_para"].pipe(fmt.arredondar_df).to_csv(out / "de_para_score_GH_PD.csv")
    with pd.ExcelWriter(out / "modelo_scorecard_final.xlsx") as w:
        pd.Series(_flat(r)).to_frame("valor").to_excel(w, sheet_name="resumo")
        res["scorecard"].tabela.pipe(fmt.arredondar_df).to_excel(w, sheet_name="scorecard", index=False)
        res["de_para"].pipe(fmt.arredondar_df).to_excel(w, sheet_name="de_para_score_GH_PD")
        res["mapa_gh"].pipe(fmt.arredondar_df).to_excel(w, sheet_name="mapa_gh_completo")
        res["lr"]["coefs"].pipe(fmt.arredondar_df).to_excel(w, sheet_name="coefs_lr", index=False)
        res["lr"]["cv"].pipe(fmt.arredondar_df).to_excel(w, sheet_name="cv_lr")
        res["lr"]["decis_oot"].pipe(fmt.arredondar_df).to_excel(w, sheet_name="decis_oot", index=False)
        res["por_safra"].pipe(fmt.arredondar_df).to_excel(w, sheet_name="por_safra", index=False)
        res["comp_var"].pipe(fmt.arredondar_df).to_excel(w, sheet_name="estab_faixas", index=False)
        res["res_var"].pipe(fmt.arredondar_df).to_excel(w, sheet_name="estab_variaveis")
        (res["log_refino"] if len(res["log_refino"]) else pd.DataFrame({"acao": ["nenhuma"]})).to_excel(w, sheet_name="log_refino", index=False)
        (res["log_gh"] if len(res["log_gh"]) else pd.DataFrame({"acao": ["nenhuma"]})).to_excel(w, sheet_name="log_malha_gh", index=False)
        if not res["betas"].empty:
            res["betas"].pipe(fmt.arredondar_df).to_excel(w, sheet_name="betas_janela")
        if res["desafiantes"] is not None:
            res["desafiantes"]["resultados"].pipe(fmt.arredondar_df).to_excel(w, sheet_name="desafiantes", index=False)
        pd.DataFrame({"edges": res["edges"]}).to_excel(w, sheet_name="cortes_gh", index=False)
    with open(out / "escoragem_novos_clientes.pkl", "wb") as f:
        pickle.dump(res["escorador"], f)
    if res["lr"]["backend"] == "sklearn":
        with open(out / "modelo_lr.pkl", "wb") as f:
            pickle.dump(res["lr"]["modelo"].modelo, f)
    else:
        try:
            res["lr"]["modelo"].exp.save_model(res["lr"]["modelo"].modelo, str(out / "modelo_lr"), verbose=False)
        except Exception:
            pass


def _flat(d, pref=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{pref}{k}."))
        else:
            out[f"{pref}{k}"] = str(v) if isinstance(v, (list, tuple)) else v
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return fmt.r3(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o