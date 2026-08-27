"""Modelos desafiantes (ML) no mesmo split — comparação com a LR e com o scorecard em OOS/OOT."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config as C
from .metrics import metr


def _sklearn(dev_train, oos, oot, cfg, k):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.inspection import permutation_importance

    num, cat = C.numericas(cfg), C.categoricas(cfg)
    tgt = cfg["target"]
    pre = ColumnTransformer([("num", SimpleImputer(strategy="mean"), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
    seed = cfg["seed"]
    cands = {
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
        "Hist. Gradient Boosting": HistGradientBoostingClassifier(random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=seed, n_jobs=1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=20, random_state=seed, n_jobs=1),
    }
    X, y = dev_train[num + cat], dev_train[tgt].values
    skf = StratifiedKFold(cfg["lr"]["fold"], shuffle=True, random_state=seed)
    linhas, pipes = [], {}
    for nome, est in cands.items():
        pipe = Pipeline([("pre", pre), ("m", est)])
        auc = cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc")
        linhas.append({"modelo": nome, "AUC_cv": float(auc.mean()), "AUC_cv_std": float(auc.std())})
        pipes[nome] = pipe
    comp = pd.DataFrame(linhas).sort_values("AUC_cv", ascending=False).reset_index(drop=True)
    resultados, importancias, preds = [], {}, {}
    for nome in comp["modelo"].head(k):
        pipe = pipes[nome].fit(X, y)
        p = {a: pipe.predict_proba(d[num + cat])[:, 1] for a, d in (("oos", oos), ("oot", oot))}
        preds[nome] = p
        resultados.append({"modelo": nome, **{f"{m}_OOS": v for m, v in metr(oos[tgt], p["oos"]).items()},
                           **{f"{m}_OOT": v for m, v in metr(oot[tgt], p["oot"]).items()}})
        pi = permutation_importance(pipe, oos[num + cat], oos[tgt].values, scoring="roc_auc", n_repeats=5, random_state=seed)
        importancias[nome] = pd.Series(pi.importances_mean, index=num + cat).sort_values(ascending=False)
    return comp, pd.DataFrame(resultados), importancias, preds


def _pycaret(exp, dev_train, oos, oot, cfg, k):
    num, cat = C.numericas(cfg), C.categoricas(cfg)
    tgt = cfg["target"]
    exp.compare_models(sort="AUC", n_select=len(exp.models()), exclude=["lr", "dummy"], verbose=False)
    comp_full = exp.pull().reset_index().rename(columns={"index": "id", "Model": "modelo", "AUC": "AUC_cv"})
    comp = comp_full[["modelo", "AUC_cv", "id"]].head(k)
    resultados, importancias, preds = [], {}, {}
    for _, r in comp.iterrows():
        m = exp.create_model(r["id"], verbose=False)
        try:
            m = exp.tune_model(m, optimize="AUC", n_iter=cfg["desafiantes"]["tune_iter"], verbose=False)
        except Exception:
            pass
        p = {a: exp.predict_model(m, data=d[num + cat], raw_score=True, verbose=False)["prediction_score_1"].values
             for a, d in (("oos", oos), ("oot", oot))}
        preds[r["modelo"]] = p
        resultados.append({"modelo": r["modelo"], **{f"{mm}_OOS": v for mm, v in metr(oos[tgt], p["oos"]).items()},
                           **{f"{mm}_OOT": v for mm, v in metr(oot[tgt], p["oot"]).items()}})
        fi = getattr(m, "feature_importances_", None)
        if fi is not None:
            importancias[r["modelo"]] = pd.Series(fi, index=exp.get_config("X_train_transformed").columns).sort_values(ascending=False)
    return comp_full[["modelo", "AUC_cv"]], pd.DataFrame(resultados), importancias, preds


def rodar(lr_res: dict, dev_train, oos, oot, cfg) -> dict:
    k = cfg["desafiantes"]["top_k"]
    if lr_res["backend"] == "pycaret" and lr_res["modelo"].exp is not None:
        comp, res, imp, preds = _pycaret(lr_res["modelo"].exp, dev_train, oos, oot, cfg, k)
    else:
        comp, res, imp, preds = _sklearn(dev_train, oos, oot, cfg, k)
    return {"comparacao_cv": comp, "resultados": res, "importancias": imp, "preds": preds}
