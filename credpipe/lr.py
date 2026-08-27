"""Regressão logística — backend PyCaret (padrão quando instalado) ou scikit-learn (mesma interface)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config as C
from .metrics import metr, decis, cortes_custo


def _backend(cfg):
    b = cfg["lr"].get("backend", "auto")
    if b in ("auto", "pycaret"):
        try:
            import pycaret.classification  # noqa: F401
            return "pycaret"
        except Exception:
            if b == "pycaret":
                raise
    return "sklearn"


class ModeloLR:
    """Resultado padronizado: predict(df)->P(default); cv; coefs; metricas por amostra."""

    def __init__(self, cfg):
        self.cfg, self.backend = cfg, _backend(cfg)
        self.exp = None          # experimento PyCaret (para desafiantes)
        self.modelo = None
        self.cv = None
        self.coefs = None
        self.intercepto = None
        self.tuning = {}

    # ------------------------------------------------------------------ sklearn
    def _fit_sklearn(self, dev_train, oot):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_validate
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from scipy.stats import loguniform

        num, cat = C.numericas(self.cfg), C.categoricas(self.cfg)
        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="mean")), ("sc", StandardScaler())]), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ])
        pipe = Pipeline([("pre", pre), ("lr", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"))])
        X, y = dev_train[num + cat], dev_train[self.cfg["target"]].values
        skf = StratifiedKFold(self.cfg["lr"]["fold"], shuffle=True, random_state=self.cfg["seed"])
        sc = {"Accuracy": "accuracy", "AUC": "roc_auc", "Recall": "recall", "Prec.": "precision", "F1": "f1"}
        cvr = cross_validate(pipe, X, y, cv=skf, scoring=sc)
        cv = pd.DataFrame({k: cvr[f"test_{k}"] for k in sc}); cv.index.name = "Fold"
        cv.loc["Mean"], cv.loc["Std"] = cv.mean(), cv.std(ddof=0)
        self.cv = cv
        auc_base = float(cv.loc["Mean", "AUC"])
        rs = RandomizedSearchCV(pipe, {"lr__C": loguniform(1e-3, 1e2)}, n_iter=self.cfg["lr"]["tune_iter"], cv=skf,
                                scoring="roc_auc", random_state=self.cfg["seed"], n_jobs=1)
        rs.fit(X, y)
        if rs.best_score_ > auc_base + 1e-4:
            self.modelo, self.tuning = rs.best_estimator_, {"C": float(rs.best_params_["lr__C"]), "auc_cv": float(rs.best_score_), "manteve_original": False}
        else:
            self.modelo, self.tuning = pipe.fit(X, y), {"C": 1.0, "auc_cv": auc_base, "manteve_original": True}
        names = list(self.modelo.named_steps["pre"].get_feature_names_out())
        names = [n.split("__", 1)[1] for n in names]
        lr = self.modelo.named_steps["lr"]
        self.coefs = pd.DataFrame({"variavel": names, "coef": lr.coef_[0]})
        self.intercepto = float(lr.intercept_[0])

    def _predict_sklearn(self, df):
        num, cat = C.numericas(self.cfg), C.categoricas(self.cfg)
        return self.modelo.predict_proba(df[num + cat])[:, 1]

    # ------------------------------------------------------------------ pycaret
    def _fit_pycaret(self, dev_train, oot):
        from pycaret.classification import ClassificationExperiment
        num, cat = C.numericas(self.cfg), C.categoricas(self.cfg)
        cols = num + cat + [self.cfg["target"]]
        exp = ClassificationExperiment()
        exp.setup(data=dev_train[cols], test_data=oot[cols], target=self.cfg["target"],
                  categorical_features=cat, numeric_features=num, normalize=True, normalize_method="zscore",
                  fold=self.cfg["lr"]["fold"], session_id=self.cfg["seed"], verbose=False, html=False)
        base = exp.create_model("lr", verbose=False)
        cv_base = exp.pull()
        tuned = exp.tune_model(base, optimize="AUC", n_iter=self.cfg["lr"]["tune_iter"], verbose=False)
        cv_tuned = exp.pull()
        manteve = tuned is base or float(cv_tuned.loc["Mean", "AUC"]) <= float(cv_base.loc["Mean", "AUC"]) + 1e-4
        self.modelo = base if manteve else tuned
        self.cv = cv_base if manteve else cv_tuned
        self.exp = exp
        self.tuning = {"C": float(getattr(self.modelo, "C", np.nan)), "auc_cv": float(self.cv.loc["Mean", "AUC"]), "manteve_original": bool(manteve)}
        names = list(exp.get_config("X_train_transformed").columns)
        self.coefs = pd.DataFrame({"variavel": names, "coef": self.modelo.coef_[0]})
        self.intercepto = float(self.modelo.intercept_[0])

    def _predict_pycaret(self, df):
        num, cat = C.numericas(self.cfg), C.categoricas(self.cfg)
        out = self.exp.predict_model(self.modelo, data=df[num + cat], raw_score=True, verbose=False)
        return out["prediction_score_1"].values

    # ------------------------------------------------------------------ interface
    def fit(self, dev_train, oot):
        (self._fit_pycaret if self.backend == "pycaret" else self._fit_sklearn)(dev_train, oot)
        self.coefs["odds_ratio"] = np.exp(self.coefs["coef"])
        self.coefs = self.coefs.reindex(self.coefs["coef"].abs().sort_values(ascending=False).index).reset_index(drop=True)
        return self

    def predict(self, df):
        return (self._predict_pycaret if self.backend == "pycaret" else self._predict_sklearn)(df)


def ajustar(dev_train, oos, oot, cfg) -> dict:
    m = ModeloLR(cfg).fit(dev_train, oot)
    tgt = cfg["target"]
    p = {k: m.predict(d) for k, d in (("dev_train", dev_train), ("oos", oos), ("oot", oot))}
    met = {k: metr(d[tgt], p[k]) for k, d in (("dev_train", dev_train), ("oos", oos), ("oot", oot))}
    fn, fp = cfg["custo_fn_fp"]
    return {"modelo": m, "backend": m.backend, "cv": m.cv, "tuning": m.tuning, "coefs": m.coefs, "intercepto": m.intercepto,
            "p": p, "metricas": met,
            "decis_oos": decis(oos[tgt], p["oos"]), "decis_oot": decis(oot[tgt], p["oot"]),
            "cortes": cortes_custo(oos[tgt], p["oos"], fn, fp)}
