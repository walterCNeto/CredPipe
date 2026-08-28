"""Interface Streamlit do credpipe: sobe base + layout, roda o pipeline, exibe o relatório e baixa os artefatos.
Uso:  streamlit run app.py
"""
from __future__ import annotations
import io
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from credpipe import config, ingest, synth
from credpipe.pipeline import run

st.set_page_config(page_title="credpipe — modelo de PD", layout="wide", initial_sidebar_state="expanded")
st.title("credpipe — modelo de PD: LR, scorecard, GH e calibração")

st.caption("📱 No celular: se a tela estiver vazia, toque na seta ‹›  no canto superior esquerdo para abrir o painel com upload, layout e o botão Rodar.")
if "base_df" not in st.session_state:
    if st.button("▶ Rodar demonstração com base simulada", type="primary"):
        st.session_state["base_df"] = synth.gerar()
        st.rerun()

EXEMPLO_LAYOUT = Path(__file__).parent / "examples" / "layout.yaml"

with st.sidebar:
    st.header("1. Base de modelagem")
    up = st.file_uploader("CSV / Parquet / XLSX (id, safra YYYYMM, target 0/1, explicativas)", type=["csv", "parquet", "xlsx"])
    if st.button("Usar base simulada de exemplo"):
        st.session_state["base_df"] = synth.gerar()
    st.header("2. Layout")
    lay_up = st.file_uploader("layout.yaml", type=["yaml", "yml"])
    texto_padrao = EXEMPLO_LAYOUT.read_text(encoding="utf-8") if EXEMPLO_LAYOUT.exists() else ""
    lay_txt = st.text_area("ou edite aqui", value=(lay_up.read().decode("utf-8") if lay_up else texto_padrao), height=380)
    rodar = st.button("3. Rodar pipeline", type="primary")

if up is not None:
    suf = Path(up.name).suffix.lower()
    st.session_state["base_df"] = (pd.read_parquet(up) if suf == ".parquet" else pd.read_excel(up) if suf == ".xlsx"
                                   else pd.read_csv(up, sep=None, engine="python"))

df = st.session_state.get("base_df")
if df is not None:
    st.subheader("Prévia da base")
    st.write(f"{len(df):,} linhas × {df.shape[1]} colunas".replace(",", "."))
    st.dataframe(df.head(20), use_container_width=720)
    try:
        cfg = config.validar(yaml.safe_load(lay_txt))
        dfv = ingest.validar(df, cfg)
        s = ingest.sumario(dfv, cfg)
        st.success(f"Layout OK — taxa de default {s['taxa_default']:.3%} | safras {s['safras'][0]} a {s['safras'][-1]} ({len(s['safras'])})")
        st.dataframe(s["por_safra"].style.format({"taxa_default": "{:.3%}"}), use_container_width=True, height=220)
    except Exception as e:
        st.error(f"Base × layout: {e}")
        rodar = False

if rodar and df is not None:
    work = Path(tempfile.mkdtemp(prefix="credpipe_"))
    base_path, lay_path, out = work / "base.csv", work / "layout.yaml", work / "run"
    df.to_csv(base_path, index=False); lay_path.write_text(lay_txt, encoding="utf-8")
    log = st.empty(); linhas = []

    import builtins
    _print = builtins.print

    def _cap(*a, **k):
        linhas.append(" ".join(str(x) for x in a)); log.code("\n".join(linhas[-12:]))
    builtins.print = _cap
    try:
        with st.spinner("Rodando: LR → escala → scorecard/refino → estabilidade → GH → calibração → desafiantes → relatório"):
            res = run(str(base_path), str(lay_path), str(out), verbose=True)
    except Exception as e:
        builtins.print = _print
        st.exception(e); st.stop()
    builtins.print = _print
    st.session_state["run_dir"] = out
    st.session_state["res_resumo"] = res["resumo"]

out = st.session_state.get("run_dir")
if out and Path(out).exists():
    r = st.session_state["res_resumo"]
    st.subheader("Resultado")
    c = st.columns(5)
    c[0].metric("AUC scorecard OOS", f"{r['metricas_scorecard']['oos']['AUC']:.3f}")
    c[1].metric("AUC scorecard OOT", f"{r['metricas_scorecard']['oot']['AUC']:.3f}")
    c[2].metric("KS OOT", f"{r['metricas_scorecard']['oot']['KS']:.3f}")
    c[3].metric("Variáveis / faixas", f"{r['n_variaveis']} / {r['n_faixas']}")
    c[4].metric("GHs", f"{r['k_gh']}")

    tabs = st.tabs(["Relatório", "Scorecard", "De-para Score→GH→PD", "Mapa GH", "Figuras", "Downloads"])
    with tabs[0]:
        st.components.v1.html((Path(out) / "relatorio.html").read_text(encoding="utf-8"), height=900, scrolling=True)
    with tabs[1]:
        st.dataframe(pd.read_csv(Path(out) / "scorecard_final.csv"), use_container_width=True, height=600)
    with tabs[2]:
        st.dataframe(pd.read_csv(Path(out) / "de_para_score_GH_PD.csv"), use_container_width=True)
    with tabs[3]:
        st.dataframe(pd.read_excel(Path(out) / "modelo_scorecard_final.xlsx", sheet_name="mapa_gh_completo"), use_container_width=True)
    with tabs[4]:
        for png in sorted((Path(out) / "figs").glob("*.png")):
            st.image(str(png), caption=png.stem, use_container_width=True)
    with tabs[5]:
        st.download_button("modelo_scorecard_final.xlsx", (Path(out) / "modelo_scorecard_final.xlsx").read_bytes(), "modelo_scorecard_final.xlsx")
        st.download_button("relatorio.html", (Path(out) / "relatorio.html").read_bytes(), "relatorio.html")
        st.download_button("escorador.pkl", (Path(out) / "escorador.pkl").read_bytes(), "escorador.pkl")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in Path(out).rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(Path(out)))
        st.download_button("run completo (.zip)", buf.getvalue(), f"credpipe_run_{datetime.now():%Y%m%d_%H%M}.zip")

    st.subheader("Escorar nova base com o modelo desta rodada")
    novo = st.file_uploader("CSV com as mesmas explicativas", type=["csv"], key="novo")
    if novo is not None:
        import pickle
        esc = pickle.load(open(Path(out) / "escorador.pkl", "rb"))
        dn = pd.read_csv(novo, sep=None, engine="python")
        sc = pd.concat([dn, esc(dn)], axis=1)
        st.dataframe(sc.head(50), use_container_width=True)
        st.download_button("baixar escorado.csv", sc.to_csv(index=False).encode("utf-8"), "escorado.csv")
