"""Relatório de documentação do modelo: Jinja2 (Markdown) → HTML (+ PDF via pandoc, se disponível)."""
from __future__ import annotations
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from . import fmt

TEMPLATES = Path(__file__).resolve().parent.parent / "templates"
CSS = """
body{font-family:Georgia,'Times New Roman',serif;max-width:980px;margin:2rem auto;padding:0 1.5rem;line-height:1.5;color:#222}
h1{font-size:1.9rem;border-bottom:2px solid #333;padding-bottom:.3rem}h2{margin-top:2.2rem;border-bottom:1px solid #999}h3{margin-top:1.5rem}
table{border-collapse:collapse;font-size:.85rem;margin:1rem 0;font-family:Arial,Helvetica,sans-serif}th,td{border:1px solid #bbb;padding:3px 8px;text-align:right}
th{background:#f0f0f0}td:first-child,th:first-child{text-align:left}img{max-width:78%;display:block;margin:.6rem auto}
.meta{color:#555;font-size:.9rem}.resumo{background:#f7f7f7;padding:1rem;border-left:4px solid #666}code{font-family:Consolas,monospace;font-size:.9em}
.ref{font-size:.9rem}
"""


def tab(df: pd.DataFrame, index: bool = False, max_rows: int = 200) -> str:
    d = df.head(max_rows)
    d = d.reset_index() if index else d
    d = fmt.formatar_df(d)
    cols = [str(c) for c in d.columns]
    linhas = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        linhas.append("| " + " | ".join(str(x).replace("|", "\\|") for x in r.values) + " |")
    return "\n".join(linhas)


_ACENTOS = {"'": {"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú", "A": "Á", "E": "É", "I": "Í", "O": "Ó", "U": "Ú"},
            "~": {"a": "ã", "o": "õ", "n": "ñ", "A": "Ã", "O": "Õ"}, '"': {"a": "ä", "e": "ë", "o": "ö", "u": "ü", "O": "Ö", "U": "Ü"},
            "^": {"a": "â", "e": "ê", "o": "ô"}, "`": {"a": "à", "e": "è"}, "c": {"c": "ç", "C": "Ç"}}


def _delatex(s: str) -> str:
    s = re.sub(r"\\url\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textsuperscript\{([^}]*)\}", lambda m: {"o": "º", "a": "ª"}.get(m.group(1), m.group(1)), s)
    def acc(m):
        return _ACENTOS.get(m.group(1), {}).get(m.group(2), m.group(2))
    s = re.sub(r"\{?\\(['~\"^`c])\s*\\?([A-Za-z])\}?", acc, s)      # {\'a} \'a {\c c} \~{a}
    s = re.sub(r"\{?\\(['~\"^`c])\{([A-Za-z])\}\}?", acc, s)
    s = re.sub(r"\\(['~\"^`c])\\i\b", lambda m: _ACENTOS[m.group(1)].get("i", "i"), s)
    return s


def referencias(bib_path: Path) -> list:
    """Parser simples de BibTeX: robusto a CRLF, chaves aninhadas e ordem dos campos."""
    txt = bib_path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    refs = []
    for bloco in re.split(r"\n(?=@)", "\n" + txt):
        m = re.match(r"@\w+\{[^,]+,(.*)\}\s*$", bloco.strip(), flags=re.S)
        if not m:
            continue
        corpo, campos, i = m.group(1), {}, 0
        while True:
            mm = re.compile(r"\s*(\w+)\s*=\s*\{").match(corpo, i)
            if not mm:
                break
            nome, j, prof = mm.group(1).lower(), mm.end(), 1
            k = j
            while k < len(corpo) and prof:
                prof += {"{": 1, "}": -1}.get(corpo[k], 0); k += 1
            campos[nome] = corpo[j:k - 1]
            i = k
            while i < len(corpo) and corpo[i] in ", \n\t":
                i += 1
        limpa = lambda s: re.sub(r"\s+", " ", _delatex(s).replace("{", "").replace("}", "").replace("\\", "")).strip()
        autor = limpa(campos.get("author") or campos.get("editor", "")).replace(" and ", "; ")
        titulo = limpa(campos.get("title", ""))
        onde = limpa(campos.get("journal") or campos.get("booktitle") or campos.get("publisher") or campos.get("institution") or campos.get("school") or "")
        vol, num, pg = campos.get("volume", ""), campos.get("number", ""), campos.get("pages", "")
        extra = (f" {vol}" + (f"({num})" if num else "") + (f": {pg}" if pg else "")) if vol else ""
        refs.append(f"{autor} ({campos.get('year', '')}). {titulo}. {onde}{extra}.".replace("..", "."))
    return sorted(refs)


def _md_to_html(md: str, titulo: str) -> str:
    import markdown
    body = markdown.markdown(md, extensions=["tables", "toc", "fenced_code"])
    return f"<!doctype html><html lang='pt-br'><head><meta charset='utf-8'><title>{titulo}</title><style>{CSS}</style></head><body>{body}</body></html>"


def gerar(res: dict, out: Path) -> dict:
    cfg = res["cfg"]; r = res["resumo"]
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), trim_blocks=True, lstrip_blocks=True)
    env.filters["f3"] = fmt.f3; env.filters["p3"] = fmt.p3; env.filters["i0"] = fmt.i0
    env.globals["tab"] = tab
    tpl = env.get_template("relatorio.md.j2")
    desc_vars = {v: cfg["variaveis"][v].get("descricao", "") for v in cfg["variaveis"]}
    sinais = {v: cfg["variaveis"][v].get("sinal_esperado", "") for v in cfg["variaveis"]}
    md = tpl.render(cfg=cfg, r=r, res=res, hoje=date.today().isoformat(), desc_vars=desc_vars, sinais=sinais,
                    refs=referencias(TEMPLATES / "refs.bib"), figs=res["figs"], np=np)
    (out / "relatorio.md").write_text(md, encoding="utf-8")
    html = _md_to_html(md, cfg["relatorio"]["titulo"])
    (out / "relatorio.html").write_text(html, encoding="utf-8")
    pdf = None
    if cfg["relatorio"].get("pdf") in ("auto", True) and shutil.which("pandoc"):
        try:
            subprocess.run(["pandoc", str(out / "relatorio.md"), "-o", str(out / "relatorio.pdf"), "--resource-path", str(out),
                            "-V", "geometry:margin=2.2cm", "--pdf-engine=xelatex"], check=True, capture_output=True, timeout=300)
            pdf = out / "relatorio.pdf"
        except Exception:
            pdf = None
    return {"md": out / "relatorio.md", "html": out / "relatorio.html", "pdf": pdf}
