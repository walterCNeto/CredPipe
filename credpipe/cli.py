"""credpipe run --data base.csv --layout layout.yaml --out runs/x  |  credpipe synth --out base.csv  |  credpipe score --run runs/x --data novos.csv"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="credpipe")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--data", required=True); r.add_argument("--layout", required=True); r.add_argument("--out", required=True); r.add_argument("--quiet", action="store_true")
    s = sub.add_parser("synth"); s.add_argument("--out", required=True); s.add_argument("--n", type=int, default=10_000); s.add_argument("--seed", type=int, default=42); s.add_argument("--drift", action="store_true")
    e = sub.add_parser("score"); e.add_argument("--run", required=True); e.add_argument("--data", required=True); e.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        from .pipeline import run
        run(a.data, a.layout, a.out, verbose=not a.quiet)
    elif a.cmd == "synth":
        from .synth import gerar
        gerar(a.n, a.seed, drift=a.drift).to_csv(a.out, index=False); print(f"gerado {a.out}")
    elif a.cmd == "score":
        from .ingest import ler
        esc = pickle.load(open(Path(a.run) / "escoragem_novos_clientes.pkl", "rb"))
        esc(ler(a.data)).to_csv(a.out, index=False); print(f"escorado → {a.out}")


if __name__ == "__main__":
    main()