import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from credpipe import config, ingest, split, synth

LAYOUT = Path(__file__).resolve().parents[1] / "examples" / "layout.yaml"


@pytest.fixture(scope="session")
def cfg():
    return config.load_layout(LAYOUT)


@pytest.fixture(scope="session")
def base(cfg):
    df = ingest.validar(synth.gerar(n=8000, seed=1), cfg)
    return df, split.dividir(df, cfg)


@pytest.fixture(scope="session")
def base_drift(cfg):
    df = ingest.validar(synth.gerar(n=8000, seed=1, drift=True), cfg)
    return df, split.dividir(df, cfg)
