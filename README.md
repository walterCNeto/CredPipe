# credpipe

Pipeline end-to-end de modelo de PD (probabilidade de default): **base + layout → regressão logística (PyCaret ou scikit-learn) → escala 100–900 → scorecard WoE com refino recursivo → estabilidade por safra → malha de GH sem cruzamento temporal → PD calibrada no OOT → desafiantes ML → relatório de documentação**.

## Instalação

```bash
# ambiente com PyCaret (Python 3.9–3.11):
conda activate pycaret
pip install -e ".[pycaret,dev]"
# sem PyCaret (qualquer Python >= 3.9): backend scikit-learn com a mesma interface
pip install -e ".[dev]"
```

## Uso

```bash
credpipe synth --out base.csv                                  # base simulada de exemplo (DGP conhecido)
credpipe run --data base.csv --layout examples/layout.yaml --out runs/2026-08-27
credpipe score --run runs/2026-08-27 --data novos.csv --out novos_escorados.csv
python -m pytest -q
```

Saídas do `run`: `relatorio.html` (+ `.md`; `.pdf` se houver pandoc+xelatex), `modelo_scorecard_final.xlsx` (resumo, scorecard, de-para Score→GH→PD, mapa GH, coeficientes, CV, decis, estabilidade, logs), `scorecard_final.csv`, `de_para_score_GH_PD.csv`, `escorador.pkl` (base bruta → score, GH, PD), `modelo_lr.pkl`, `resultados.json`, `figs/`.

## Layout da base

Uma linha por operação; colunas `id`, `safra` (YYYYMM), `target` (0/1, já marcado e maturado) e as explicativas. Tudo que muda entre bases está em `examples/layout.yaml`: nomes das colunas, OOT, variáveis com tipo/sinal esperado/bins manuais/descrição, restrições do refino e da malha de GH, escala, MoC, desafiantes, formato numérico (3 casas, vírgula).

## Módulos

| módulo | função |
|---|---|
| `ingest` | valida layout: id único, target 0/1, safra YYYYMM, maturação, tipos |
| `split` | dev_train / OOS / OOT pela safra |
| `lr` | regressão logística (backend `pycaret` ou `sklearn`), CV, tuning de C, coeficientes, decis, cortes |
| `scale` | Factor/Offset/PDO; prob ↔ score |
| `scorecard` | faixas, WoE/IV (missing = faixa própria; níveis raros = OUTROS), LR sobre WoE, pontos |
| `refine` | Algoritmo H: mescla/descarta até IV, população, monotonicidade, inversão, ρ, CSI, sinal dos betas e sinal esperado |
| `stability` | métricas/PSI por safra; CSI, ρ, inversões por variável |
| `grades` | malha de GH recursiva: monotônica no dev, massa mínima, sem cruzamento (significativo) em nenhuma janela |
| `calibration` | PD por GH = taxa OOT, IC Jeffreys, binomial, HHI, de-para, `Escorador` |
| `challengers` | GBM/RF/ET (sklearn) ou `compare_models` (PyCaret) no mesmo split, OOS/OOT, importâncias |
| `plots`, `report` | figuras e relatório Jinja2 → Markdown → HTML (referências de `templates/refs.bib`) |
| `pipeline`, `cli` | orquestração e linha de comando |

## Decisões que ficam com o usuário (no yaml)

- `gh.alpha_cruz`: `0.10` conta só cruzamentos significativos (z unilateral); `null` conta qualquer inversão — estrito, mas colapsa a malha com células pequenas.
- `calibracao.moc`: `IC95_sup` (limite superior de Jeffreys), fator numérico ou `null`.
- `escala.modo`: `pdo` (convenção de mercado, ancorada) ou `faixa_cheia` (percentis 0,5%/99,5% em 100/900).
- A PD calibrada é PIT (taxa do OOT). Capital IRB exige recalibração à média de longo prazo — fora do escopo.
