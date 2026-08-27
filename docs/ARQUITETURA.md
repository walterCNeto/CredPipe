# Arquitetura

Fluxo de `pipeline.run(data, layout, out)`:

1. `config.load_layout` → `ingest.validar` → `split.dividir` (dev_train, OOS, OOT por safra)
2. `lr.ajustar` — LR no dev_train com OOT como test_data (PyCaret) / pipeline sklearn; CV, tuning de C, métricas, decis, cortes
3. `scale.Escala.ajustar` — a partir das PDs do dev_train
4. `scorecard.spec_inicial` → `refine.refinar` (Algoritmo H) → `scorecard.Scorecard.ajustar`
5. `stability.por_safra` (PSI/AUC/KS por safra) e `stability.por_variavel` (CSI/ρ/inversões)
6. `grades.malha_recursiva` (monotonia dev, massa mínima, sem cruzamento por janela) → `calibration.mapa_gh` → `calibration.de_para` → `Escorador`
7. `challengers.rodar` — mesmo split
8. `plots.*` → `report.gerar` (templates/relatorio.md.j2 + refs.bib)

Contratos: toda função recebe DataFrame + `cfg` e devolve dicts/DataFrames serializáveis; nada imprime/exibe. Backend PyCaret e sklearn expõem a mesma `ModeloLR.predict(df) -> P(default)`.

Referências de desenvolvimento em `docs/referencia/` (notebook original e artigo em LaTeX).
