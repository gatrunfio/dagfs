# Protocollo di valutazione “k-grid” (curve 10..300) vs baselines

Nota (aggiornamento): per il paper stiamo usando come protocollo principale il confronto a \emph{percentuali di feature} ($p$-grid, con tabella a $p=20\\%$ e curve $5\\%..50\\%$). Vedi `paper/protocol_pgrid_baselines.md`.

Questo protocollo definisce un confronto **curva-based** (in stile SRFS) per multilabel feature selection, dove si valuta ogni metodo per una griglia di feature selezionate `k = 10,20,...,300` (clippata a `d` quando `d<300`) e si aggregano i risultati su **5-fold**.

## 1) Dati e split (fissi)

- Usare gli split già presenti in `data/paper_matlab_minmax/<DATASET>/fold{i}.mat` (i = 0..4).
- Ogni `fold{i}.mat` contiene tipicamente:
  - `X_train`, `Y_train`, `X_test`, `Y_test`
- Preprocessing: la directory `paper_matlab_minmax` indica che le feature sono già in scala nonnegativa / min-max (coerente con SRFS e con DAGFS).

Nota sugli split:
- Gli split possono essere generati come **repeated holdout** (ripetizioni stratificate train/test) oppure come **k-fold CV** stratificato. In questo repo l’exporter supporta entrambe le modalità (`scripts/export_cv_splits_to_mat.py --split-mode ...`), e la scelta va riportata nel paper per trasparenza.

Regola di correttezza:
- La feature selection (ranking) deve essere calcolata **solo su train** (`X_train`,`Y_train`).
- L’accuratezza va misurata **solo su test** (`X_test`,`Y_test`) per ogni `k`.
- Se il metodo ha iperparametri “data-driven”, può stimarli **solo** con split interni di training (mai usando il test fold).

## 2) Griglia delle feature (k-grid)

Definizione standard:
- `k_min = 10`
- `k_max = 300`
- `k_step = 10`
- `k_values = k_min:k_step:min(k_max, d)`

Nota: se `d < 300`, `k_max` viene automaticamente clippato a `d` (vedi `baselines/run_baseline.m`).

## 3) Modello di valutazione (ML-kNN)

Per ogni metodo/dataset/fold:
1. Calcolare un **ranking** `r` (indici 1-based) su `X_train`,`Y_train`.
2. Per ogni `k` nella griglia:
   - selezionare `S_k = r(1:k)`
   - valutare ML-kNN su `X_train(:,S_k) -> X_test(:,S_k)`
3. Salvare le curve con metriche:
   - `micro_f1(k)`
   - `macro_f1(k)`
   - `hamming_loss(k)`

Parametri standard ML-kNN (coerenti con la pipeline MATLAB):
- `mlknn_k = 10`
- `mlknn_smooth = 1`

Implementazione usata:
- `baselines/_eval_mlknn/MLKNN_eval_kgrid.m`

## 4) Output atteso su disco

Per ogni `<METHOD>/<DATASET>_fold<i>`:
- `results/<RUN_DIR>/<METHOD>/<DATASET>_fold<i>_ranking.csv`
- `results/<RUN_DIR>/<METHOD>/<DATASET>_fold<i>_time.txt`
- `results/<RUN_DIR>/<METHOD>/<DATASET>_fold<i>_kgrid_metrics.json`

Formato minimo `*_kgrid_metrics.json` (liste parallele, una voce per ogni `k`):
- `k_values`
- `micro_f1`
- `macro_f1`
- `hamming_loss`

## 5) Esecuzione: baselines MATLAB (ranking + curve)

La via più semplice è usare `run_suite`/`run_baseline` in modalità k-grid.

Esempio (tutti i metodi e dataset, 5-fold):

```bash
matlab -batch "addpath(genpath(pwd)); run_suite( ...
  'results/bench_paper_kgrid', ...
  'data/paper_matlab_minmax', ...
  {'Arts','Business','Education','Entertain','Health','Recreation','Reference','Science','Social', ...
   'emotions','enron','genbase','medical','scene','yeast'}, ...
  {'LRMFS','GRRO','SRFS','SCNMF','LSMFS','LRDG','RFSFS'}, ...
  'folds',5,'mlknn_k',10,'mlknn_smooth',1, ...
  'k_min',10,'k_max',300,'k_step',10, ...
  'skip_existing',true,'overwrite',false);"
```

Nota: **DAGFS** (proposto) viene eseguito via runner Python e produce solo i ranking; la valutazione k-grid avviene poi con `eval_rankings_matlab` (o con gli script Python p-grid).

Esempio (ranking DAGFS su tutti i dataset/fold):
```bash
python3 scripts/run_dagfs_custom.py --method-name DAGFS --folds 5 --overwrite
```

Note pratiche:
- `skip_existing=true` evita di ricalcolare fold già completi.
- `overwrite=true` forza il rerun (utile se cambia la variante).

## 6) Esecuzione: varianti (ranking) + valutazione successiva (curve)

Se una variante produce solo `*_ranking.csv` (e magari `*_time.txt`), si può valutare dopo con:
- `baselines/eval_rankings_matlab.m`

Esempio:
```bash
matlab -batch "addpath(genpath(pwd)); eval_rankings_matlab( ...
  'results/bench_paper_kgrid', 'data/paper_matlab_minmax', ...
  {'DAGFS'}, ...
  {'Arts','Business','Education','Entertain','Health','Recreation','Reference','Science','Social', ...
   'emotions','enron','genbase','medical','scene','yeast'}, ...
  'folds',5,'mlknn_k',10,'mlknn_smooth',1,'k_min',10,'k_max',300,'k_step',10,'skip_existing',true);"
```

Vincolo: il ranking deve essere 1-based e ordinato per importanza decrescente.

## 7) Aggregazione delle curve in un singolo punteggio (per fold)

Per confrontare i metodi con una tabella compatta (e test statistici), si riduce ogni curva a uno scalare per fold:

- **Fold-average**: per ogni fold si fa la media su tutti i k:
  - `micro_f1_fold = mean_k micro_f1(k)`
  - `macro_f1_fold = mean_k macro_f1(k)`
  - `hamming_loss_fold = mean_k hamming_loss(k)`

Questa è esattamente la logica implementata in:
- `scripts/aggregate_kgrid_and_make_tables.py` (funzione `fold_avg`)

## 8) Tabelle finali (mean ± std su 5 fold)

Per ogni dataset e metodo:
- si calcolano i 5 valori (uno per fold) ottenuti al punto 7
- si riporta `mean ± std` sui 5 fold

Script:
```bash
python3 scripts/aggregate_kgrid_and_make_tables.py \
  --results-dir results/bench_paper_kgrid \
  --methods DAGFS LRMFS GRRO SRFS SCNMF LSMFS LRDG RFSFS \
  --ref-method DAGFS \
  --out-dir paper/tables \
  --display-map paper/tables/display_map_paper.json
```

Output in `paper/tables/`:
- `table_microf1.tex`
- `table_macrof1.tex`
- `table_hamming.tex`
- `table_stats.tex`
- `benchmark_means_kgrid.csv`

## 9) Test statistici (across datasets)

Usiamo una valutazione **per dataset** basata sul punteggio k-grid-averaged:
- per ogni dataset e metodo: media sui 5 fold del valore “fold-average” (punto 7)

Poi:
1. **Friedman test** (metodi come trattamenti; dataset come blocchi) per verificare differenze globali.
2. **Wilcoxon signed-rank paired** tra metodo di riferimento (es. `DAGFS`) e ogni baseline, sui dataset.
3. **Correzione di Holm** sui p-value delle comparazioni multiple.

Questo è eseguito automaticamente da:
- `scripts/aggregate_kgrid_and_make_tables.py` (sezione `make_stats_tex`)

## 10) Check-list di riproducibilità

- Stessi split per tutti: `data/paper_matlab_minmax/.../fold{i}.mat`
- Stessa griglia: `10..300 step 10` (clippata a `d`)
- Stesso valutatore: ML-kNN con `k=10`, `smooth=1`
- Nessuna informazione di test usata in training/tuning
- Stesso formato output: `*_ranking.csv` e `*_kgrid_metrics.json`
