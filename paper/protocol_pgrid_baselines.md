# Protocollo di valutazione “p-grid” (percentuali) vs baselines

Questo protocollo definisce un confronto **curva-based** per multilabel feature selection, dove si valuta ogni metodo su una griglia di **percentuali di feature selezionate**:

- $p \in \{5\%,10\%,15\%,\ldots,50\%\}$
- per ciascun dataset con $d$ feature: $k(p)=\\max(1,\\mathrm{round}(p\\,d))$

Nel paper riportiamo:

- una **tabella principale** a $p=20\%$ (mean $\\pm$ std su 5 fold),
- le **curve** per tutte le $p$ (aggregate sui dataset).

## 1) Dati e split (fissi)

- Usare gli split in `data/paper_matlab_minmax/<DATASET>/fold{i}.mat` (i = 0..4).
- Ogni `fold{i}.mat` contiene:
  - `X_train`, `Y_train`, `X_test`, `Y_test`
- Preprocessing: min--max su train $\rightarrow [0,1]$ (nonnegativo) ed applicato a test.

Regole anti-leakage:
- Il ranking (feature selection) deve usare **solo** `X_train`,`Y_train`.
- Le metriche vanno calcolate **solo** su `X_test`,`Y_test`.
- Se un metodo usa iperparametri “data-driven”, può stimarli solo con split interni del train fold (mai sul test fold).

## 2) Griglia percentuale (p-grid)

Definizione standard:
- `p_values = 0.05:0.05:0.50`
- `k_values(p) = max(1, round(p * d))`

Nota: quando $d$ è piccolo, percentuali diverse possono mappare allo stesso $k$; in quel caso le metriche vengono replicate sul p-grid.

## 3) Modello di valutazione (ML-kNN)

Per ogni metodo/dataset/fold:
1. Calcolare un ranking `r` (indici 1-based) su `X_train`,`Y_train`.
2. Per ogni $p$ nel p-grid:
   - selezionare `S_p = r(1:k(p))`
   - valutare ML-kNN su `X_train(:,S_p) -> X_test(:,S_p)`

Metriche reportate:
- Micro-F1
- Macro-F1
- Hamming Loss
- Micro PR-AUC (AUPRC, threshold-free)
- Macro PR-AUC (AUPRC, threshold-free)
- AvgPrec (Average Precision example-based / LRAP, higher is better)
- One-error (lower is better)

Parametri standard ML-kNN (pipeline MATLAB):
- `mlknn_k = 10`
- `mlknn_smooth = 1`

## 4) Output atteso su disco

Per ogni `<METHOD>/<DATASET>_fold<i>`:
- `results/<RUN_DIR>/<METHOD>/<DATASET>_fold<i>_ranking.csv`
- `results/<RUN_DIR>/<METHOD>/<DATASET>_fold<i>_time.txt`
- `results/<RUN_DIR>/<METHOD>/<DATASET>_fold<i>_pgrid_metrics.json`

`*_pgrid_metrics.json` contiene liste parallele (una voce per ogni $p$):
- `p_values`
- `k_values`
- `micro_f1`
- `macro_f1`
- `hamming_loss`
- `avg_precision`
- `one_error`
- più i campi di comodità `*_at_p_target` (con `p_target=0.20`).

Nel draft del paper, i PR-AUC vengono computati a $p=20\\%$ (operating point principale) dai posterior ML-kNN e salvati come campi di comodità:
- `micro_pr_auc_at_p_target`
- `macro_pr_auc_at_p_target`

Nel draft del paper, AvgPrec e One-error sono computati lungo tutto il p-grid dai posterior ML-kNN e salvati anche come campi di comodità a $p=20\\%$:
- `avg_precision_at_p_target`
- `one_error_at_p_target`

## 5) Comandi principali (repo)

Calcolo dei ranking DAGFS (proposto) su tutti i dataset/fold:
```bash
python3 scripts/run_dagfs_custom.py --method-name DAGFS --folds 5 --overwrite
```

Valutazione delle curve dato un insieme di ranking già calcolati:

```bash
matlab -batch "cd('/home/andrea/paper_MLFS'); addpath('baselines'); \
  methods={'DAGFS','LRMFS','GRRO','SRFS','SCNMF','LSMFS','LRDG','RFSFS'}; \
  datasets={'Arts','Business','Education','Entertain','Health','Recreation','Reference','Science','Social', \
            'bibtex','corel5k','emotions','genbase','medical','yeast'}; \
  eval_rankings_matlab('results/bench_paper_kgrid','data/paper_matlab_minmax',methods,datasets, \
    'grid_mode','pgrid','folds',5,'p_min',0.05,'p_max',0.50,'p_step',0.05,'p_target',0.20, \
    'skip_existing',true);"
```

Augmentazione dei JSON con PR-AUC a $p=20\\%$ (solo p-target; training-only, nessun leakage):

```bash
python3 scripts/augment_pgrid_with_pr_auc.py \
  --results-dir results/bench_paper_kgrid \
  --data-dir data/paper_matlab_minmax \
  --methods DAGFS LRMFS GRRO SRFS SCNMF LSMFS LRDG RFSFS \
  --datasets Arts Business Education Entertain Health Recreation Reference Science Social bibtex corel5k emotions genbase medical yeast \
  --folds 5 --p-target 0.2
```

Generazione tabelle a $p=20\\%$:

```bash
python3 scripts/aggregate_kgrid_and_make_tables.py \
  --results-dir results/bench_paper_kgrid \
  --grid-mode pgrid --p-target 0.2 \
  --methods DAGFS LRMFS GRRO SRFS SCNMF LSMFS LRDG RFSFS \
  --datasets Arts Business Education Entertain Health Recreation Reference Science Social bibtex corel5k emotions genbase medical yeast \
  --ref-method DAGFS \
  --out-dir paper/tables \
  --display-map paper/tables/display_map_paper.json \
  --split-cols 4
```

Curve aggregate + figura:

```bash
python3 scripts/make_pgrid_curves.py \
  --results-dir results/bench_paper_kgrid \
  --methods DAGFS LRMFS GRRO SRFS SCNMF LSMFS LRDG RFSFS \
  --datasets Arts Business Education Entertain Health Recreation Reference Science Social bibtex corel5k emotions genbase medical yeast \
  --out-dir paper/figures \
  --display-map paper/tables/display_map_paper.json \
  --include-extra-metrics
```
