# Context-Aware Recommender System — Project TODO

> Sequential steps for an IDE agent (Claude Code, Cursor, Copilot Workspace, etc.)
> Every task is atomic and testable. Complete steps in order within each phase.
> Dataset: MovieLens 1M (training) · MovieLens 100K (smoke tests / fast iteration)
> Owner tags: [A] = Person A (Data/Modeling) · [B] = Person B (Eval/MLOps) · [AB] = both

---

## Project structure reference

```
context-aware-recsys/
├── configs/                        ← YAML configs per model + ablation variants
│   └── ablation/
├── data/
│   ├── raw/
│   │   └── movielens/              ← ml-1m/ and ml-100k/ unzipped here
│   ├── processed/                  ← cleaned parquet splits + encoders + movie metadata
│   └── features/                   ← feature-enriched parquet + scalers/stats
├── docker/
│   └── Dockerfile                  ← single multi-stage Dockerfile (base → api → demo)
├── docs/                           ← deliverable markdown documents
├── notebooks/                      ← EDA + interpretability notebooks
├── outputs/
│   ├── checkpoints/                ← model .ckpt files per run
│   ├── figures/                    ← SHAP plots, gate heatmaps, EDA charts
│   └── logs/                       ← MLflow mlruns + W&B cache
├── src/
│   ├── ablation/                   ← AblationDataset + ablation runner
│   ├── api/                        ← FastAPI app, schemas, feature builder, Gradio demo
│   ├── data/
│   │   ├── features/               ← temporal, session, device, context, negative sampling
│   │   ├── movielens.py            ← load + split
│   │   ├── dataset.py              ← RecSysDataset
│   │   ├── datamodule.py           ← RecSysDataModule
│   │   └── download.py             ← download + checksum
│   ├── evaluation/                 ← metrics, evaluator
│   ├── interpretability/           ← SHAP analysis, attention viz
│   ├── models/
│   │   ├── modules/                ← GMF, MLP, ContextGate building blocks
│   │   ├── ncf.py
│   │   ├── context_ncf_late.py
│   │   ├── context_ncf_early.py
│   │   └── context_ncf_attn.py
│   └── training/                   ← train.py, ExperimentLogger
├── tests/
├── main.py                         ← unified Typer CLI
├── pyproject.toml
├── requirements.txt
├── params.yaml
├── dvc.yaml
├── docker-compose.yml
├── SCHEMA_CONTRACT.md
└── README.md
```

---

## Key design decision — synthesizing sessions from MovieLens

MovieLens 1M is a rating log spanning years — it has no native session structure.
Sessions are synthesized from each user's chronological rating history:

- Sort each user's ratings by `timestamp` ascending
- Start a new session whenever the gap to the previous rating exceeds **1 hour** (3600 seconds)
- This models realistic "viewing sessions" — a burst of ratings in one sitting
- Derived columns: `session_id` (global unique int), `session_pos` (0-indexed position within session), `session_len` (total ratings in session)

This is academically sound and produces comparable session signals to what event-log datasets provide natively.

---

## Phase 0 — Bootstrap [AB]

- [ ] **Step 0.1** — Install dependencies using `uv`:

  ```bash
  uv venv .venv && source .venv/bin/activate
  uv pip install -r requirements.txt
  ```

  `requirements.txt` must contain:

  ```
  torch
  pytorch-lightning
  torchmetrics
  pandas
  polars
  numpy
  scikit-learn
  shap
  mlflow
  dvc
  wandb
  fastapi
  uvicorn
  redis
  gradio
  onnx
  onnxruntime
  pytest
  httpx
  pydantic
  python-dotenv
  omegaconf
  typer
  matplotlib
  seaborn
  jupyterlab
  tabulate
  ```

- [ ] **Step 0.2** — Configure `pyproject.toml`:
  - `[tool.pytest.ini_options] testpaths = ["tests"]`
  - `[tool.ruff]` linting: `line-length = 100`, `select = ["E", "F", "I"]`
  - Project metadata: `name = "context-aware-recsys"`, `requires-python = ">=3.11"`

- [ ] **Step 0.3** — Initialize DVC:

  ```bash
  dvc init
  ```

  Verify `.dvc/` folder and `.dvcignore` are created

- [ ] **Step 0.4** — Create `.gitignore`:

  ```
  .venv/
  __pycache__/
  *.pyc
  .env
  data/raw/
  *.onnx
  wandb/
  outputs/logs/mlruns/
  outputs/checkpoints/
  ```

- [ ] **Step 0.5** — Create `params.yaml`:

  ```yaml
  data:
    seed: 42
    movielens_version: "1m" # "100k" for smoke tests
    session_gap_seconds: 3600 # 1-hour gap = new session
    negative_samples_train: 4
    negative_samples_eval: 99
    test_split_ratio: 0.1
    val_split_ratio: 0.1
    min_user_interactions: 5 # drop users with fewer ratings

  model:
    embedding_dim: 64
    mlp_layers: [128, 64, 32]
    dropout: 0.2
    lr: 0.001
    weight_decay: 0.0001
    batch_size: 256
    max_epochs: 50
    patience: 5

  eval:
    k_values: [5, 10, 20]
  ```

- [ ] **Step 0.6** — Create `SCHEMA_CONTRACT.md`:

  ```markdown
  ## Processed interaction schema

  | Column      | Type  | Notes                          |
  | ----------- | ----- | ------------------------------ |
  | user_id     | int   | 0-indexed via LabelEncoder     |
  | item_id     | int   | 0-indexed via LabelEncoder     |
  | timestamp   | int   | Unix seconds                   |
  | rating      | float | 1.0–5.0                        |
  | split       | str   | train / val / test             |
  | session_id  | int   | synthesized from rating bursts |
  | session_pos | int   | 0-indexed within session       |
  | session_len | int   | total ratings in session       |

  ## Context vector — 9 dimensions, always this order

  [sin_hour, cos_hour, sin_dow, cos_dow,
  session_pos_norm, session_len_norm,
  device_0, device_1, device_2]

  ## Model interface contract

  predict(user_id: int, item_ids: List[int], context: Dict) -> List[float]

  ## Checkpoint naming

  outputs/checkpoints/{experiment_name}/best.ckpt
  ```

- [ ] **Step 0.7** — Create all `__init__.py` files so `src/` and all subpackages are importable

- [ ] **Step 0.8** — Create config stubs in `configs/`:
      `ncf.yaml`, `context_late.yaml`, `context_early.yaml`, `context_attn.yaml`
      (fill content in Step 4.6)

- [ ] **Step 0.9** — Create ablation config stubs in `configs/ablation/`:
      `no_temporal.yaml`, `no_session.yaml`, `no_device.yaml`, `context_only.yaml`
      (fill content in Step 4.6)

- [ ] **Step 0.10** — Initial commit:
  ```bash
  git add . && git commit -m "chore: project skeleton"
  git push origin main
  ```

---

## Phase 1 — Data ingestion [A]

- [ ] **Step 1.1** — Create `src/data/download.py`:
  - `download_movielens(version: str = "1m", dest: str = "data/raw/movielens/") -> Path`
    - URLs:
      - 1M: `https://files.grouplens.org/datasets/movielens/ml-1m.zip`
      - 100K: `https://files.grouplens.org/datasets/movielens/ml-100k.zip`
    - Download with `urllib.request.urlretrieve`; unzip with `zipfile.ZipFile`
    - Verify MD5 checksums (known values below); raise `ValueError` on mismatch:
      - 1M MD5: `c4d9eecfca2ab87c1945afe126590906`
      - 100K MD5: `0e33842e24a9c977be4e0107933c0723`
    - Return path to unzipped folder

- [ ] **Step 1.2** — Register raw data in DVC:

  ```bash
  dvc add data/raw/movielens/
  git add data/raw/movielens.dvc .gitignore
  git commit -m "data: register MovieLens raw data in DVC"
  ```

- [ ] **Step 1.3** — Create `notebooks/01_eda_movielens.ipynb` with the following cells:
  - **Load:** read `ratings.dat` (1M) into DataFrame
  - **Basic stats:** print n_users, n_items, n_ratings, sparsity `= 1 - n_ratings/(n_users*n_items)`
  - **Rating distribution:** `plt.hist(df.rating, bins=9)` → save `outputs/figures/rating_dist.png`
  - **User activity:** interactions per user, log-scale sorted descending → save `outputs/figures/user_activity.png`
  - **Item popularity:** long-tail plot → save `outputs/figures/item_popularity.png`
  - **Temporal density:** bin by month, plot interaction count → save `outputs/figures/temporal_density.png`
  - **Session preview:** apply session synthesis (gap=3600s), print mean/median/max session length, histogram → save `outputs/figures/session_length_dist.png`
  - **Timeline sample:** plot rating timelines for 5 random users to visually validate session synthesis

- [ ] **Step 1.4** — Create `src/data/movielens.py`:

  **`load_movielens(path, version="1m") -> pd.DataFrame`**
  - 1M: parse `{path}/ml-1m/ratings.dat`, `sep="::"`, `engine="python"`, no header
  - 100K: parse `{path}/ml-100k/u.data`, `sep="\t"`, no header
  - Assign columns: `["user_id", "item_id", "rating", "timestamp"]`
  - Cast types: `user_id/item_id → int32`, `rating → float32`, `timestamp → int64`
  - Drop users with fewer than `params.data.min_user_interactions` ratings
  - Apply `LabelEncoder` to `user_id` and `item_id` independently (0-indexed)
  - Save `{"user": enc_user, "item": enc_item}` to `data/processed/encoders.pkl`
  - Return df sorted by `timestamp` ascending

  **`load_movie_metadata(path, version="1m") -> pd.DataFrame`**
  - 1M: parse `{path}/ml-1m/movies.dat`, `sep="::"`, columns `["item_id_raw","title","genres"]`
  - 100K: parse `{path}/ml-100k/u.item`, `sep="|"`, extract `item_id_raw` and `title`
  - Apply saved item `LabelEncoder` to remap `item_id_raw → item_id`
  - Save to `data/processed/movies.parquet` (used by API `/items/{item_id}`)

  **`time_split(df, val_ratio, test_ratio) -> Tuple[df, df, df]`**
  - Sort by `timestamp` ascending
  - Chronological slice: train = first fraction, val = next, test = last
  - Drop users in val/test not present in train (cold-start removal)
  - Add `split` column: `"train"` / `"val"` / `"test"`
  - Save: `data/processed/train.parquet`, `val.parquet`, `test.parquet`

- [ ] **Step 1.5** — Add `preprocess` stage to `dvc.yaml`:

  ```yaml
  stages:
    preprocess:
      cmd: python main.py preprocess
      deps:
        - src/data/movielens.py
        - src/data/download.py
        - data/raw/movielens/
        - params.yaml
      outs:
        - data/processed/train.parquet
        - data/processed/val.parquet
        - data/processed/test.parquet
        - data/processed/encoders.pkl
        - data/processed/movies.parquet
  ```

- [ ] **Step 1.6** — Run and verify:
  ```bash
  python main.py preprocess
  python -c "
  import pandas as pd, pickle
  df = pd.read_parquet('data/processed/train.parquet')
  enc = pickle.load(open('data/processed/encoders.pkl','rb'))
  print(df.shape, df.dtypes)
  print('n_users:', len(enc['user'].classes_), 'n_items:', len(enc['item'].classes_))
  "
  dvc repro && git add dvc.lock && git commit -m "data: preprocess MovieLens 1M"
  ```

---

## Phase 2 — Feature engineering [A]

- [ ] **Step 2.1** — Create `src/data/features/temporal.py`:

  **`encode_temporal(df: pd.DataFrame) -> pd.DataFrame`**
  - Convert `timestamp` (Unix seconds) to datetime via `pd.to_datetime(df.timestamp, unit="s")`
  - Extract `hour` (0–23), `day_of_week` (0–6, Mon=0), `month` (1–12)
  - Cyclic encoding — preserves circular continuity at boundaries:
    ```python
    sin_hour = np.sin(2 * np.pi * hour / 24)
    cos_hour = np.cos(2 * np.pi * hour / 24)
    sin_dow  = np.sin(2 * np.pi * day_of_week / 7)
    cos_dow  = np.cos(2 * np.pi * day_of_week / 7)
    ```
    (month encoding kept as `sin_month`/`cos_month` but not part of the 9-dim context vector;
    keep as auxiliary column for EDA only)
  - Drop raw `hour`, `day_of_week`, `month` columns
  - Return df with 4 new context columns + 2 auxiliary

- [ ] **Step 2.2** — Create `src/data/features/session.py`:

  **`synthesize_sessions(df: pd.DataFrame, gap_seconds: int = 3600) -> pd.DataFrame`**
  - Sort by `["user_id", "timestamp"]` (in-place for efficiency)
  - Per user group: compute `time_delta = timestamp.diff().fillna(gap_seconds + 1)`
  - Mark new session where `time_delta > gap_seconds` → cumulative sum = `session_id` (global unique)
  - `session_pos`: within each `session_id` group, use `.cumcount()`
  - `session_len`: `groupby("session_id")["session_pos"].transform("count")`
  - Return df with `session_id`, `session_pos`, `session_len` added

  **`encode_session(df: pd.DataFrame, scaler_path: str, fit: bool = False) -> pd.DataFrame`**
  - `session_pos_norm = session_pos / (session_len - 1 + 1e-8)` → always in `[0, 1]`
  - If `fit=True`: fit `MinMaxScaler` on `session_len`, save to `scaler_path`; always call with `fit=True` on train split only
  - Load scaler and transform `session_len → session_len_norm`
  - Return df with `session_pos_norm`, `session_len_norm` added

- [ ] **Step 2.3** — Create `src/data/features/device.py`:

  **`encode_device_proxy(df: pd.DataFrame) -> pd.DataFrame`**
  - `device_idx = df.user_id % 3` (0=mobile, 1=tablet, 2=desktop) — deterministic, no randomness
  - One-hot: `device_0`, `device_1`, `device_2` as `int8`
  - Drop `device_idx`
  - Return df

- [ ] **Step 2.4** — Create `src/data/features/context.py`:

  **`build_context_vector(df: pd.DataFrame) -> np.ndarray`**
  - Canonical column order (matches SCHEMA_CONTRACT):
    ```python
    CONTEXT_COLS = [
        "sin_hour", "cos_hour", "sin_dow", "cos_dow",
        "session_pos_norm", "session_len_norm",
        "device_0", "device_1", "device_2",
    ]
    ```
  - Assert all 9 columns exist; raise `KeyError` with helpful message if missing
  - Return `df[CONTEXT_COLS].to_numpy(dtype=np.float32)` — shape `(N, 9)`

  **`save_context_stats(arr: np.ndarray, path: str) -> None`**
  - Compute per-column mean and std on **train set only**
  - Save to `data/features/context_stats.json`:
    ```json
    {"columns": ["sin_hour", ...], "mean": [...], "std": [...], "dim": 9}
    ```
  - Loaded at inference time by `src/api/feature_builder.py`

- [ ] **Step 2.5** — Create `src/data/features/negative_sampling.py`:

  **`sample_negatives_train(df, n_neg=4, seed=42) -> pd.DataFrame`**
  - `user_items`: dict mapping each user_id → set of interacted item_ids
  - Item popularity weights: `pop[item] = count(item) / sum(counts)` — proportional sampling
  - For each row: sample `n_neg` items not in `user_items[user_id]`, weighted by popularity
  - Add column `item_neg` (single int — used in BPR loss)
  - Return df

  **`sample_negatives_eval(df, all_items: np.ndarray, n_neg=99, seed=42) -> pd.DataFrame`**
  - For each row (one per user's held-out item): sample `n_neg` items not seen by that user, uniformly
  - Add column `candidates`: `List[int]` of length 100 where `candidates[0]` = positive item
  - Return df

- [ ] **Step 2.6** — Create `src/data/features/pipeline.py`:

  ```python
  def run_feature_pipeline(split: str, fit_scalers: bool = False):
      df = pd.read_parquet(f"data/processed/{split}.parquet")
      df = synthesize_sessions(df, gap_seconds=params["data"]["session_gap_seconds"])
      df = encode_temporal(df)
      df = encode_session(df, "data/features/session_scaler.pkl", fit=fit_scalers)
      df = encode_device_proxy(df)
      ctx = build_context_vector(df)
      if fit_scalers:
          save_context_stats(ctx, "data/features/context_stats.json")
      if split == "train":
          df = sample_negatives_train(df, n_neg=params["data"]["negative_samples_train"])
      else:
          all_items = np.arange(n_items)
          df = sample_negatives_eval(df, all_items, n_neg=params["data"]["negative_samples_eval"])
      df.to_parquet(f"data/features/{split}.parquet", index=False)

  # Call order: train (fit_scalers=True) → val → test
  ```

- [ ] **Step 2.7** — Add `featurize` stage to `dvc.yaml`:

  ```yaml
  featurize:
    cmd: python main.py featurize
    deps:
      - src/data/features/
      - data/processed/train.parquet
      - data/processed/val.parquet
      - data/processed/test.parquet
      - params.yaml
    outs:
      - data/features/train.parquet
      - data/features/val.parquet
      - data/features/test.parquet
      - data/features/session_scaler.pkl
      - data/features/context_stats.json
  ```

- [ ] **Step 2.8** — Write `tests/test_features.py`:
  - Cyclic features: all values in `[-1.0, 1.0]`; `sin²+cos² ≈ 1.0` for both (hour, dow) pairs
  - `build_context_vector`: shape `(N, 9)`, dtype `float32`, zero NaNs
  - `session_pos_norm` always in `[0.0, 1.0]`
  - `session_len_norm` always in `[0.0, 1.0]`
  - Device: exactly one of `device_0/1/2` equals 1 per row
  - Train negatives: no `(user_id, item_neg)` pair also appears as a positive for that user
  - Eval negatives: `candidates[0]` always equals the positive item; list length always 100

---

## Phase 3 — PyTorch Dataset & DataModule [A]

- [ ] **Step 3.1** — Create `src/data/dataset.py`:

  ```python
  CONTEXT_COLS = [
      "sin_hour","cos_hour","sin_dow","cos_dow",
      "session_pos_norm","session_len_norm",
      "device_0","device_1","device_2",
  ]

  class RecSysDataset(Dataset):
      def __init__(self, parquet_path: str, mode: Literal["train","eval"]):
          self.df   = pd.read_parquet(parquet_path)
          self.mode = mode

      def __len__(self): return len(self.df)

      def __getitem__(self, idx):
          row = self.df.iloc[idx]
          ctx = torch.tensor(row[CONTEXT_COLS].values.astype(np.float32))
          if self.mode == "train":
              return {
                  "user":     torch.tensor(row.user_id,  dtype=torch.long),
                  "item_pos": torch.tensor(row.item_id,  dtype=torch.long),
                  "item_neg": torch.tensor(row.item_neg, dtype=torch.long),
                  "context":  ctx,           # shape (9,)
              }
          else:
              return {
                  "user":      torch.tensor(row.user_id, dtype=torch.long),
                  "items":     torch.tensor(row.candidates, dtype=torch.long),  # shape (100,)
                  "context":   ctx,
                  "label_idx": torch.tensor(0, dtype=torch.long),
              }
  ```

- [ ] **Step 3.2** — Create `src/data/datamodule.py` — `class RecSysDataModule(LightningDataModule)`:
  - Constructor: `(params: dict)`; reads `n_users`/`n_items` from `data/processed/encoders.pkl`
  - `setup(stage)`: instantiate `RecSysDataset` for each split
  - `train_dataloader()`: `shuffle=True, num_workers=4, pin_memory=True, batch_size=params.batch_size`
  - `val_dataloader()` / `test_dataloader()`: `shuffle=False`
  - Properties: `n_users: int`, `n_items: int`, `context_dim: int = 9`

- [ ] **Step 3.3** — Write `tests/test_dataset.py`:
  - `len(dataset)` equals parquet row count
  - Train: `user/item_pos/item_neg → torch.long`; `context → torch.float32` shape `(9,)`
  - Eval: `items` shape `(100,)`; `label_idx == 0`
  - DataModule: `train_dataloader()` iterates 2 batches without error
  - `n_users` and `n_items` are positive integers matching encoder classes

---

## Phase 4 — Models [A]

### Step 4.1 — Reusable building blocks

- [ ] Create `src/models/modules/gmf.py` — `class GMFBranch(nn.Module)`:
  - `forward(user_emb, item_emb) -> Tensor`: elementwise product, shape `(B, embedding_dim)`

- [ ] Create `src/models/modules/mlp.py` — `class MLPBranch(nn.Module)`:
  - Constructor: `(input_dim: int, layer_sizes: List[int], dropout: float)`
  - `nn.Sequential`: `[Linear → BatchNorm1d → ReLU → Dropout] × len(layer_sizes)`
  - `forward(x) -> Tensor`: input = concat `[user_emb, item_emb]`; output = last hidden layer

- [ ] Create `src/models/modules/gate.py` — `class ContextGate(nn.Module)`:
  - Constructor: `(context_dim: int, embedding_dim: int)`
  - Architecture: `Linear(context_dim, embedding_dim) → ReLU → Linear(embedding_dim, embedding_dim) → Sigmoid`
  - `forward(context_vec) -> Tensor`: gate in `[0,1]`, shape `(B, embedding_dim)`
  - Stores `self.last_gate = gate.detach()` after every forward (for interpretability)

### Step 4.2 — NCF baseline

- [ ] Create `src/models/ncf.py` — `class NCF(LightningModule)`:
  - Constructor: `(n_users, n_items, embedding_dim, mlp_layers, dropout, lr, weight_decay)`
  - `nn.Embedding(n_users+1, embedding_dim, padding_idx=0)` for user; same for item
  - Uses `GMFBranch` and `MLPBranch`
  - Final: `nn.Linear(embedding_dim + mlp_layers[-1], 1)` → `nn.Sigmoid()`
  - BPR loss: `L = -mean(log(σ(pos_score - neg_score) + 1e-8))`
  - `training_step`: log `train_loss`
  - `validation_step` + `on_validation_epoch_end`: compute and log `val_ndcg_10`
  - `configure_optimizers`: `AdamW` + `CosineAnnealingLR(T_max=max_epochs)`
  - `predict_score(user_ids, item_ids, context=None) -> Tensor`: context ignored in base NCF (accepted for interface compatibility)

- [ ] Write `tests/test_ncf.py`:
  - Forward shape `(B, 1)`, values in `[0, 1]`
  - BPR loss is scalar, positive, and `.backward()` runs without error
  - `predict_score` output in `[0, 1]`
  - Gradients flow to embedding tables after backward

### Step 4.3 — Context-NCF Late Fusion

- [ ] Create `src/models/context_ncf_late.py` — `class ContextNCFLate(NCF)`:
  - Final linear: `nn.Linear(embedding_dim + mlp_layers[-1] + context_dim, 1)`
  - Forward: concatenate context **after** NCF tower, before final linear layer
  - `predict_score(user_ids, item_ids, context) -> Tensor`

### Step 4.4 — Context-NCF Early Fusion

- [ ] Create `src/models/context_ncf_early.py` — `class ContextNCFEarly(NCF)`:
  - After embedding lookup: `user_in = torch.cat([user_emb, context], dim=-1)`; same for item
  - `MLPBranch` input dim: `2 * (embedding_dim + context_dim)`
  - `GMFBranch` operates on `(embedding_dim + context_dim)`-dim vectors

### Step 4.5 — Context-NCF Attention Gate (primary model)

- [ ] Create `src/models/context_ncf_attn.py` — `class ContextNCFAttn(NCF)`:
  - Add `self.gate = ContextGate(context_dim, embedding_dim)`
  - Forward:
    1. `g = self.gate(context_vec)` — `(B, embedding_dim)`, values in `[0, 1]`
    2. `user_emb_gated = g * user_emb`
    3. `item_emb_gated = g * item_emb`
    4. Feed gated embeddings to `GMFBranch` and `MLPBranch`
  - Gate stored in `self.gate.last_gate` automatically

- [ ] Write `tests/test_context_models.py`:
  - All 3 context models produce identical output shapes
  - All 3 produce different outputs for the same input (distinct architectures)
  - `ContextNCFAttn.gate.last_gate` is in `[0, 1]`, shape `(B, embedding_dim)`
  - All 3 `predict_score` accept a `context` tensor argument

### Step 4.6 — Config files

- [ ] Fill `configs/ncf.yaml`:

  ```yaml
  experiment_name: ncf_baseline
  model:
    type: NCF
    context_dim: null
  ```

- [ ] Fill `configs/context_late.yaml`, `context_early.yaml`, `context_attn.yaml`
      with `experiment_name` and `model.type` / `model.context_dim: 9`

- [ ] Fill `configs/ablation/no_temporal.yaml`:
  ```yaml
  experiment_name: ablation_no_temporal
  model:
    type: ContextNCFAttn
    context_dim: 9
  ablation:
    zeroed_cols: [sin_hour, cos_hour, sin_dow, cos_dow]
  ```
  Fill analogously:
  - `no_session.yaml` → zero `session_pos_norm`, `session_len_norm`
  - `no_device.yaml` → zero `device_0`, `device_1`, `device_2`
  - `context_only.yaml` → zero user and item embedding outputs (only context drives prediction)

### Step 4.7 — Training entry point

- [ ] Create `src/training/train.py` — `def train(config_path: str, smoke_test: bool = False)`:
  1. `cfg = OmegaConf.merge(OmegaConf.load("params.yaml"), OmegaConf.load(config_path))`
  2. `mlflow.set_experiment("context-recsys")` + `mlflow.start_run(run_name=cfg.experiment_name)`
  3. Instantiate `ExperimentLogger(cfg)`; `logger.log_params(OmegaConf.to_container(cfg))`
  4. Model registry: `{"NCF": NCF, "ContextNCFLate": ContextNCFLate, "ContextNCFEarly": ContextNCFEarly, "ContextNCFAttn": ContextNCFAttn}`
  5. If `cfg.get("ablation")`: use `AblationDataModule(zeroed_cols=cfg.ablation.zeroed_cols)` else `RecSysDataModule`
  6. If `smoke_test=True`: `max_epochs=1`; limit dataset to first 1000 rows
  7. `pl.Trainer(max_epochs, callbacks=[EarlyStopping("val_ndcg_10", patience, mode="max"), ModelCheckpoint(dirpath=f"outputs/checkpoints/{cfg.experiment_name}", monitor="val_ndcg_10", mode="max", save_top_k=1)])`
  8. `trainer.fit(model, datamodule)` then `trainer.test(model, datamodule)`
  9. `mlflow.log_artifact(f"outputs/checkpoints/{cfg.experiment_name}/best.ckpt")`

- [ ] Wire into `main.py`:

  ```python
  import typer
  app = typer.Typer()

  @app.command()
  def preprocess(): ...          # calls src/data/movielens.py

  @app.command()
  def featurize(): ...           # calls src/data/features/pipeline.py

  @app.command()
  def train(config: str, smoke_test: bool = False): ...

  @app.command()
  def evaluate(checkpoint: str): ...

  @app.command()
  def serve(): ...               # starts FastAPI

  if __name__ == "__main__":
      app()
  ```

---

## Phase 5 — Evaluation harness [B]

- [ ] **Step 5.1** — Create `src/evaluation/metrics.py` — pure functions, zero model dependency:
  - `ndcg_at_k(ranked_list, true_item, k) -> float`
    - If `true_item not in ranked_list[:k]`: `0.0`
    - Else: `1.0 / math.log2(rank + 2)` (0-indexed rank); ideal DCG = `1.0`

  - `hit_rate_at_k(ranked_list, true_item, k) -> float`
    - `1.0` if `true_item in ranked_list[:k]` else `0.0`

  - `mrr(ranked_list, true_item) -> float`
    - `1.0 / (rank + 1)` (0-indexed); `0.0` if not found

  - `coverage(all_recs: List[List[int]], n_items: int) -> float`
    - `len({item for recs in all_recs for item in recs}) / n_items`

  - `novelty(all_recs, item_pop: Dict[int,float]) -> float`
    - `mean(-log2(item_pop[item] + 1e-10))` across all recommended items

- [ ] **Step 5.2** — Write `tests/test_metrics.py`:
  - `ndcg_at_k([3,1,2], true_item=3, k=10)` → `1.0`
  - `ndcg_at_k([1,2,3], true_item=3, k=10)` → strictly between 0 and 1
  - `ndcg_at_k([1,2,3], true_item=9, k=10)` → `0.0`
  - `hit_rate_at_k([5,3,1], true_item=5, k=1)` → `1.0`
  - `hit_rate_at_k([1,2,3], true_item=5, k=3)` → `0.0`
  - `mrr([5,3,1], true_item=5)` → `1.0`
  - `mrr([5,3,1], true_item=1)` → `1/3`
  - `coverage([[1,2],[2,3]], n_items=10)` → `0.3`

- [ ] **Step 5.3** — Create `src/evaluation/evaluator.py` — `class Evaluator`:
  - Constructor: `(model, test_dataloader, k_values=[5,10,20], n_items: int)`
  - `run() -> dict`:
    - Model in `eval()` mode + `torch.no_grad()`
    - Per batch: `scores = model.predict_score(users, candidate_items, context)` shape `(B, 100)`
    - Argsort descending → ranked candidate indices → map to item IDs
    - Accumulate all metrics with running mean
    - Return `{"NDCG": {5:.., 10:.., 20:..}, "HR": {...}, "MRR": float, "Coverage": float, "Novelty": float}`
  - `summary_table() -> pd.DataFrame`: one row, all metrics, 4 decimal places

- [ ] **Step 5.4** — Create `src/ablation/ablation_runner.py`:
  - Discover all checkpoints: `glob("outputs/checkpoints/*/best.ckpt")`
  - For each: infer model class from folder name; load checkpoint; `Evaluator.run()`
  - Build DataFrame:
    `model_name | NDCG@5 | NDCG@10 | NDCG@20 | HR@5 | HR@10 | HR@20 | MRR | Coverage | Novelty | Δ_NDCG@10`
  - `Δ_NDCG@10` = relative to `ncf_baseline` row
  - Sort by `NDCG@10` descending
  - Save to `outputs/ablation_table.csv`; print with `tabulate`

- [ ] **Step 5.5** — Create `src/ablation/ablation_dataset.py` — `class AblationDataset(RecSysDataset)`:
  - Constructor adds `zeroed_cols: List[str]`
  - `__getitem__`: build context tensor normally; zero out indices from `CONTEXT_COLS` list
  - e.g. `zeroed_cols=["sin_hour","cos_hour"]` → zero indices 0 and 1 in the 9-dim vector

---

## Phase 6 — Experiment tracking [B]

- [ ] **Step 6.1** — Create `src/training/logger.py` — `class ExperimentLogger`:
  - `__init__(cfg)`: `wandb.init(project="context-recsys", name=cfg.experiment_name, config=...)`
  - `log_params(params)`: `mlflow.log_params(params)` + `wandb.config.update(params)`
  - `log_metrics(metrics, step)`: both backends
  - `log_artifact(path)`: `mlflow.log_artifact(path)`

- [ ] **Step 6.2** — Configure MLflow in `.env`:

  ```
  MLFLOW_TRACKING_URI=outputs/logs/mlruns
  ```

  Load in `train.py`: `from dotenv import load_dotenv; load_dotenv()`

- [ ] **Step 6.3** — Create `configs/sweep.yaml`:
  ```yaml
  method: bayes
  metric:
    name: val_ndcg_10
    goal: maximize
  parameters:
    embedding_dim:
      values: [32, 64, 128]
    dropout:
      distribution: uniform
      min: 0.1
      max: 0.4
    lr:
      distribution: log_uniform_values
      min: 0.0001
      max: 0.01
    mlp_layers:
      values:
        - [128, 64, 32]
        - [256, 128, 64]
        - [64, 32]
  ```
  Run: `wandb sweep configs/sweep.yaml` → paste sweep ID into `README.md`

---

## Phase 7 — CI pipeline [B]

- [ ] **Step 7.1** — Create `.github/workflows/ci.yml`:

  ```yaml
  name: CI
  on: [push, pull_request]
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with:
            python-version: "3.11"
        - name: Install
          run: pip install uv && uv pip install -r requirements.txt --system
        - name: Unit tests
          run: pytest tests/ -v --tb=short
        - name: Smoke train (100K, 1 epoch)
          run: python main.py train --config configs/ncf.yaml --smoke-test
  ```

- [ ] **Step 7.2** — Push a feature branch; confirm CI passes before merging to `main`

- [ ] **Step 7.3** — Add CI status badge to `README.md`

---

## Phase 8 — Interpretability [A]

- [ ] **Step 8.1** — Create `src/interpretability/shap_analysis.py`:
  - Load best `ContextNCFAttn` checkpoint
  - `class ModelWrapper`:
    - `__call__(X: np.ndarray) -> np.ndarray`
    - Input context `(N, 9)`; uses fixed "median user" embedding and fixed popular item; only context varies
    - Returns predicted scores `(N,)` as numpy float32
  - `background = train_context_array[np.random.choice(N_train, 100, replace=False)]`
  - `explainer = shap.KernelExplainer(wrapper, background)`
  - `shap_values = explainer.shap_values(test_context_500)` — shape `(500, 9)`
  - Save to `outputs/figures/shap_values.npy`
  - Summary plot → `outputs/figures/shap_summary.png`
  - Importance bar (mean `|SHAP|`) → `outputs/figures/shap_importance.png`

- [ ] **Step 8.2** — Create `src/interpretability/attention_viz.py`:
  - Load `ContextNCFAttn` checkpoint
  - Define 6 context scenarios covering morning/evening × session start/mid/end
  - Per scenario: run inference on 10 users; extract and average `model.gate.last_gate` `(embedding_dim,)`
  - Heatmap (rows=scenarios, cols=embedding dims, diverging colormap centered at 0.5)
  - Save to `outputs/figures/gate_heatmap.png`

- [ ] **Step 8.3** — Create `notebooks/03_interpretability.ipynb`:
  - Display SHAP summary + importance plots with written narrative
  - Display gate heatmap + written narrative
  - Worked example: user 42 — top-5 at 8 AM vs 10 PM, show diff + gate activation diff
  - Written conclusion: "Which context signal drives the most recommendation change?"

---

## Phase 9 — FastAPI serving [B]

- [ ] **Step 9.1** — Create `src/api/model_export.py`:
  - Load best `ContextNCFAttn` from `outputs/checkpoints/context_ncf_attn/best.ckpt`
  - TorchScript: `torch.jit.script(model).save("outputs/model.pt")`
  - ONNX: `torch.onnx.export(..., opset_version=17)` → `outputs/model.onnx`
  - Verify: `onnxruntime.InferenceSession`; assert max abs diff < `1e-4` vs PyTorch output
  - Print `"Export verified ✓"`

- [ ] **Step 9.2** — Create `src/api/schemas.py`:

  ```python
  class ContextInput(BaseModel):
      hour:        int = Field(..., ge=0, le=23)
      day_of_week: int = Field(..., ge=0, le=6)
      session_pos: int = Field(..., ge=0)
      session_len: int = Field(..., ge=1)
      device:      int = Field(..., ge=0, le=2)  # 0=mobile 1=tablet 2=desktop

  class RecommendRequest(BaseModel):
      user_id: int
      context: ContextInput
      k: int = Field(default=10, ge=1, le=50)

  class RecommendResponse(BaseModel):
      user_id:    int
      items:      List[int]
      titles:     List[str]
      scores:     List[float]
      latency_ms: float
  ```

- [ ] **Step 9.3** — Create `src/api/feature_builder.py`:
  - `build_context_tensor(ctx: ContextInput) -> torch.Tensor` shape `(9,)`
  - Apply same cyclic encoding as `src/data/features/temporal.py`
  - Load `data/features/session_scaler.pkl`; normalize `session_len`
  - `session_pos_norm = session_pos / (session_len - 1 + 1e-8)`
  - One-hot device: `[int(ctx.device==i) for i in range(3)]`
  - Assemble in SCHEMA_CONTRACT order

- [ ] **Step 9.4** — Create `src/api/app.py`:
  - `lifespan`: load TorchScript model + Redis + movies lookup at startup
  - `GET /health` → `{"status": "ok", "model": "ContextNCFAttn"}`
  - `POST /recommend`:
    1. Build context tensor
    2. Redis cache check; key = `f"rec:u{user_id}:ctx:{hash(ctx_tuple)}:k{k}"`
    3. Miss: inference → argsort → top-k item IDs + titles from movies lookup → cache TTL=300s
    4. Return `RecommendResponse` with `latency_ms`
  - `GET /items/{item_id}` → `{"id", "title", "genres"}` from movies lookup

- [ ] **Step 9.5** — Write `tests/test_api.py`:
  - `GET /health` → 200
  - `POST /recommend` valid request → 200, `len(items) == k`
  - `POST /recommend` with `hour=99` → 422
  - `GET /items/1` → 200 with `title` field
  - Full latency < 200ms

---

## Phase 10 — Gradio demo [B]

- [ ] **Step 10.1** — Create `src/api/demo.py` with `gr.Blocks()`:

  **Tab 1 — "Get Recommendations":**
  - `gr.Slider(0, 23, step=1, value=12, label="Hour of day")`
  - `gr.Slider(0, 6, step=1, value=1, label="Day of week (0=Mon)")`
  - `gr.Slider(1, 20, step=1, value=1, label="Session position")`
  - `gr.Slider(1, 20, step=1, value=5, label="Session length")`
  - `gr.Radio(["Mobile","Tablet","Desktop"], value="Mobile", label="Device")`
  - `gr.Number(precision=0, value=1, label="User ID")`
  - `gr.Button("Recommend")`
  - `gr.Dataframe(headers=["Rank","Item ID","Title","Score"])`
  - Button → `httpx.post(f"{API_URL}/recommend", json=payload)` → populate dataframe

  **Tab 2 — "Ablation Comparison":**
  - Same inputs; on submit calls all model variants; shows 5 side-by-side dataframes labeled by model name

  - `demo.launch(server_name="0.0.0.0", server_port=7860)`

---

## Phase 11 — Docker [B]

- [ ] **Step 11.1** — Rewrite `docker/Dockerfile` as multi-stage:

  ```dockerfile
  FROM python:3.11-slim AS base
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install uv && uv pip install -r requirements.txt --system

  FROM base AS api
  COPY src/ ./src/
  COPY main.py .
  COPY outputs/model.pt ./outputs/
  COPY data/features/session_scaler.pkl ./data/features/
  COPY data/features/context_stats.json ./data/features/
  COPY data/processed/movies.parquet ./data/processed/
  EXPOSE 8000
  CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

  FROM base AS demo
  COPY src/api/demo.py ./src/api/demo.py
  EXPOSE 7860
  CMD ["python", "src/api/demo.py"]
  ```

- [ ] **Step 11.2** — Create `docker-compose.yml`:

  ```yaml
  services:
    redis:
      image: redis:7-alpine
      ports: ["6379:6379"]
    api:
      build: { context: ., dockerfile: docker/Dockerfile, target: api }
      ports: ["8000:8000"]
      depends_on: [redis]
      environment:
        REDIS_URL: redis://redis:6379
    demo:
      build: { context: ., dockerfile: docker/Dockerfile, target: demo }
      ports: ["7860:7860"]
      depends_on: [api]
      environment:
        API_URL: http://api:8000
  ```

- [ ] **Step 11.3** — Smoke test:
  ```bash
  docker-compose up --build -d
  curl http://localhost:8000/health
  curl -X POST http://localhost:8000/recommend \
    -H "Content-Type: application/json" \
    -d '{"user_id":1,"context":{"hour":10,"day_of_week":1,"session_pos":0,"session_len":5,"device":0},"k":10}'
  # Open http://localhost:7860
  ```

---

## Phase 12 — Training runs [AB]

- [ ] **Step 12.1** — `python main.py train --config configs/ncf.yaml`
      Record: `run_id`, `NDCG@10`, `HR@10`, `MRR` from MLflow UI

- [ ] **Step 12.2** — `python main.py train --config configs/context_late.yaml`

- [ ] **Step 12.3** — `python main.py train --config configs/context_early.yaml`

- [ ] **Step 12.4** — `python main.py train --config configs/context_attn.yaml`

- [ ] **Step 12.5** — Run all 4 ablations:

  ```bash
  for cfg in configs/ablation/*.yaml; do python main.py train --config "$cfg"; done
  ```

- [ ] **Step 12.6** — W&B sweep (≥15 trials):

  ```bash
  wandb agent <sweep_id>
  ```

  Record best hyperconfig in `README.md`

- [ ] **Step 12.7** — Verify: every run in MLflow has loss curve, `val_ndcg_10`, checkpoint artifact

- [ ] **Step 12.8** — Select best model on **val NDCG@10 only** — do not inspect test set yet

- [ ] **Step 12.9** — Final test-set evaluation:
  ```bash
  python main.py evaluate --checkpoint outputs/checkpoints/context_ncf_attn/best.ckpt
  ```
  Save all metrics to `outputs/final_results.json`

---

## Phase 13 — Deliverables [AB]

- [ ] **Step 13.1** — Create `docs/context_integration_design.md`:
  - Section 1: why static NCF is blind to _when_ and _how_ users engage
  - Section 2: context feature taxonomy — table with feature, what it captures, range, encoding method
  - Section 3: session synthesis rationale — why 1-hour gap suits MovieLens rating bursts
  - Section 4: fusion strategy comparison table — expressiveness · interpretability · inference cost
  - Section 5: attention gate math — `g = σ(W₂·ReLU(W₁·ctx))`, `ũ = g⊙u`, information flow diagram
  - Section 6: decision rationale citing ablation results

- [ ] **Step 13.2** — Create `docs/ranking_eval_report.md`:
  - Protocol: leave-one-out, 99-negative uniform sampling, k ∈ {5,10,20}
  - Full ablation table from `outputs/ablation_table.csv`
  - Statistical significance: paired t-test (`ContextNCFAttn` vs `ncf_baseline`), NDCG@10 per user; report t-stat and p-value
  - Error analysis: segment users by activity quartile; plot NDCG@10 per model per quartile
  - Limitations: synthesized sessions, synthetic device proxy, cold-start not addressed

- [ ] **Step 13.3** — Create `docs/interpretability_analysis.md`:
  - SHAP summary + importance bar with narrative — top 3 features by mean |SHAP|
  - Gate heatmap with narrative — context-sensitive embedding dimensions
  - Worked example: user 42, 8 AM vs 10 PM, top-5 diff + gate activation diff
  - Conclusion: quantified contribution per context signal

- [ ] **Step 13.4** — Update `README.md`:
  - 2-paragraph overview; dataset: MovieLens 1M with synthesized sessions
  - Quickstart: `git clone → uv install → python main.py preprocess → python main.py featurize → docker-compose up`
  - Reproduce training: `python main.py train --config configs/context_attn.yaml`
  - Reproduce eval: `python main.py evaluate --checkpoint outputs/checkpoints/context_ncf_attn/best.ckpt`
  - Ablation results table; W&B link; MLflow UI cmd: `mlflow ui --backend-store-uri outputs/logs/mlruns`

---

## Phase 14 — Final checks [AB]

- [ ] **Step 14.1** — `pytest tests/ -v --tb=short` → 0 failures, 0 errors
- [ ] **Step 14.2** — DVC: `rm -rf data/processed/ data/features/ && dvc repro` → hashes match
- [ ] **Step 14.3** — MLflow: all 8 runs (4 models + 4 ablations) have loss curves and checkpoint artifacts
- [ ] **Step 14.4** — `docker-compose up --build` succeeds on a clean machine; 3 services healthy
- [ ] **Step 14.5** — Gradio: verify recommendations for user IDs 1, 42, 100; change context sliders and confirm ranking changes
- [ ] **Step 14.6** — `outputs/ablation_table.csv`: ≥1 context variant strictly outperforms NCF baseline on NDCG@10
- [ ] **Step 14.7** — `outputs/figures/shap_values.npy` shape is `(500, 9)`; all 9 dims covered
- [ ] **Step 14.8** — All 3 deliverable docs render on GitHub; no broken image links
- [ ] **Step 14.9** — `git log --oneline` ≥ 1 commit per phase with descriptive messages
- [ ] **Step 14.10** — Repo accessible to instructor; README CI badge is green

---

## Tools reference

| Task                     | Tool / Library                            |
| ------------------------ | ----------------------------------------- |
| Package management       | `uv`                                      |
| Data loading & wrangling | `pandas`, `polars`                        |
| Feature engineering      | `numpy`, `scikit-learn`                   |
| Config management        | `OmegaConf`, `params.yaml`                |
| Model training           | `torch`, `pytorch-lightning`              |
| CLI entry point          | `typer` (via `main.py`)                   |
| Experiment tracking      | `mlflow`, `wandb`                         |
| Hyperparameter search    | `wandb sweep`                             |
| Data versioning          | `dvc`                                     |
| Model export             | `torch.jit.script`, `onnx`, `onnxruntime` |
| Interpretability         | `shap`, `matplotlib`, `seaborn`           |
| API serving              | `fastapi`, `uvicorn`, `pydantic`          |
| Caching                  | `redis`                                   |
| UI demo                  | `gradio`                                  |
| Containerization         | `docker` multi-stage, `docker-compose`    |
| CI pipeline              | `github actions`                          |
| Testing                  | `pytest`, `httpx`                         |
| Notebooks                | `jupyterlab`                              |

