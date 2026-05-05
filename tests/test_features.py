from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.features.context import CONTEXT_COLS, build_context_vector
from src.data.features.device import encode_device_proxy
from src.data.features.negative_sampling import sample_negatives_eval, sample_negatives_train
from src.data.features.session import encode_session, synthesize_sessions
from src.data.features.temporal import encode_temporal


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 2, 2, 3, 3],
            "item_id": [0, 1, 0, 2, 1, 3, 2, 4],
            "rating": [5.0, 4.0, 3.5, 4.5, 2.0, 5.0, 3.0, 4.0],
            "timestamp": [
                1_700_000_000,
                1_700_000_300,
                1_700_001_200,
                1_700_001_500,
                1_700_002_000,
                1_700_004_800,
                1_700_005_000,
                1_700_008_900,
            ],
        }
    )


def _feature_df(tmp_path) -> pd.DataFrame:
    df = _toy_df()
    df = synthesize_sessions(df, gap_seconds=3600)
    scaler_path = tmp_path / "session_scaler.pkl"
    df = encode_temporal(df)
    df = encode_session(df, str(scaler_path), fit=True)
    df = encode_device_proxy(df)
    return df


def test_temporal_cyclic_features_are_bounded_and_unit_circle(tmp_path) -> None:
    df = _feature_df(tmp_path)

    for col in ["sin_hour", "cos_hour", "sin_dow", "cos_dow"]:
        assert ((df[col] >= -1.0) & (df[col] <= 1.0)).all()

    hour_norm = (df["sin_hour"] ** 2 + df["cos_hour"] ** 2).to_numpy(dtype=np.float64)
    dow_norm = (df["sin_dow"] ** 2 + df["cos_dow"] ** 2).to_numpy(dtype=np.float64)
    assert np.allclose(hour_norm, 1.0, atol=1e-6)
    assert np.allclose(dow_norm, 1.0, atol=1e-6)


def test_build_context_vector_contract_shape_dtype_and_nans(tmp_path) -> None:
    df = _feature_df(tmp_path)
    ctx = build_context_vector(df)

    assert CONTEXT_COLS == [
        "sin_hour",
        "cos_hour",
        "sin_dow",
        "cos_dow",
        "session_pos_norm",
        "session_len_norm",
        "device_0",
        "device_1",
        "device_2",
    ]
    assert ctx.shape == (len(df), 9)
    assert ctx.dtype == np.float32
    assert not np.isnan(ctx).any()


def test_session_normalizations_are_in_0_1(tmp_path) -> None:
    df = _feature_df(tmp_path)

    assert ((df["session_pos_norm"] >= 0.0) & (df["session_pos_norm"] <= 1.0)).all()
    assert ((df["session_len_norm"] >= 0.0) & (df["session_len_norm"] <= 1.0)).all()


def test_device_one_hot_has_single_active_dimension(tmp_path) -> None:
    df = _feature_df(tmp_path)
    one_hot_sum = df[["device_0", "device_1", "device_2"]].sum(axis=1)
    assert (one_hot_sum == 1).all()


def test_train_negatives_never_overlap_user_positives() -> None:
    df = _toy_df()
    sampled = sample_negatives_train(df, n_neg=4, seed=7)

    user_items = (
        df.groupby("user_id")["item_id"]
        .apply(lambda x: set(int(item_id) for item_id in x.tolist()))
        .to_dict()
    )

    assert "item_neg" in sampled.columns
    for _, row in sampled.iterrows():
        negs = row["item_neg"]
        assert isinstance(negs, list)
        for neg_id in negs:
            assert int(neg_id) not in user_items[int(row["user_id"])]


def test_eval_negatives_have_positive_first_and_length_100() -> None:
    df = _toy_df()
    all_items = np.arange(12, dtype=np.int64)
    sampled = sample_negatives_eval(df, all_items, n_neg=99, seed=9)

    for _, row in sampled.iterrows():
        candidates = row["candidates"]
        assert isinstance(candidates, list)
        assert len(candidates) == 100
        assert int(candidates[0]) == int(row["item_id"])
