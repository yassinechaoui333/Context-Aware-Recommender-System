## Processed interaction schema

| Column      | Type  | Notes                          |
| ----------- | ----- | ------------------------------ |
| user_id     | int   | 0-indexed via LabelEncoder     |
| item_id     | int   | 0-indexed via LabelEncoder     |
| timestamp   | int   | Unix seconds                   |
| rating      | float | 1.0-5.0                        |
| split       | str   | train / val / test             |
| session_id  | int   | synthesized from rating bursts |
| session_pos | int   | 0-indexed within session       |
| session_len | int   | total ratings in session       |

## Context vector - 9 dimensions, always this order

[sin_hour, cos_hour, sin_dow, cos_dow,
session_pos_norm, session_len_norm,
device_0, device_1, device_2]

## Model interface contract

predict(user_id: int, item_ids: List[int], context: Dict) -> List[float]

## Checkpoint naming

outputs/checkpoints/{experiment_name}/best.ckpt
