"""
test_eval_interface.py
======================

Validates MLE_AGENT/Amazon_Reviews_Task/eval_interface.run_evaluation()
against the original evaluate() in sasrec/utils.py using the same SASRec
checkpoint, so results should be nearly identical.

Usage
-----
    python test_eval_interface.py \
        --state_dict_path /path/to/SASRec_saving.epoch=200....pth \
        --dataset         Industrial_and_Scientific \
        --data_dir        /path/to/data_dir \
        --test_dir        /path/to/test_dir \
        --device          cpu \
        --maxlen          128 \
        --seed            42

Note on expected differences
-----------------------------
Results may differ slightly (< 0.01) due to minor protocol differences
between the two evaluators (user ordering, negative exclusion sets).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
import types

import numpy as np
import torch

# ------------------------------------------------------------------ #
# Path setup                                                          #
# ------------------------------------------------------------------ #
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))          # LLM-SRec/SeqRec
SASREC_DIR  = os.path.join(SCRIPT_DIR, 'sasrec')
# AGENT_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, '../../MLE_AGENT/Amazon_Reviews_Task'))
AGENT_DIR = '/mmu_vcg2_wjc_ssd/songzeyu/MLE_AGENT/Amazon_Reviews_Task'

for p in [SASREC_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model import SASRec                          # noqa: E402
from utils import evaluate, data_partition        # noqa: E402


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def load_model(state_dict_path: str, device: str) -> SASRec:
    kwargs, state_dict = torch.load(state_dict_path, map_location=device)
    kwargs['args'].device = device
    model = SASRec(**kwargs).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_original_evaluate(model, dataset, args_ns, seed: int) -> dict:
    """Call utils.evaluate() for @10 and @20, reset seed each time."""
    random.seed(seed); np.random.seed(seed)
    ndcg10, hr10 = evaluate(model, dataset, args_ns, mode=1, ranking=10)
    random.seed(seed); np.random.seed(seed)
    ndcg20, hr20 = evaluate(model, dataset, args_ns, mode=1, ranking=20)
    return {
        'HR@10':   round(hr10,   4),
        'HR@20':   round(hr20,   4),
        'NDCG@10': round(ndcg10, 4),
        'NDCG@20': round(ndcg20, 4),
    }


def make_predict_fn(model: SASRec, maxlen: int):
    """
    Build a predict callable that wraps model.predict() directly.
    Signature matches eval_interface contract:
        predict(user_id, history, candidates) -> List[float]
    """
    def predict(user_id: int, history: list, candidates: list) -> list:
        seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for item in reversed(history):
            seq[idx] = item
            idx -= 1
            if idx == -1:
                break

        with torch.no_grad():
            logits = model.predict(
                np.array([user_id]),
                np.array([seq]),
                np.array(candidates),
            )
            scores = logits.squeeze().cpu().numpy()

        if scores.ndim == 0:
            scores = scores.reshape(1)
        return scores.tolist()

    return predict


def run_eval_interface(
    model: SASRec,
    dataset_name: str,
    data_dir: str,
    test_dir: str,
    maxlen: int,
    device: str,
    seed: int,
    max_eval_users: int,
) -> dict:
    """
    Load eval_interface from AGENT_DIR and run it using model.predict() directly.
    """
    # Build predict function from model.predict()
    predict_fn = make_predict_fn(model, maxlen)

    # Load eval_interface module
    ei_path = os.path.join(AGENT_DIR, 'eval_interface.py')
    spec = importlib.util.spec_from_file_location('eval_interface', ei_path)
    ei = importlib.util.module_from_spec(spec)
    if AGENT_DIR not in sys.path:
        sys.path.insert(0, AGENT_DIR)
    spec.loader.exec_module(ei)

    # Patch _load_predict_fn to return our closure directly
    ei._load_predict_fn = lambda path: predict_fn

    return ei.run_evaluation(
        dataset_name   = dataset_name,
        predict_script = 'model.predict',   # placeholder, not actually read
        data_dir       = data_dir,
        test_dir       = test_dir,
        num_candidates = 99,
        max_eval_users = max_eval_users,
        seed           = seed,
    )


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description='Compare eval_interface.run_evaluation() with original evaluate().'
    )
    parser.add_argument('--state_dict_path', required=True,
                        help='Path to SASRec .pth checkpoint (same as main.py --state_dict_path)')
    parser.add_argument('--dataset',         required=True,
                        help='Dataset name, e.g. Industrial_and_Scientific')
    parser.add_argument('--data_dir',        default=None,
                        help='Directory with *_train.txt and *_valid.txt')
    parser.add_argument('--test_dir',        default=None,
                        help='Directory with *_test.txt (defaults to data_dir)')
    parser.add_argument('--device',          default='cpu')
    parser.add_argument('--maxlen',          default=128, type=int)
    parser.add_argument('--seed',            default=42, type=int)
    parser.add_argument('--max_eval_users',  default=10000, type=int)
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(SCRIPT_DIR, '..', f'data_{args.dataset}')
    test_dir = args.test_dir or data_dir

    # ---- Load model ------------------------------------------------- #
    print(f"Loading checkpoint: {args.state_dict_path}")
    model = load_model(args.state_dict_path, args.device)
    print(f"Model loaded.  device={args.device}  maxlen={args.maxlen}")

    args_ns = types.SimpleNamespace(
        maxlen  = args.maxlen,
        device  = args.device,
        dataset = args.dataset,
    )

    # ---- Load dataset ----------------------------------------------- #
    print("\nLoading dataset ...")
    dataset = data_partition(args.dataset, args_ns, data_dir=data_dir, test_dir=test_dir)
    usernum, itemnum = dataset[3], dataset[4]
    print(f"usernum={usernum}  itemnum={itemnum}")

    # ---- Original evaluate() ---------------------------------------- #
    print("\n========== original evaluate() from sasrec/utils.py ==========")
    orig = run_original_evaluate(model, dataset, args_ns, seed=args.seed)
    print("\nResults:")
    for k, v in orig.items():
        print(f"  {k}: {v:.4f}")

    # ---- eval_interface --------------------------------------------- #
    print("\n========== eval_interface.run_evaluation() ==========")
    ei = run_eval_interface(
        model          = model,
        dataset_name   = args.dataset,
        data_dir       = data_dir,
        test_dir       = test_dir,
        maxlen         = args.maxlen,
        device         = args.device,
        seed           = args.seed,
        max_eval_users = args.max_eval_users,
    )
    print("\nResults:")
    for k, v in ei.items():
        print(f"  {k}: {v:.4f}")

    # ---- Side-by-side comparison ------------------------------------ #
    print("\n========== Comparison ==========")
    print(f"{'Metric':<12} {'Original':>10} {'Interface':>10} {'Delta':>10}")
    print("-" * 46)
    all_close = True
    for k in ['HR@10', 'HR@20', 'NDCG@10', 'NDCG@20']:
        o, e  = orig.get(k, float('nan')), ei.get(k, float('nan'))
        delta = e - o
        flag  = "  *** MISMATCH ***" if abs(delta) > 0.01 else ""
        if abs(delta) > 0.01:
            all_close = False
        print(f"{k:<12} {o:>10.4f} {e:>10.4f} {delta:>+10.4f}{flag}")

    print()
    if all_close:
        print("✓  eval_interface is consistent with original evaluate() (|delta| ≤ 0.01).")
    else:
        print("✗  Significant discrepancies — check eval_interface implementation.")

    return 0 if all_close else 1


if __name__ == '__main__':
    sys.exit(main())
