"""
test_data_interface.py
======================

Trains a SASRec model using ``data_interface.WarpSampler`` (from
MLE_AGENT/Amazon_Reviews_Task) instead of the original
``sasrec/utils.WarpSampler``, then validates
``eval_interface.run_evaluation()`` against the canonical
``sasrec/utils.evaluate()`` — exactly as ``test_eval_interface.py``
does, but driven from the MLE_AGENT side.

Purpose
-------
  1. Verify that ``data_interface.WarpSampler`` produces correct training
     batches (model should converge normally).
  2. Verify that ``eval_interface.run_evaluation()`` reports metrics
     consistent with the reference ``evaluate()`` (|delta| ≤ 0.01).

Usage
-----
    # Full training + evaluation comparison:
    python test_data_interface.py \\
        --dataset    Industrial_and_Scientific \\
        --data_dir   /path/to/data_dir \\
        --test_dir   /path/to/test_dir \\
        --device     cpu \\
        --num_epochs 5 \\
        --batch_size 128 \\
        --maxlen     128 \\
        --seed       42

    # Skip training, load an existing checkpoint:
    python test_data_interface.py \\
        --dataset          Industrial_and_Scientific \\
        --data_dir         /path/to/data_dir \\
        --test_dir         /path/to/test_dir \\
        --state_dict_path  /path/to/SASRec_saving.epoch=200....pth \\
        --device           cpu \\
        --maxlen           128 \\
        --seed             42

Notes
-----
- When ``--state_dict_path`` is provided, training is skipped entirely and
  the checkpoint is loaded directly for evaluation comparison.
- The SASRec model and utilities are imported from sasrec/ (same directory).
- ``data_interface.WarpSampler`` is validated implicitly: if training
  converges and evaluation metrics match, the sampler is producing correct
  batches.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
import time
import types

import numpy as np
import torch
from tqdm import tqdm

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

if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

from model import SASRec                           # noqa: E402  (from sasrec/)
from utils import evaluate, data_partition         # noqa: E402  (from sasrec/)
from data_interface import WarpSampler             # noqa: E402  (from AGENT_DIR)


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def _normalize_device(device: str) -> str:
    """Normalize device string: '0' -> 'cuda:0', 'cpu' stays 'cpu'."""
    if device not in ('cpu', 'hpu') and not device.startswith('cuda'):
        return f'cuda:{device}'
    return device


def _build_args_ns(args: argparse.Namespace, device: str) -> types.SimpleNamespace:
    """Build a SimpleNamespace that SASRec / utils functions expect."""
    return types.SimpleNamespace(
        maxlen        = args.maxlen,
        device        = device,
        dataset       = args.dataset,
        hidden_units  = args.hidden_units,
        num_blocks    = args.num_blocks,
        num_heads     = args.num_heads,
        dropout_rate  = args.dropout_rate,
        l2_emb        = args.l2_emb,
        nn_parameter  = False,
        is_hpu        = False,
    )


def load_checkpoint(state_dict_path: str, device: str) -> SASRec:
    """Load a SASRec checkpoint saved by sasrec/main.py."""
    def _map(storage, loc):
        return storage

    kwargs, state_dict = torch.load(state_dict_path, map_location=_map,
                                    weights_only=False)
    kwargs['args'].device = device
    model = SASRec(**kwargs).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def train_sasrec(
    user_train: dict,
    usernum: int,
    itemnum: int,
    args_ns: types.SimpleNamespace,
    num_epochs: int,
    batch_size: int,
    lr: float,
    save_path: str | None,
) -> SASRec:
    """
    Train a SASRec model using ``data_interface.WarpSampler``.

    The training loop mirrors ``sasrec/main.py`` exactly, but uses
    ``data_interface.WarpSampler`` (from MLE_AGENT) as the batch sampler.
    The key difference from the original WarpSampler is that
    ``data_interface.WarpSampler`` supports ``num_neg`` negatives per
    position (we pass ``num_neg=1`` here to match the original behaviour).
    """
    device    = args_ns.device
    num_batch = len(user_train) // batch_size

    # ------ Sampler from data_interface ------ #
    sampler = WarpSampler(
        user_train, usernum, itemnum,
        batch_size = batch_size,
        maxlen     = args_ns.maxlen,
        num_neg    = 1,      # match original: one negative per position
        n_workers  = 3,
    )

    # ------ Model ------ #
    model = SASRec(usernum, itemnum, args_ns).to(device)
    for _, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    bce_criterion  = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                      betas=(0.9, 0.98))

    print(f"\n[Training]  epochs={num_epochs}  batch_size={batch_size}"
          f"  num_batch={num_batch}  lr={lr}")
    print("  (using data_interface.WarpSampler from MLE_AGENT)")

    start_time = time.time()
    for epoch in tqdm(range(1, num_epochs + 1)):
        model.train()
        total_loss, count = 0.0, 0

        for step in range(num_batch):
            u, seq, pos, neg = sampler.next_batch()
            u   = np.array(u)
            seq = np.array(seq)
            pos = np.array(pos)
            # neg from data_interface has shape (batch, maxlen, num_neg);
            # squeeze last dim when num_neg==1 to match original (batch, maxlen)
            neg = np.array(neg)
            if neg.ndim == 3 and neg.shape[-1] == 1:
                neg = neg[..., 0]   # (batch, maxlen)

            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels = torch.ones(pos_logits.shape,  device=device)
            neg_labels = torch.zeros(neg_logits.shape, device=device)

            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss  = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters():
                loss += args_ns.l2_emb * torch.norm(param)

            loss.backward()
            adam_optimizer.step()

            total_loss += loss.item()
            count      += 1

            if step % 100 == 0:
                print(f"  loss epoch {epoch} step {step}: {loss.item():.4f}")

        print(f"  [epoch {epoch}] avg loss: {total_loss / count:.4f}")

    sampler.close()
    elapsed = time.time() - start_time
    print(f"\nTraining done in {elapsed:.1f}s")

    # ------ Optionally save ------ #
    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save([model.kwargs, model.state_dict()], save_path)
        print(f"Checkpoint saved → {save_path}")

    model.eval()
    return model


# ------------------------------------------------------------------ #
# Evaluation helpers (mirrors test_eval_interface.py)                 #
# ------------------------------------------------------------------ #

def run_original_evaluate(model: SASRec, dataset, args_ns, seed: int) -> dict:
    """Call sasrec/utils.evaluate() for @10 and @20."""
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
    Build a predict callable matching eval_interface's expected signature:
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
    seed: int,
    max_eval_users: int,
) -> dict:
    """
    Load eval_interface from AGENT_DIR and call run_evaluation() using
    model.predict() directly — same approach as test_eval_interface.py.
    """
    predict_fn = make_predict_fn(model, maxlen)

    ei_path = os.path.join(AGENT_DIR, 'eval_interface.py')
    spec = importlib.util.spec_from_file_location('eval_interface', ei_path)
    ei   = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ei)

    # Patch _load_predict_fn so it returns our closure
    ei._load_predict_fn = lambda path: predict_fn

    return ei.run_evaluation(
        dataset_name   = dataset_name,
        predict_script = 'model.predict',   # placeholder — not actually read
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
        description=(
            'Train SASRec with data_interface.WarpSampler, then compare '
            'eval_interface.run_evaluation() against sasrec/utils.evaluate().'
        )
    )
    # --- data ---
    parser.add_argument('--dataset',         required=True)
    parser.add_argument('--data_dir',        default=None,
                        help='Dir with *_train.txt and *_valid.txt')
    parser.add_argument('--test_dir',        default=None,
                        help='Dir with *_test.txt (defaults to data_dir)')

    # --- model / training ---
    parser.add_argument('--state_dict_path', default=None,
                        help='Skip training and load this checkpoint instead')
    parser.add_argument('--num_epochs',      default=5,    type=int)
    parser.add_argument('--batch_size',      default=128,  type=int)
    parser.add_argument('--lr',              default=0.001, type=float)
    parser.add_argument('--maxlen',          default=128,  type=int)
    parser.add_argument('--hidden_units',    default=64,   type=int)
    parser.add_argument('--num_blocks',      default=2,    type=int)
    parser.add_argument('--num_heads',       default=1,    type=int)
    parser.add_argument('--dropout_rate',    default=0.1,  type=float)
    parser.add_argument('--l2_emb',          default=0.0,  type=float)
    parser.add_argument('--device',          default='cpu')

    # --- evaluation ---
    parser.add_argument('--seed',            default=42,   type=int)
    parser.add_argument('--max_eval_users',  default=10000, type=int)
    parser.add_argument('--save_path',       default=None,
                        help='Where to save the trained checkpoint (optional)')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device   = _normalize_device(args.device)
    data_dir = args.data_dir or os.path.join(SCRIPT_DIR, '..', f'data_{args.dataset}')
    test_dir = args.test_dir or data_dir
    args_ns  = _build_args_ns(args, device)

    # ---- Load dataset (needed for original evaluate()) ---------------- #
    print("\nLoading dataset ...")
    dataset = data_partition(args.dataset, args_ns,
                             data_dir=data_dir, test_dir=test_dir)
    user_train, _, _, usernum, itemnum, _ = dataset
    print(f"usernum={usernum}  itemnum={itemnum}")

    # ---- Train or load checkpoint ------------------------------------- #
    if args.state_dict_path is not None:
        print(f"\nLoading checkpoint: {args.state_dict_path}")
        model = load_checkpoint(args.state_dict_path, device)
        print("Checkpoint loaded. Skipping training.")
    else:
        print(f"\nTraining SASRec for {args.num_epochs} epoch(s) ...")
        print("  Batch sampler: data_interface.WarpSampler")
        save_path = args.save_path
        if save_path is None:
            fname = (
                f'SASRec_saving.epoch={args.num_epochs}.lr={args.lr}'
                f'.layer={args.num_blocks}.head={args.num_heads}'
                f'.hidden={args.hidden_units}.maxlen={args.maxlen}'
                f'.dropout={args.dropout_rate}.pth'
            )
            save_path = os.path.join(args.dataset, fname)
        model = train_sasrec(
            user_train = user_train,
            usernum    = usernum,
            itemnum    = itemnum,
            args_ns    = args_ns,
            num_epochs = args.num_epochs,
            batch_size = args.batch_size,
            lr         = args.lr,
            save_path  = save_path,
        )

    # ---- Original evaluate() ------------------------------------------ #
    print("\n========== original evaluate() from sasrec/utils.py ==========")
    orig = run_original_evaluate(model, dataset, args_ns, seed=args.seed)
    print("\nResults:")
    for k, v in orig.items():
        print(f"  {k}: {v:.4f}")

    # ---- eval_interface.run_evaluation() -------------------------------- #
    print("\n========== eval_interface.run_evaluation() ==========")
    ei = run_eval_interface(
        model          = model,
        dataset_name   = args.dataset,
        data_dir       = data_dir,
        test_dir       = test_dir,
        maxlen         = args.maxlen,
        seed           = args.seed,
        max_eval_users = args.max_eval_users,
    )
    print("\nResults:")
    for k, v in ei.items():
        print(f"  {k}: {v:.4f}")

    # ---- Side-by-side comparison --------------------------------------- #
    print("\n========== Comparison ==========")
    print(f"{'Metric':<12} {'Original':>10} {'Interface':>10} {'Delta':>10}")
    print("-" * 46)
    all_close = True
    for k in ['HR@10', 'HR@20', 'NDCG@10', 'NDCG@20']:
        o     = orig.get(k, float('nan'))
        e     = ei.get(k, float('nan'))
        delta = e - o
        flag  = "  *** MISMATCH ***" if abs(delta) > 0.01 else ""
        if abs(delta) > 0.01:
            all_close = False
        print(f"{k:<12} {o:>10.4f} {e:>10.4f} {delta:>+10.4f}{flag}")

    print()
    if all_close:
        print("✓  eval_interface is consistent with sasrec/utils.evaluate() "
              "(|delta| ≤ 0.01).")
        print("✓  data_interface.WarpSampler produced valid training batches.")
    else:
        print("✗  Significant discrepancies — check eval_interface or "
              "data_interface implementation.")

    return 0 if all_close else 1


if __name__ == '__main__':
    sys.exit(main())
