"""
test_data_interface.py
======================

Trains two SASRec models from scratch:
  1. Model A — using ``data_interface.WarpSampler`` (from MLE_AGENT)
  2. Model B — using ``sasrec/utils.WarpSampler`` (original)

Then evaluates both with ``sasrec/utils.evaluate()`` and prints a
side-by-side comparison to verify the two samplers produce equivalent
training signal.

Purpose
-------
  Verify that ``data_interface.WarpSampler`` produces training batches
  equivalent to the original ``sasrec/utils.WarpSampler``: both models
  should reach similar HR / NDCG after the same number of epochs
  (|delta| <= 0.01 is treated as a pass).

Usage
-----
    # Train both models and compare:
    python test_data_interface.py \\
        --dataset    Industrial_and_Scientific \\
        --data_dir   /path/to/data_dir \\
        --test_dir   /path/to/test_dir \\
        --device     cpu \\
        --num_epochs 5 \\
        --batch_size 128 \\
        --maxlen     128 \\
        --seed       42

    # Load two existing checkpoints instead of training:
    python test_data_interface.py \\
        --dataset               Industrial_and_Scientific \\
        --data_dir              /path/to/data_dir \\
        --test_dir              /path/to/test_dir \\
        --state_dict_path_di    /path/to/model_data_interface.pth \\
        --state_dict_path_orig  /path/to/model_original.pth \\
        --device                cpu \\
        --maxlen                128 \\
        --seed                  42

Notes
-----
- Both models are initialised with the same random seed so that only
  the sampler differs.
- The SASRec model and utilities are imported from sasrec/ (same directory).
"""

from __future__ import annotations

import argparse
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

from model import SASRec                                        # noqa: E402  (from sasrec/)
from utils import evaluate, data_partition                      # noqa: E402  (from sasrec/)
from utils import WarpSampler as OrigWarpSampler                # noqa: E402  (from sasrec/)
from data_interface import WarpSampler as DIWarpSampler         # noqa: E402  (from AGENT_DIR)
from data_interface import load_dataset as di_load_dataset      # noqa: E402  (from AGENT_DIR)


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


def _make_model(usernum: int, itemnum: int, args_ns: types.SimpleNamespace,
                seed: int) -> SASRec:
    """Create and Xavier-initialise a fresh SASRec model with fixed seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = SASRec(usernum, itemnum, args_ns).to(args_ns.device)
    for _, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass
    return model


def _train_loop(
    sampler,
    model: SASRec,
    user_train: dict,
    args_ns: types.SimpleNamespace,
    num_epochs: int,
    batch_size: int,
    lr: float,
    label: str,
    save_path: str | None,
) -> SASRec:
    """Shared training loop; sampler must implement .next_batch() / .close()."""
    device    = args_ns.device
    num_batch = len(user_train) // batch_size

    bce_criterion  = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                      betas=(0.9, 0.98))

    print(f"\n[Training — {label}]  epochs={num_epochs}  batch_size={batch_size}"
          f"  num_batch={num_batch}  lr={lr}")

    start_time = time.time()
    for epoch in tqdm(range(1, num_epochs + 1)):
        model.train()
        total_loss, count = 0.0, 0

        for step in range(num_batch):
            u, seq, pos, neg = sampler.next_batch()
            u   = np.array(u)
            seq = np.array(seq)
            pos = np.array(pos)
            neg = np.array(neg)
            # data_interface neg has shape (batch, maxlen, num_neg);
            # squeeze last dim when num_neg==1 to match original (batch, maxlen)
            if neg.ndim == 3 and neg.shape[-1] == 1:
                neg = neg[..., 0]

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

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save([model.kwargs, model.state_dict()], save_path)
        print(f"Checkpoint saved -> {save_path}")

    model.eval()
    return model


def train_sasrec_di(
    dataset_name: str,
    data_dir: str,
    usernum: int,
    itemnum: int,
    args_ns: types.SimpleNamespace,
    num_epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    save_path: str | None,
) -> SASRec:
    """Train SASRec using data_interface.load_dataset + WarpSampler (MLE_AGENT)."""
    # Load dataset via data_interface (not data_partition)
    train_data, _ = di_load_dataset(
        dataset_name = dataset_name,
        data_dir     = data_dir,
        num_neg      = 1,
        seed         = seed,
    )
    # Reconstruct user_train dict required by WarpSampler and _train_loop
    user_train: dict = {}
    for sample in train_data:
        uid = sample['user_id']
        user_train[uid] = sample['history'] + [sample['target']]

    sampler = DIWarpSampler(
        user_train, usernum, itemnum,
        batch_size = batch_size,
        maxlen     = args_ns.maxlen,
        num_neg    = 1,
        n_workers  = 3,
    )
    model = _make_model(usernum, itemnum, args_ns, seed)
    return _train_loop(sampler, model, user_train, args_ns,
                       num_epochs, batch_size, lr,
                       label='data_interface.WarpSampler (MLE_AGENT)',
                       save_path=save_path)


def train_sasrec_orig(
    user_train: dict,
    usernum: int,
    itemnum: int,
    args_ns: types.SimpleNamespace,
    num_epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    save_path: str | None,
) -> SASRec:
    """Train SASRec using sasrec/utils.WarpSampler (original)."""
    sampler = OrigWarpSampler(
        user_train, usernum, itemnum,
        batch_size = batch_size,
        maxlen     = args_ns.maxlen,
        n_workers  = 3,
    )
    model = _make_model(usernum, itemnum, args_ns, seed)
    return _train_loop(sampler, model, user_train, args_ns,
                       num_epochs, batch_size, lr,
                       label='sasrec/utils.WarpSampler (original)',
                       save_path=save_path)


# ------------------------------------------------------------------ #
# Evaluation helpers                                                  #
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


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Train two SASRec models (data_interface vs original WarpSampler) '
            'and compare their metrics with sasrec/utils.evaluate().'
        )
    )
    # --- data ---
    parser.add_argument('--dataset',  required=True)
    parser.add_argument('--data_dir', default=None,
                        help='Dir with *_train.txt and *_valid.txt')
    parser.add_argument('--test_dir', default=None,
                        help='Dir with *_test.txt (defaults to data_dir)')

    # --- model / training ---
    parser.add_argument('--state_dict_path_di',   default=None,
                        help='Skip training model-A and load this checkpoint instead')
    parser.add_argument('--state_dict_path_orig', default=None,
                        help='Skip training model-B and load this checkpoint instead')
    parser.add_argument('--num_epochs',   default=5,     type=int)
    parser.add_argument('--batch_size',   default=128,   type=int)
    parser.add_argument('--lr',           default=0.001, type=float)
    parser.add_argument('--maxlen',       default=128,   type=int)
    parser.add_argument('--hidden_units', default=64,    type=int)
    parser.add_argument('--num_blocks',   default=2,     type=int)
    parser.add_argument('--num_heads',    default=1,     type=int)
    parser.add_argument('--dropout_rate', default=0.1,   type=float)
    parser.add_argument('--l2_emb',       default=0.0,   type=float)
    parser.add_argument('--device',       default='cpu')

    # --- evaluation ---
    parser.add_argument('--seed',           default=42,    type=int)
    parser.add_argument('--max_eval_users', default=10000, type=int)
    parser.add_argument('--save_path_di',   default=None,
                        help='Where to save model-A checkpoint (optional)')
    parser.add_argument('--save_path_orig', default=None,
                        help='Where to save model-B checkpoint (optional)')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device   = _normalize_device(args.device)
    data_dir = args.data_dir or os.path.join(SCRIPT_DIR, '..', f'data_{args.dataset}')
    test_dir = args.test_dir or data_dir
    args_ns  = _build_args_ns(args, device)

    # ---- Load dataset ------------------------------------------------- #
    print("\nLoading dataset ...")
    dataset = data_partition(args.dataset, args_ns,
                             data_dir=data_dir, test_dir=test_dir)
    user_train, _, _, usernum, itemnum, _ = dataset
    print(f"usernum={usernum}  itemnum={itemnum}")

    # ---- Model A: data_interface.WarpSampler -------------------------- #
    print("\n" + "=" * 60)
    print("Model A: data_interface.WarpSampler (MLE_AGENT)")
    print("=" * 60)
    if args.state_dict_path_di is not None:
        print(f"Loading checkpoint: {args.state_dict_path_di}")
        model_di = load_checkpoint(args.state_dict_path_di, device)
    else:
        fname_di = (
            f'SASRec_DI.epoch={args.num_epochs}.lr={args.lr}'
            f'.layer={args.num_blocks}.head={args.num_heads}'
            f'.hidden={args.hidden_units}.maxlen={args.maxlen}'
            f'.dropout={args.dropout_rate}.pth'
        )
        save_path_di = args.save_path_di or os.path.join(args.dataset, fname_di)
        model_di = train_sasrec_di(
            dataset_name = args.dataset,
            data_dir     = data_dir,
            usernum      = usernum,
            itemnum      = itemnum,
            args_ns      = args_ns,
            num_epochs   = args.num_epochs,
            batch_size   = args.batch_size,
            lr           = args.lr,
            seed         = args.seed,
            save_path    = save_path_di,
        )

    # ---- Model B: sasrec/utils.WarpSampler ---------------------------- #
    print("\n" + "=" * 60)
    print("Model B: sasrec/utils.WarpSampler (original)")
    print("=" * 60)
    if args.state_dict_path_orig is not None:
        print(f"Loading checkpoint: {args.state_dict_path_orig}")
        model_orig = load_checkpoint(args.state_dict_path_orig, device)
    else:
        fname_orig = (
            f'SASRec_ORIG.epoch={args.num_epochs}.lr={args.lr}'
            f'.layer={args.num_blocks}.head={args.num_heads}'
            f'.hidden={args.hidden_units}.maxlen={args.maxlen}'
            f'.dropout={args.dropout_rate}.pth'
        )
        save_path_orig = args.save_path_orig or os.path.join(args.dataset, fname_orig)
        model_orig = train_sasrec_orig(
            user_train = user_train,
            usernum    = usernum,
            itemnum    = itemnum,
            args_ns    = args_ns,
            num_epochs = args.num_epochs,
            batch_size = args.batch_size,
            lr         = args.lr,
            seed       = args.seed,
            save_path  = save_path_orig,
        )

    # ---- Evaluate both models with sasrec/utils.evaluate() ------------ #
    print("\n========== Evaluating Model A (data_interface.WarpSampler) ==========")
    metrics_di = run_original_evaluate(model_di, dataset, args_ns, seed=args.seed)
    print("Results:")
    for k, v in metrics_di.items():
        print(f"  {k}: {v:.4f}")

    print("\n========== Evaluating Model B (sasrec/utils.WarpSampler) ==========")
    metrics_orig = run_original_evaluate(model_orig, dataset, args_ns, seed=args.seed)
    print("Results:")
    for k, v in metrics_orig.items():
        print(f"  {k}: {v:.4f}")

    # ---- Side-by-side comparison --------------------------------------- #
    print("\n========== Comparison ==========")
    print(f"{'Metric':<12} {'Model-A (DI)':>14} {'Model-B (Orig)':>14} {'Delta':>10}")
    print("-" * 54)
    all_close = True
    for k in ['HR@10', 'HR@20', 'NDCG@10', 'NDCG@20']:
        a     = metrics_di.get(k,   float('nan'))
        b     = metrics_orig.get(k, float('nan'))
        delta = a - b
        flag  = "  *** MISMATCH ***" if abs(delta) > 0.01 else ""
        if abs(delta) > 0.01:
            all_close = False
        print(f"{k:<12} {a:>14.4f} {b:>14.4f} {delta:>+10.4f}{flag}")

    print()
    if all_close:
        print("PASS  Both samplers produce models with consistent metrics "
              "(|delta| <= 0.01).")
        print("      data_interface.WarpSampler is equivalent to the original.")
    else:
        print("FAIL  Significant metric gap between the two models.")
        print("      Check data_interface.WarpSampler for sampling differences.")

    return 0 if all_close else 1


if __name__ == '__main__':
    sys.exit(main())
