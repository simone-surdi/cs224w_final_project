# =============================================================================
# FILE: ab_train_with_antigen.py
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import json
import csv
import math
import random
import sys
import numpy as np
import argparse
import os

from structgen.data_with_antigen import (
    AntibodyAntigenDataset, 
    StructureLoader, 
    completize, 
    completize_with_antigen
)
from structgen.utils import compute_rmsd
from structgen.hierarchical_with_antigen import AntigenAwareHierarchicalDecoder
from tqdm import tqdm


MIN_AG_LENGTH = 5


def evaluate(model, loader, args):
    model.eval()
    val_nll = val_tot = 0.
    val_rmsd = []
    n_skipped = 0
    
    with torch.no_grad():
        for hbatch in tqdm(loader, desc="Evaluating"):
            hX, hS, hL, hmask, hX_ag, hS_ag, hmask_ag = completize_with_antigen(hbatch)
            
            for i in range(len(hbatch)):
                L = hmask[i].sum().long().item()
                if L == 0:
                    continue
                
                X_ag_i = None
                S_ag_i = None
                mask_ag_i = None
                
                if args.use_antigen and hX_ag is not None and hmask_ag is not None:
                    L_ag = hmask_ag[i].sum().long().item()
                    if L_ag >= MIN_AG_LENGTH:
                        X_ag_i = hX_ag[i:i+1, :L_ag]
                        S_ag_i = hS_ag[i:i+1, :L_ag]
                        mask_ag_i = hmask_ag[i:i+1, :L_ag]
                
                try:
                    out = model.log_prob(
                        hS[i:i+1, :L], 
                        [hL[i]], 
                        hmask[i:i+1, :L],
                        X_ag=X_ag_i,
                        S_ag=S_ag_i,
                        mask_ag=mask_ag_i
                    )
                    
                    nll, X_pred = out.nll, out.X_cdr
                    val_nll += nll.item() * hL[i].count(args.cdr_type)
                    val_tot += hL[i].count(args.cdr_type)
                    
                    l, r = hL[i].index(args.cdr_type), hL[i].rindex(args.cdr_type)
                    rmsd = compute_rmsd(
                        X_pred[:, :, 1, :], 
                        hX[i:i+1, l:r+1, 1, :], 
                        hmask[i:i+1, l:r+1]
                    )
                    val_rmsd.append(rmsd.item())
                    
                except Exception as e:
                    n_skipped += 1
                    if n_skipped <= 5:
                        print(f"  Skipping sample {i}: {e}")
                    continue

    if n_skipped > 0:
        print(f"  Skipped {n_skipped} samples due to errors")
    
    ppl = math.exp(val_nll / val_tot) if val_tot > 0 else float('inf')
    avg_rmsd = sum(val_rmsd) / len(val_rmsd) if val_rmsd else float('inf')
    
    return ppl, avg_rmsd


parser = argparse.ArgumentParser()
parser.add_argument('--train_path', default='data/sabdab/hcdr3_cluster/train_data.jsonl')
parser.add_argument('--val_path', default='data/sabdab/hcdr3_cluster/val_data.jsonl')
parser.add_argument('--test_path', default='data/sabdab/hcdr3_cluster/test_data.jsonl')
parser.add_argument('--save_dir', default='ckpts/antigen_aware')
parser.add_argument('--load_model', default=None)

parser.add_argument('--cdr_type', default='3')

parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--batch_tokens', type=int, default=100)
parser.add_argument('--k_neighbors', type=int, default=9)
parser.add_argument('--block_size', type=int, default=8)
parser.add_argument('--update_freq', type=int, default=1)
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--vocab_size', type=int, default=21)
parser.add_argument('--num_rbf', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('--use_antigen', action='store_true', default=True)
parser.add_argument('--no_antigen', action='store_true', default=False)
parser.add_argument('--use_distance_bias', action='store_true', default=True)
parser.add_argument('--cross_k_neighbors', type=int, default=8)
parser.add_argument('--interface_cutoff', type=float, default=10.0)
parser.add_argument('--max_ag_len', type=int, default=200)
parser.add_argument('--require_antigen', action='store_true', default=False)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=5.0)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--print_iter', type=int, default=50)

args = parser.parse_args()

if args.no_antigen:
    args.use_antigen = False

print(args)

os.makedirs(args.save_dir, exist_ok=True)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

loaders = []
for path in [args.train_path, args.val_path, args.test_path]:
    data = AntibodyAntigenDataset(
        path, 
        cdr_type=args.cdr_type,
        max_ag_len=args.max_ag_len,
        require_antigen=args.require_antigen
    )
    loader = StructureLoader(
        data.data, 
        batch_tokens=args.batch_tokens, 
        interval_sort=int(args.cdr_type)
    )
    loaders.append(loader)

loader_train, loader_val, loader_test = loaders

model = AntigenAwareHierarchicalDecoder(args).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

if args.load_model:
    model_ckpt, opt_ckpt, model_args = torch.load(args.load_model, weights_only=False)
    model = AntigenAwareHierarchicalDecoder(model_args).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.load_state_dict(model_ckpt)
    optimizer.load_state_dict(opt_ckpt)

print(f'Training:{len(loader_train.dataset)}, Validation:{len(loader_val.dataset)}, Test:{len(loader_test.dataset)}')
print(f'Use antigen: {args.use_antigen}')

best_ppl, best_epoch = 100, -1

for e in range(args.epochs):
    model.train()
    meter = 0
    n_batches = 0
    
    for i, hbatch in enumerate(tqdm(loader_train)):
        optimizer.zero_grad()
        
        hX, hS, hL, hmask, hX_ag, hS_ag, hmask_ag = completize_with_antigen(hbatch)
        
        if hmask.sum().item() == 0:
            continue
        
        use_ag = args.use_antigen and hX_ag is not None and hmask_ag is not None
        if use_ag:
            ag_lengths = hmask_ag.sum(dim=1)
            if ag_lengths.min().item() < MIN_AG_LENGTH:
                use_ag = False
        
        loss, snll = model(
            hX, hS, hL, hmask,
            X_ag=hX_ag if use_ag else None,
            S_ag=hS_ag if use_ag else None,
            mask_ag=hmask_ag if use_ag else None
        )
        
        loss.backward()
        
        if args.clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        
        optimizer.step()
        
        meter += snll.exp().item()
        n_batches += 1
        
        if (i + 1) % args.print_iter == 0:
            avg_ppl = meter / n_batches if n_batches > 0 else 0
            print(f'[{i + 1}] Train PPL = {avg_ppl:.3f}')
            meter = 0
            n_batches = 0
    
    val_ppl, val_rmsd = evaluate(model, loader_val, args)
    
    ckpt = (model.state_dict(), optimizer.state_dict(), args)
    torch.save(ckpt, os.path.join(args.save_dir, f"model.ckpt.{e}"))
    print(f'Epoch {e}, Val PPL = {val_ppl:.3f}, Val RMSD = {val_rmsd:.3f}')
    
    if val_ppl < best_ppl:
        best_ppl = val_ppl
        best_epoch = e
        torch.save(ckpt, os.path.join(args.save_dir, "model.best.ckpt"))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] *= args.anneal_rate

if best_epoch >= 0:
    best_ckpt = os.path.join(args.save_dir, "model.best.ckpt")
    model.load_state_dict(torch.load(best_ckpt, weights_only=False)[0])

test_ppl, test_rmsd = evaluate(model, loader_test, args)
print(f'Test PPL = {test_ppl:.3f}, Test RMSD = {test_rmsd:.3f}')

with open(file_path, 'w') as f:
    f.write(script_content)

print("✓ ab_train_with_antigen.py saved successfully!")