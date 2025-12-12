"""
CDR Inference and Visualization Script
=======================================

Loads a trained model from checkpoint and runs inference on test samples,
then visualizes predicted vs ground truth structures.

Usage in Colab:
---------------
# Basic usage - random sample
!python inference_cdr.py --checkpoint ckpts/cdr3_with_antigen/model.best.ckpt \
                         --test_path data/sabdab/hcdr3_cluster/test_data.jsonl \
                         --cdr_type 3

# Specific sample by index
!python inference_cdr.py --checkpoint ckpts/cdr3_with_antigen/model.best.ckpt \
                         --test_path data/sabdab/hcdr3_cluster/test_data.jsonl \
                         --sample_idx 42

# Multiple samples
!python inference_cdr.py --checkpoint ckpts/cdr3_with_antigen/model.best.ckpt \
                         --test_path data/sabdab/hcdr3_cluster/test_data.jsonl \
                         --n_samples 10 --save_dir predictions/

# As Python module
from inference_cdr import CDRPredictor
predictor = CDRPredictor('ckpts/cdr3_with_antigen/model.best.ckpt')
result = predictor.predict_sample(test_loader, sample_idx=0)
predictor.visualize(result)
"""

import torch
import torch.nn as nn
import numpy as np
import json
import argparse
import os
import random
import math
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Visualization imports (optional)
try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# Amino acid mappings
AA_LIST = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
           'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
IDX_TO_AA = {idx: aa for idx, aa in enumerate(AA_LIST)}


def coords_to_pdb(coords: np.ndarray, sequence: str, chain_id: str = 'A', 
                  start_res: int = 1) -> str:
    """Convert coordinates to PDB format string."""
    
    ATOM_NAMES = ['N', 'CA', 'C', 'O']
    
    pdb_lines = []
    atom_num = 1
    
    # Handle different coordinate formats
    if coords.ndim == 2:
        coords = coords[:, np.newaxis, :]
        atom_names = ['CA']
    elif coords.ndim == 3 and coords.shape[1] == 4:
        atom_names = ATOM_NAMES
    elif coords.ndim == 3 and coords.shape[1] == 1:
        atom_names = ['CA']
    else:
        atom_names = ATOM_NAMES[:coords.shape[1]]
    
    # 1-letter to 3-letter amino acid mapping
    AA_MAP = {
        'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
        'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
        'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
        'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
        'X': 'UNK'
    }
    
    for res_idx, aa in enumerate(sequence):
        res_num = start_res + res_idx
        res_name = AA_MAP.get(aa, 'UNK')
        
        for atom_idx, atom_name in enumerate(atom_names):
            if atom_idx >= coords.shape[1]:
                continue
            
            x, y, z = coords[res_idx, atom_idx]
            
            line = (
                f"ATOM  {atom_num:5d}  {atom_name:<3s} {res_name:3s} {chain_id:1s}"
                f"{res_num:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           "
                f"{atom_name[0]:>2s}  "
            )
            pdb_lines.append(line)
            atom_num += 1
    
    pdb_lines.append("END")
    return "\n".join(pdb_lines)


def kabsch_align(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Align P onto Q using Kabsch algorithm.
    Returns aligned P and RMSD.
    """
    # Center both
    P_centered = P - P.mean(axis=0)
    Q_centered = Q - Q.mean(axis=0)
    
    # Compute rotation matrix
    H = P_centered.T @ Q_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply rotation and translation
    P_aligned = P_centered @ R + Q.mean(axis=0)
    
    # Compute RMSD
    rmsd = np.sqrt(np.mean(np.sum((P_aligned - Q) ** 2, axis=1)))
    
    return P_aligned, rmsd


def compute_rmsd_with_structgen(pred_coords: np.ndarray, true_coords: np.ndarray) -> float:
    """
    Compute RMSD using the same function as training (structgen.utils.compute_rmsd).
    This ensures consistency with training metrics.
    
    Args:
        pred_coords: Predicted coordinates (L, 4, 3) or (L, 3)
        true_coords: True coordinates (L, 4, 3) or (L, 3)
    
    Returns:
        RMSD in Angstroms
    """
    from structgen.utils import compute_rmsd
    
    # Extract CA atoms if full backbone
    if pred_coords.ndim == 3 and pred_coords.shape[1] == 4:
        pred_ca = pred_coords[:, 1, :]  # CA is index 1
        true_ca = true_coords[:, 1, :]
    else:
        pred_ca = pred_coords
        true_ca = true_coords
    
    # Convert to tensors with correct shape (1, L, 3)
    pred_tensor = torch.tensor(pred_ca).unsqueeze(0).float()
    true_tensor = torch.tensor(true_ca).unsqueeze(0).float()
    mask = torch.ones(1, pred_ca.shape[0])
    
    rmsd = compute_rmsd(pred_tensor, true_tensor, mask)
    return rmsd.item()


class CDRPredictor:
    """
    Load trained model and run inference on test samples.
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        """
        Initialize predictor with trained model.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.checkpoint_path = checkpoint_path
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        self.ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Extract args
        if isinstance(self.ckpt, dict):
            self.args = self.ckpt.get('args')
            model_state = self.ckpt.get('model_state_dict')
        elif isinstance(self.ckpt, tuple):
            model_state = self.ckpt[0]
            self.args = self.ckpt[2] if len(self.ckpt) > 2 else None
        else:
            raise ValueError(f"Unknown checkpoint format")
        
        # Import model class
        from structgen.hierarchical_with_antigen import AntigenAwareHierarchicalDecoder
        
        # Build model
        print(f"Building model...")
        self.model = AntigenAwareHierarchicalDecoder(self.args).to(self.device)
        self.model.load_state_dict(model_state)
        self.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"  CDR type: {self.args.cdr_type}")
        print(f"  Use antigen: {self.args.use_antigen}")
        print(f"  Device: {self.device}")
    
    def load_test_data(self, test_path: str, batch_tokens: int = 100):
        """Load test dataset."""
        from structgen.data_with_antigen import AntibodyAntigenDataset, StructureLoader
        
        print(f"Loading test data from {test_path}...")
        dataset = AntibodyAntigenDataset(
            test_path,
            cdr_type=self.args.cdr_type,
            max_ag_len=getattr(self.args, 'max_ag_len', 200),
            require_antigen=getattr(self.args, 'require_antigen', False)
        )
        
        loader = StructureLoader(
            dataset.data,
            batch_tokens=batch_tokens,
            interval_sort=int(self.args.cdr_type)
        )
        
        print(f"  Loaded {len(dataset.data)} samples")
        return loader, dataset.data
    
    def predict_sample(self, loader, sample_idx: int = None, 
                       random_sample: bool = False) -> Dict:
        """
        Run inference on a single sample.
        
        Args:
            loader: Data loader
            sample_idx: Index of sample to predict (None for random)
            random_sample: If True, select random sample
        
        Returns:
            Dict with prediction results
        """
        from structgen.data_with_antigen import completize_with_antigen
        
        # Get total samples
        all_data = list(loader.dataset) if hasattr(loader, 'dataset') else loader.data
        n_samples = len(all_data)
        
        # Select sample
        if sample_idx is None or random_sample:
            sample_idx = random.randint(0, n_samples - 1)
        
        print(f"\nPredicting sample {sample_idx}/{n_samples}...")
        
        # Get the sample
        sample = all_data[sample_idx]
        
        # Prepare batch with single sample
        hX, hS, hL, hmask, hX_ag, hS_ag, hmask_ag = completize_with_antigen([sample])
        
        # Move to device
        hX = hX.to(self.device)
        hS = hS.to(self.device)
        hmask = hmask.to(self.device)
        if hX_ag is not None:
            hX_ag = hX_ag.to(self.device)
            hS_ag = hS_ag.to(self.device)
            hmask_ag = hmask_ag.to(self.device)
        
        L = hmask[0].sum().long().item()
        
        # Prepare antigen
        X_ag_i, S_ag_i, mask_ag_i = None, None, None
        has_antigen = False
        
        if self.args.use_antigen and hX_ag is not None and hmask_ag is not None:
            L_ag = hmask_ag[0].sum().long().item()
            if L_ag >= 5:  # MIN_AG_LENGTH
                X_ag_i = hX_ag[0:1, :L_ag]
                S_ag_i = hS_ag[0:1, :L_ag]
                mask_ag_i = hmask_ag[0:1, :L_ag]
                has_antigen = True
        
        # Run inference
        with torch.no_grad():
            out = self.model.log_prob(
                hS[0:1, :L],
                [hL[0]],
                hmask[0:1, :L],
                X_ag=X_ag_i,
                S_ag=S_ag_i,
                mask_ag=mask_ag_i
            )
        
        X_pred = out.X_cdr  # Predicted CDR coordinates
        nll = out.nll
        
        # Get CDR boundaries
        # Handle cdr_type as string or int
        cdr_type_int = int(self.args.cdr_type)
        L_list = hL[0]
        
        # Convert L_list elements to int for comparison (they might be strings)
        L_list_int = [int(x) if isinstance(x, (int, str)) else x for x in L_list]
        
        cdr_start = L_list_int.index(cdr_type_int)
        cdr_end = len(L_list_int) - 1 - L_list_int[::-1].index(cdr_type_int)
        cdr_type = cdr_type_int
        cdr_len = cdr_end - cdr_start + 1
        
        # Extract coordinates
        pred_coords = X_pred[0].cpu().numpy()  # (cdr_len, 4, 3)
        true_coords = hX[0, cdr_start:cdr_end+1].cpu().numpy()  # (cdr_len, 4, 3)
        
        # Get full antibody coordinates for context
        full_coords = hX[0, :L].cpu().numpy()
        
        # Extract sequence
        seq_indices = hS[0, :L].cpu().numpy()
        full_seq = ''.join([IDX_TO_AA.get(int(s), 'X') for s in seq_indices])
        cdr_seq = full_seq[cdr_start:cdr_end+1]
        
        # Extract antigen info if available
        antigen_seq = None
        antigen_coords = None
        if has_antigen:
            ag_seq_indices = S_ag_i[0].cpu().numpy()
            antigen_seq = ''.join([IDX_TO_AA.get(int(s), 'X') for s in ag_seq_indices])
            antigen_coords = X_ag_i[0].cpu().numpy()
        
        # Compute RMSD using the same function as training
        rmsd = compute_rmsd_with_structgen(pred_coords, true_coords)
        
        # Also compute aligned coordinates for visualization
        pred_ca = pred_coords[:, 1, :]  # CA is index 1
        true_ca = true_coords[:, 1, :]
        pred_ca_aligned, _ = kabsch_align(pred_ca, true_ca)
        
        # Align full backbone for visualization
        pred_aligned = pred_coords.copy()
        pred_aligned[:, 1, :] = pred_ca_aligned
        
        # Get PDB info if available
        pdb_id = sample.get('pdb', sample.get('pdb_id', f'sample_{sample_idx}'))
        
        result = {
            'sample_idx': sample_idx,
            'pdb_id': pdb_id,
            'cdr_type': cdr_type,
            'cdr_seq': cdr_seq,
            'cdr_len': cdr_len,
            'cdr_start': cdr_start,
            'cdr_end': cdr_end,
            'pred_coords': pred_coords,
            'pred_coords_aligned': pred_aligned,
            'true_coords': true_coords,
            'full_seq': full_seq,
            'full_coords': full_coords,
            'rmsd': rmsd,
            'nll': nll.item(),
            'ppl': math.exp(nll.item()),
            'has_antigen': has_antigen,
            'antigen_seq': antigen_seq,
            'antigen_coords': antigen_coords,
        }
        
        print(f"  PDB: {pdb_id}")
        print(f"  CDR{cdr_type} sequence: {cdr_seq}")
        print(f"  CDR length: {cdr_len}")
        print(f"  RMSD (CA): {rmsd:.3f} Å")
        print(f"  PPL: {result['ppl']:.3f}")
        print(f"  Antigen: {'Yes' if has_antigen else 'No'}")
        
        return result
    
    def predict_multiple(self, loader, n_samples: int = 10, 
                        random_samples: bool = True) -> List[Dict]:
        """Predict multiple samples and return list of results."""
        
        all_data = list(loader.dataset) if hasattr(loader, 'dataset') else loader.data
        n_total = len(all_data)
        
        if random_samples:
            indices = random.sample(range(n_total), min(n_samples, n_total))
        else:
            indices = list(range(min(n_samples, n_total)))
        
        results = []
        for idx in indices:
            result = self.predict_sample(loader, sample_idx=idx)
            results.append(result)
        
        # Summary statistics
        rmsds = [r['rmsd'] for r in results]
        print(f"\n{'='*50}")
        print(f"Summary ({len(results)} samples)")
        print(f"{'='*50}")
        print(f"  Mean RMSD: {np.mean(rmsds):.3f} Å")
        print(f"  Std RMSD:  {np.std(rmsds):.3f} Å")
        print(f"  Min RMSD:  {np.min(rmsds):.3f} Å")
        print(f"  Max RMSD:  {np.max(rmsds):.3f} Å")
        
        return results
    
    def visualize_interactive(self, result: Dict, width: int = 800, 
                              height: int = 600, show_framework: bool = True):
        """
        Create interactive 3D visualization using py3Dmol.
        
        Args:
            result: Prediction result from predict_sample()
            width, height: Viewer dimensions
            show_framework: Whether to show antibody framework
        
        Returns:
            py3Dmol view object (call .show() to display)
        """
        if not PY3DMOL_AVAILABLE:
            print("py3Dmol not available. Install with: pip install py3Dmol")
            return None
        
        view = py3Dmol.view(width=width, height=height)
        
        # Add predicted CDR (green)
        pred_pdb = coords_to_pdb(result['pred_coords_aligned'], result['cdr_seq'], 
                                  'P', result['cdr_start'] + 1)
        view.addModel(pred_pdb, 'pdb')
        view.setStyle({'model': 0}, {
            'cartoon': {'color': 'green'},
            'stick': {'color': 'green', 'radius': 0.15}
        })
        
        # Add ground truth CDR (blue)
        true_pdb = coords_to_pdb(result['true_coords'], result['cdr_seq'], 
                                  'T', result['cdr_start'] + 1)
        view.addModel(true_pdb, 'pdb')
        view.setStyle({'model': 1}, {
            'cartoon': {'color': 'blue'},
            'stick': {'color': 'blue', 'radius': 0.15}
        })
        
        # Add framework (gray, transparent)
        if show_framework and result['full_coords'] is not None:
            # Exclude CDR region from framework
            fw_coords = result['full_coords'].copy()
            fw_seq = result['full_seq']
            
            fw_pdb = coords_to_pdb(fw_coords, fw_seq, 'F', 1)
            view.addModel(fw_pdb, 'pdb')
            view.setStyle({'model': 2}, {
                'cartoon': {'color': 'gray', 'opacity': 0.5}
            })
        
        # Add antigen if available (orange)
        if result['has_antigen'] and result['antigen_coords'] is not None:
            ag_pdb = coords_to_pdb(result['antigen_coords'], result['antigen_seq'], 'A', 1)
            view.addModel(ag_pdb, 'pdb')
            view.setStyle({'model': 3}, {
                'cartoon': {'color': 'orange', 'opacity': 0.7}
            })
        
        view.zoomTo()
        view.setBackgroundColor('white')
        
        print(f"\nVisualization Legend:")
        print(f"  Green: Predicted CDR{result['cdr_type']}")
        print(f"  Blue:  Ground Truth CDR{result['cdr_type']}")
        if show_framework:
            print(f"  Gray:  Antibody Framework")
        if result['has_antigen']:
            print(f"  Orange: Antigen")
        print(f"\nRMSD: {result['rmsd']:.3f} Å")
        
        return view
    
    def save_structures(self, result: Dict, save_dir: str = 'predictions'):
        """Save predicted and true structures as PDB files."""
        os.makedirs(save_dir, exist_ok=True)
        
        pdb_id = result['pdb_id']
        cdr_type = result['cdr_type']
        
        # Save predicted CDR
        pred_path = os.path.join(save_dir, f'{pdb_id}_cdr{cdr_type}_predicted.pdb')
        pred_pdb = coords_to_pdb(result['pred_coords_aligned'], result['cdr_seq'], 'A', 1)
        with open(pred_path, 'w') as f:
            f.write(pred_pdb)
        
        # Save ground truth CDR
        true_path = os.path.join(save_dir, f'{pdb_id}_cdr{cdr_type}_true.pdb')
        true_pdb = coords_to_pdb(result['true_coords'], result['cdr_seq'], 'A', 1)
        with open(true_path, 'w') as f:
            f.write(true_pdb)
        
        # Save result as JSON (without numpy arrays)
        result_json = {k: v for k, v in result.items() 
                       if not isinstance(v, np.ndarray)}
        result_json['rmsd'] = float(result['rmsd'])
        
        json_path = os.path.join(save_dir, f'{pdb_id}_cdr{cdr_type}_result.json')
        with open(json_path, 'w') as f:
            json.dump(result_json, f, indent=2)
        
        print(f"\nSaved structures to {save_dir}/")
        print(f"  {pred_path}")
        print(f"  {true_path}")
        print(f"  {json_path}")
        
        return pred_path, true_path


def main():
    parser = argparse.ArgumentParser(description='CDR Inference and Visualization')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test_path', type=str, required=True,
                        help='Path to test data JSON file')
    parser.add_argument('--cdr_type', type=int, default=None,
                        help='CDR type (1, 2, or 3). If not specified, uses checkpoint args')
    parser.add_argument('--sample_idx', type=int, default=None,
                        help='Specific sample index to predict')
    parser.add_argument('--n_samples', type=int, default=1,
                        help='Number of samples to predict')
    parser.add_argument('--random', action='store_true',
                        help='Select random samples')
    parser.add_argument('--save_dir', type=str, default='predictions',
                        help='Directory to save predictions')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = CDRPredictor(args.checkpoint, device=args.device)
    
    # Override CDR type if specified
    if args.cdr_type is not None:
        predictor.args.cdr_type = str(args.cdr_type)
    
    # Load test data
    loader, data = predictor.load_test_data(args.test_path)
    
    # Run predictions
    if args.n_samples == 1:
        result = predictor.predict_sample(loader, sample_idx=args.sample_idx,
                                          random_sample=args.random or args.sample_idx is None)
        predictor.save_structures(result, args.save_dir)
        
        # Try interactive visualization
        if PY3DMOL_AVAILABLE:
            view = predictor.visualize_interactive(result)
            print("\nTo display in Jupyter/Colab, run: view.show()")
    else:
        results = predictor.predict_multiple(loader, n_samples=args.n_samples,
                                             random_samples=args.random)
        for result in results:
            predictor.save_structures(result, args.save_dir)


if __name__ == '__main__':
    main()