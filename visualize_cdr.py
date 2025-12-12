"""
CDR Structure Visualization: Predicted vs Ground Truth
=======================================================

This script visualizes predicted CDR structures against ground truth
using PyMOL and py3Dmol in Google Colab.

Installation in Colab:
----------------------
!pip install pymol-open-source biopython py3Dmol

Usage:
------
# OPTION 1: Use with inference_cdr.py output (RECOMMENDED)
from inference_cdr import CDRPredictor
from visualize_cdr import visualize_inference_result, visualize_multiple_results

# Run inference
predictor = CDRPredictor('ckpts/cdr3_with_antigen/model.best.ckpt')
loader, data = predictor.load_test_data('data/test.jsonl')
result = predictor.predict_sample(loader, sample_idx=0)

# Visualize
view = visualize_inference_result(result)
view.show()

# Or save as image
visualize_inference_result(result, save_path='cdr_comparison.png')

# OPTION 2: Visualize multiple results
results = predictor.predict_multiple(loader, n_samples=5)
visualize_multiple_results(results, save_dir='figures/')

# OPTION 3: Direct from coordinates
from visualize_cdr import visualize_coords
view = visualize_coords(pred_coords, true_coords, sequence)
view.show()
"""

import os
import sys
import numpy as np
import torch
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import tempfile
import warnings

# Try importing visualization libraries
try:
    from pymol import cmd, finish_launching
    PYMOL_AVAILABLE = True
except ImportError:
    PYMOL_AVAILABLE = False
    print("PyMOL not available. Install with: pip install pymol-open-source")

try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False
    print("py3Dmol not available. Install with: pip install py3Dmol")

try:
    from Bio.PDB import PDBIO, Structure, Model, Chain, Residue, Atom
    from Bio.PDB.Polypeptide import one_to_three
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("BioPython not available. Install with: pip install biopython")


# Amino acid mappings
AA_LIST = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
           'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AA_LIST)}
IDX_TO_AA = {idx: aa for idx, aa in enumerate(AA_LIST)}

# Atom names for backbone
ATOM_NAMES = ['N', 'CA', 'C', 'O']


def coords_to_pdb(coords: np.ndarray, sequence: str, chain_id: str = 'A', 
                  start_res: int = 1) -> str:
    """
    Convert coordinates to PDB format string.
    
    Args:
        coords: Array of shape (L, 4, 3) for backbone atoms [N, CA, C, O]
                or (L, 3) for CA only
        sequence: Amino acid sequence (1-letter codes)
        chain_id: Chain identifier
        start_res: Starting residue number
    
    Returns:
        PDB format string
    """
    pdb_lines = []
    atom_num = 1
    
    # Handle different coordinate formats
    if coords.ndim == 2:
        # CA only - shape (L, 3)
        coords = coords[:, np.newaxis, :]  # (L, 1, 3)
        atom_names = ['CA']
    elif coords.ndim == 3 and coords.shape[1] == 4:
        atom_names = ATOM_NAMES
    elif coords.ndim == 3 and coords.shape[1] == 1:
        atom_names = ['CA']
    else:
        raise ValueError(f"Unexpected coords shape: {coords.shape}")
    
    for res_idx, aa in enumerate(sequence):
        res_num = start_res + res_idx
        
        # Convert 1-letter to 3-letter code
        try:
            res_name = one_to_three(aa) if aa != 'X' else 'UNK'
        except:
            res_name = 'UNK'
        
        for atom_idx, atom_name in enumerate(atom_names):
            if atom_idx >= coords.shape[1]:
                continue
                
            x, y, z = coords[res_idx, atom_idx]
            
            # PDB ATOM record format
            line = (
                f"ATOM  {atom_num:5d}  {atom_name:<3s} {res_name:3s} {chain_id:1s}"
                f"{res_num:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           "
                f"{atom_name[0]:>2s}  "
            )
            pdb_lines.append(line)
            atom_num += 1
    
    pdb_lines.append("END")
    return "\n".join(pdb_lines)


def save_pdb(coords: np.ndarray, sequence: str, filepath: str, 
             chain_id: str = 'A', start_res: int = 1):
    """Save coordinates as PDB file."""
    pdb_str = coords_to_pdb(coords, sequence, chain_id, start_res)
    with open(filepath, 'w') as f:
        f.write(pdb_str)
    return filepath


class CDRVisualizer:
    """
    Visualize CDR predictions vs ground truth structures.
    """
    
    def __init__(self, model=None, args=None, device='cuda'):
        """
        Initialize visualizer.
        
        Args:
            model: Trained CDR generation model
            args: Model arguments
            device: Device to run model on
        """
        self.model = model
        self.args = args
        self.device = device
        
        if model is not None:
            self.model.eval()
    
    def extract_cdr_from_batch(self, batch: Dict, idx: int = 0) -> Dict:
        """
        Extract CDR coordinates and sequence from a batch.
        
        Returns dict with:
            - sequence: CDR sequence
            - coords: CDR coordinates (L, 4, 3)
            - framework_coords: Full antibody coordinates
            - framework_seq: Full antibody sequence
            - cdr_start: CDR start index
            - cdr_end: CDR end index
        """
        # This depends on your data format - adjust as needed
        hX = batch.get('X', batch.get('hX'))
        hS = batch.get('S', batch.get('hS'))
        hL = batch.get('L', batch.get('hL'))
        hmask = batch.get('mask', batch.get('hmask'))
        
        if isinstance(hL, list):
            L_list = hL[idx] if isinstance(hL[idx], list) else hL
        else:
            L_list = hL[idx].tolist() if torch.is_tensor(hL) else hL[idx]
        
        cdr_type = int(self.args.cdr_type) if self.args else 3
        
        # Find CDR boundaries
        try:
            cdr_start = L_list.index(cdr_type)
            cdr_end = len(L_list) - 1 - L_list[::-1].index(cdr_type)
        except ValueError:
            raise ValueError(f"CDR type {cdr_type} not found in L_list")
        
        # Extract coordinates and sequence
        if torch.is_tensor(hX):
            coords = hX[idx].cpu().numpy()
        else:
            coords = np.array(hX[idx])
        
        if torch.is_tensor(hS):
            seq_indices = hS[idx].cpu().numpy()
        else:
            seq_indices = np.array(hS[idx])
        
        # Convert indices to sequence
        sequence = ''.join([IDX_TO_AA.get(int(s), 'X') for s in seq_indices])
        
        # Extract CDR region
        cdr_coords = coords[cdr_start:cdr_end+1]
        cdr_seq = sequence[cdr_start:cdr_end+1]
        
        return {
            'sequence': cdr_seq,
            'coords': cdr_coords,
            'framework_coords': coords,
            'framework_seq': sequence,
            'cdr_start': cdr_start,
            'cdr_end': cdr_end,
            'L_list': L_list
        }
    
    def predict_cdr(self, batch: Dict, idx: int = 0) -> Dict:
        """
        Generate CDR prediction for a sample.
        
        Returns dict with:
            - pred_coords: Predicted CDR coordinates
            - pred_seq: Predicted sequence (if available)
            - true_coords: Ground truth coordinates
            - true_seq: Ground truth sequence
        """
        if self.model is None:
            raise ValueError("Model not provided")
        
        # Import here to avoid circular imports
        try:
            from structgen.data_with_antigen import completize_with_antigen
        except ImportError:
            completize_with_antigen = None
        
        self.model.eval()
        
        with torch.no_grad():
            # Prepare batch
            if completize_with_antigen:
                hX, hS, hL, hmask, hX_ag, hS_ag, hmask_ag = completize_with_antigen([batch[idx]] if isinstance(batch, list) else [batch])
            else:
                hX = batch['X'].to(self.device)
                hS = batch['S'].to(self.device)
                hL = batch['L']
                hmask = batch['mask'].to(self.device)
                hX_ag, hS_ag, hmask_ag = None, None, None
            
            L = hmask[0].sum().long().item()
            
            # Prepare antigen if available
            X_ag_i, S_ag_i, mask_ag_i = None, None, None
            if self.args and self.args.use_antigen and hX_ag is not None:
                L_ag = hmask_ag[0].sum().long().item() if hmask_ag is not None else 0
                if L_ag >= 5:
                    X_ag_i = hX_ag[0:1, :L_ag]
                    S_ag_i = hS_ag[0:1, :L_ag]
                    mask_ag_i = hmask_ag[0:1, :L_ag]
            
            # Get prediction
            out = self.model.log_prob(
                hS[0:1, :L],
                [hL[0]] if isinstance(hL, list) else [hL[0].tolist()],
                hmask[0:1, :L],
                X_ag=X_ag_i,
                S_ag=S_ag_i,
                mask_ag=mask_ag_i
            )
            
            X_pred = out.X_cdr  # Predicted CDR coordinates
            
            # Get CDR boundaries
            L_list = hL[0] if isinstance(hL, list) else hL[0].tolist()
            cdr_type = int(self.args.cdr_type)
            cdr_start = L_list.index(cdr_type)
            cdr_end = len(L_list) - 1 - L_list[::-1].index(cdr_type)
            
            # Extract true coordinates
            true_coords = hX[0, cdr_start:cdr_end+1].cpu().numpy()
            pred_coords = X_pred[0].cpu().numpy()
            
            # Get sequence
            seq_indices = hS[0, cdr_start:cdr_end+1].cpu().numpy()
            cdr_seq = ''.join([IDX_TO_AA.get(int(s), 'X') for s in seq_indices])
            
            return {
                'pred_coords': pred_coords,
                'true_coords': true_coords,
                'sequence': cdr_seq,
                'cdr_start': cdr_start,
                'cdr_end': cdr_end,
                'framework_coords': hX[0].cpu().numpy(),
                'framework_seq': ''.join([IDX_TO_AA.get(int(s), 'X') for s in hS[0].cpu().numpy()])
            }
    
    def compute_rmsd(self, pred_coords: np.ndarray, true_coords: np.ndarray, 
                     use_ca_only: bool = True) -> float:
        """Compute RMSD between predicted and true coordinates."""
        if use_ca_only:
            # Use CA atoms only (index 1)
            if pred_coords.ndim == 3:
                pred = pred_coords[:, 1, :]  # CA
                true = true_coords[:, 1, :]
            else:
                pred = pred_coords
                true = true_coords
        else:
            pred = pred_coords.reshape(-1, 3)
            true = true_coords.reshape(-1, 3)
        
        # Kabsch alignment
        pred_centered = pred - pred.mean(axis=0)
        true_centered = true - true.mean(axis=0)
        
        H = pred_centered.T @ true_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        pred_aligned = pred_centered @ R
        
        rmsd = np.sqrt(np.mean(np.sum((pred_aligned - true_centered) ** 2, axis=1)))
        return rmsd
    
    def visualize_pymol(self, pred_coords: np.ndarray, true_coords: np.ndarray,
                        sequence: str, save_path: str = 'cdr_comparison.png',
                        framework_coords: np.ndarray = None,
                        framework_seq: str = None,
                        cdr_start: int = 0,
                        width: int = 1200, height: int = 800) -> str:
        """
        Create PyMOL visualization comparing predicted vs true CDR.
        
        Args:
            pred_coords: Predicted CDR coordinates
            true_coords: Ground truth CDR coordinates
            sequence: CDR sequence
            save_path: Path to save image
            framework_coords: Optional full antibody coordinates
            framework_seq: Optional full antibody sequence
            cdr_start: CDR starting residue number
            width, height: Image dimensions
        
        Returns:
            Path to saved image
        """
        if not PYMOL_AVAILABLE:
            raise ImportError("PyMOL not available. Install with: pip install pymol-open-source")
        
        # Initialize PyMOL
        finish_launching(['pymol', '-qc'])  # Quiet, no GUI
        cmd.reinitialize()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save structures as PDB
            pred_pdb = os.path.join(tmpdir, 'predicted.pdb')
            true_pdb = os.path.join(tmpdir, 'ground_truth.pdb')
            
            # Handle coordinate dimensions
            if pred_coords.ndim == 2:
                pred_coords = pred_coords[:, np.newaxis, :]
            if true_coords.ndim == 2:
                true_coords = true_coords[:, np.newaxis, :]
            
            save_pdb(pred_coords, sequence, pred_pdb, 'P', cdr_start)
            save_pdb(true_coords, sequence, true_pdb, 'T', cdr_start)
            
            # Load structures
            cmd.load(pred_pdb, 'predicted')
            cmd.load(true_pdb, 'ground_truth')
            
            # Optional: Load framework
            if framework_coords is not None and framework_seq is not None:
                fw_pdb = os.path.join(tmpdir, 'framework.pdb')
                save_pdb(framework_coords, framework_seq, fw_pdb, 'F', 1)
                cmd.load(fw_pdb, 'framework')
                cmd.color('gray80', 'framework')
                cmd.show('cartoon', 'framework')
                cmd.set('cartoon_transparency', 0.7, 'framework')
            
            # Style predicted structure (green)
            cmd.color('green', 'predicted')
            cmd.show('cartoon', 'predicted')
            cmd.show('sticks', 'predicted and name CA')
            cmd.set('cartoon_tube_radius', 0.4, 'predicted')
            
            # Style ground truth structure (blue)
            cmd.color('marine', 'ground_truth')
            cmd.show('cartoon', 'ground_truth')
            cmd.show('sticks', 'ground_truth and name CA')
            cmd.set('cartoon_tube_radius', 0.4, 'ground_truth')
            
            # Align structures
            rmsd = cmd.align('predicted', 'ground_truth')[0]
            
            # Set view
            cmd.zoom('all')
            cmd.center('ground_truth')
            cmd.set('ray_shadows', 0)
            cmd.set('antialias', 2)
            cmd.bg_color('white')
            
            # Add labels
            cmd.set('label_color', 'black')
            cmd.set('label_size', 20)
            
            # Ray trace and save
            cmd.ray(width, height)
            cmd.png(save_path)
            
            print(f"Saved PyMOL visualization to: {save_path}")
            print(f"RMSD (PyMOL align): {rmsd:.3f} Å")
        
        return save_path
    
    def interactive_view(self, pred_coords: np.ndarray, true_coords: np.ndarray,
                        sequence: str, framework_coords: np.ndarray = None,
                        framework_seq: str = None, cdr_start: int = 0,
                        width: int = 800, height: int = 600):
        """
        Create interactive 3D view using py3Dmol (works in Jupyter/Colab).
        
        Returns py3Dmol view object for display in notebook.
        """
        if not PY3DMOL_AVAILABLE:
            raise ImportError("py3Dmol not available. Install with: pip install py3Dmol")
        
        # Handle coordinate dimensions
        if pred_coords.ndim == 2:
            pred_coords = pred_coords[:, np.newaxis, :]
        if true_coords.ndim == 2:
            true_coords = true_coords[:, np.newaxis, :]
        
        # Create PDB strings
        pred_pdb = coords_to_pdb(pred_coords, sequence, 'P', cdr_start)
        true_pdb = coords_to_pdb(true_coords, sequence, 'T', cdr_start)
        
        # Create view
        view = py3Dmol.view(width=width, height=height)
        
        # Add predicted structure (green)
        view.addModel(pred_pdb, 'pdb')
        view.setStyle({'model': 0}, {
            'cartoon': {'color': 'green', 'tube': True, 'radius': 0.3},
            'stick': {'radius': 0.15, 'color': 'green'}
        })
        
        # Add ground truth structure (blue)
        view.addModel(true_pdb, 'pdb')
        view.setStyle({'model': 1}, {
            'cartoon': {'color': 'blue', 'tube': True, 'radius': 0.3},
            'stick': {'radius': 0.15, 'color': 'blue'}
        })
        
        # Add framework if provided
        if framework_coords is not None and framework_seq is not None:
            if framework_coords.ndim == 2:
                framework_coords = framework_coords[:, np.newaxis, :]
            fw_pdb = coords_to_pdb(framework_coords, framework_seq, 'F', 1)
            view.addModel(fw_pdb, 'pdb')
            view.setStyle({'model': 2}, {
                'cartoon': {'color': 'gray', 'opacity': 0.5}
            })
        
        view.zoomTo()
        view.setBackgroundColor('white')
        
        # Compute RMSD for display
        rmsd = self.compute_rmsd(pred_coords, true_coords)
        print(f"CDR Sequence: {sequence}")
        print(f"RMSD (CA): {rmsd:.3f} Å")
        print(f"Legend: Green = Predicted, Blue = Ground Truth")
        
        return view
    
    def visualize_sample(self, loader, sample_idx: int = 0, 
                        save_path: str = 'cdr_comparison.png',
                        use_pymol: bool = True,
                        show_interactive: bool = True) -> Dict:
        """
        Visualize a sample from the data loader.
        
        Args:
            loader: Data loader
            sample_idx: Index of sample to visualize
            save_path: Path to save PyMOL image
            use_pymol: Whether to generate PyMOL image
            show_interactive: Whether to show interactive 3D view
        
        Returns:
            Dict with prediction results and RMSD
        """
        # Get sample
        for i, batch in enumerate(loader):
            if i == sample_idx // loader.batch_size:
                break
        
        batch_idx = sample_idx % loader.batch_size
        
        # Get prediction
        result = self.predict_cdr(batch, batch_idx)
        
        # Compute RMSD
        rmsd = self.compute_rmsd(result['pred_coords'], result['true_coords'])
        result['rmsd'] = rmsd
        
        print(f"\n{'='*50}")
        print(f"Sample {sample_idx}")
        print(f"{'='*50}")
        print(f"CDR Sequence: {result['sequence']}")
        print(f"CDR Length: {len(result['sequence'])}")
        print(f"RMSD (CA): {rmsd:.3f} Å")
        
        # PyMOL visualization
        if use_pymol and PYMOL_AVAILABLE:
            self.visualize_pymol(
                result['pred_coords'],
                result['true_coords'],
                result['sequence'],
                save_path=save_path,
                cdr_start=result['cdr_start']
            )
        
        # Interactive view
        if show_interactive and PY3DMOL_AVAILABLE:
            view = self.interactive_view(
                result['pred_coords'],
                result['true_coords'],
                result['sequence'],
                cdr_start=result['cdr_start']
            )
            result['view'] = view
        
        return result


# ============================================================================
# STANDALONE FUNCTIONS (no model needed)
# ============================================================================

def visualize_structures(pred_coords: np.ndarray, true_coords: np.ndarray,
                        sequence: str, save_path: str = 'comparison.png',
                        method: str = 'both') -> Dict:
    """
    Standalone function to visualize two structures.
    
    Args:
        pred_coords: Predicted coordinates (L, 3) or (L, 4, 3)
        true_coords: Ground truth coordinates (L, 3) or (L, 4, 3)
        sequence: Amino acid sequence
        save_path: Path to save image
        method: 'pymol', 'interactive', or 'both'
    
    Returns:
        Dict with RMSD and optional view object
    """
    viz = CDRVisualizer()
    result = {'sequence': sequence}
    
    # Compute RMSD
    result['rmsd'] = viz.compute_rmsd(pred_coords, true_coords)
    print(f"RMSD (CA): {result['rmsd']:.3f} Å")
    
    if method in ['pymol', 'both'] and PYMOL_AVAILABLE:
        viz.visualize_pymol(pred_coords, true_coords, sequence, save_path)
    
    if method in ['interactive', 'both'] and PY3DMOL_AVAILABLE:
        result['view'] = viz.interactive_view(pred_coords, true_coords, sequence)
    
    return result


def visualize_from_arrays(pred_ca: np.ndarray, true_ca: np.ndarray,
                         sequence: str = None) -> 'py3Dmol.view':
    """
    Quick visualization from CA coordinate arrays.
    
    Args:
        pred_ca: Predicted CA coordinates (L, 3)
        true_ca: Ground truth CA coordinates (L, 3)
        sequence: Optional sequence (will use 'A' * L if not provided)
    
    Returns:
        py3Dmol view object
    """
    if sequence is None:
        sequence = 'A' * len(pred_ca)
    
    viz = CDRVisualizer()
    return viz.interactive_view(pred_ca, true_ca, sequence)


# ============================================================================
# COLAB SETUP HELPER
# ============================================================================

def setup_colab():
    """
    Setup visualization libraries in Google Colab.
    Run this cell first in your Colab notebook.
    """
    setup_code = """
# Run this in a Colab cell to install required packages:

!pip install -q pymol-open-source biopython py3Dmol

# For PyMOL to work in Colab, you may also need:
!apt-get install -qq libgl1-mesa-glx

# Verify installation:
import pymol
import py3Dmol
from Bio.PDB import PDBParser
print("✓ All visualization libraries installed successfully!")
"""
    print(setup_code)


# ============================================================================
# FUNCTIONS TO USE WITH INFERENCE_CDR.PY OUTPUT
# ============================================================================

def visualize_inference_result(result: dict, 
                                save_path: Optional[str] = None,
                                method: str = 'interactive',
                                show_framework: bool = False,
                                show_antigen: bool = True,
                                width: int = 800,
                                height: int = 600):
    """
    Visualize output from inference_cdr.py CDRPredictor.
    
    Args:
        result: Dict output from CDRPredictor.predict_sample()
                Must contain: pred_coords, true_coords, cdr_seq, rmsd
        save_path: Path to save image (for PyMOL)
        method: 'interactive' (py3Dmol), 'pymol', or 'both'
        show_framework: Show antibody framework
        show_antigen: Show antigen if available
        width, height: Viewer dimensions
    
    Returns:
        py3Dmol view object if method includes 'interactive'
    
    Example:
        from inference_cdr import CDRPredictor
        from visualize_cdr import visualize_inference_result
        
        predictor = CDRPredictor('checkpoint.pt')
        loader, data = predictor.load_test_data('test.jsonl')
        result = predictor.predict_sample(loader, sample_idx=0)
        
        view = visualize_inference_result(result)
        view.show()
    """
    # Extract data from inference result
    pred_coords = result.get('pred_coords_aligned', result.get('pred_coords'))
    true_coords = result['true_coords']
    sequence = result['cdr_seq']
    cdr_start = result.get('cdr_start', 1)
    rmsd = result.get('rmsd', 0)
    pdb_id = result.get('pdb_id', 'unknown')
    cdr_type = result.get('cdr_type', 3)
    
    # Optional data
    framework_coords = result.get('full_coords') if show_framework else None
    framework_seq = result.get('full_seq') if show_framework else None
    antigen_coords = result.get('antigen_coords') if show_antigen else None
    antigen_seq = result.get('antigen_seq') if show_antigen else None
    
    print(f"\n{'='*50}")
    print(f"Visualizing: {pdb_id} CDR{cdr_type}")
    print(f"{'='*50}")
    print(f"Sequence: {sequence}")
    print(f"Length: {len(sequence)}")
    print(f"RMSD: {rmsd:.3f} Å")
    if result.get('has_antigen'):
        print(f"Antigen: Yes ({len(antigen_seq)} residues)")
    
    view = None
    
    # Interactive visualization with py3Dmol
    if method in ['interactive', 'both']:
        if not PY3DMOL_AVAILABLE:
            print("py3Dmol not available. Install with: pip install py3Dmol")
        else:
            view = _create_py3dmol_view(
                pred_coords, true_coords, sequence,
                cdr_start=cdr_start,
                framework_coords=framework_coords,
                framework_seq=framework_seq,
                antigen_coords=antigen_coords,
                antigen_seq=antigen_seq,
                width=width, height=height
            )
            print(f"\nLegend: Green=Predicted, Blue=Ground Truth", end="")
            if show_framework:
                print(", Gray=Framework", end="")
            if antigen_coords is not None:
                print(", Orange=Antigen", end="")
            print()
    
    # PyMOL visualization
    if method in ['pymol', 'both'] and save_path:
        if not PYMOL_AVAILABLE:
            print("PyMOL not available. Install with: pip install pymol-open-source")
        else:
            viz = CDRVisualizer()
            viz.visualize_pymol(
                pred_coords, true_coords, sequence,
                save_path=save_path,
                framework_coords=framework_coords,
                framework_seq=framework_seq,
                cdr_start=cdr_start,
                width=width, height=height
            )
    
    return view


def _create_py3dmol_view(pred_coords, true_coords, sequence,
                          cdr_start=1, framework_coords=None, framework_seq=None,
                          antigen_coords=None, antigen_seq=None,
                          width=800, height=600):
    """Internal function to create py3Dmol view."""
    
    view = py3Dmol.view(width=width, height=height)
    model_idx = 0
    
    # Handle coordinate dimensions
    if pred_coords.ndim == 2:
        pred_coords = pred_coords[:, np.newaxis, :]
    if true_coords.ndim == 2:
        true_coords = true_coords[:, np.newaxis, :]
    
    # Add predicted CDR (green)
    pred_pdb = coords_to_pdb(pred_coords, sequence, 'P', cdr_start)
    view.addModel(pred_pdb, 'pdb')
    view.setStyle({'model': model_idx}, {
        'cartoon': {'color': 'green'},
        'stick': {'color': 'green', 'radius': 0.15}
    })
    model_idx += 1
    
    # Add ground truth CDR (blue)
    true_pdb = coords_to_pdb(true_coords, sequence, 'T', cdr_start)
    view.addModel(true_pdb, 'pdb')
    view.setStyle({'model': model_idx}, {
        'cartoon': {'color': 'blue'},
        'stick': {'color': 'blue', 'radius': 0.15}
    })
    model_idx += 1
    
    # Add framework (gray)
    if framework_coords is not None and framework_seq is not None:
        if framework_coords.ndim == 2:
            framework_coords = framework_coords[:, np.newaxis, :]
        fw_pdb = coords_to_pdb(framework_coords, framework_seq, 'F', 1)
        view.addModel(fw_pdb, 'pdb')
        view.setStyle({'model': model_idx}, {
            'cartoon': {'color': 'gray', 'opacity': 0.5}
        })
        model_idx += 1
    
    # Add antigen (orange)
    if antigen_coords is not None and antigen_seq is not None:
        if antigen_coords.ndim == 2:
            antigen_coords = antigen_coords[:, np.newaxis, :]
        ag_pdb = coords_to_pdb(antigen_coords, antigen_seq, 'A', 1)
        view.addModel(ag_pdb, 'pdb')
        view.setStyle({'model': model_idx}, {
            'cartoon': {'color': 'orange', 'opacity': 0.7}
        })
        model_idx += 1
    
    view.zoomTo()
    view.setBackgroundColor('white')
    
    return view


def visualize_multiple_results(results: list, 
                                save_dir: str = 'figures',
                                method: str = 'pymol',
                                show_framework: bool = False,
                                width: int = 1200,
                                height: int = 800):
    """
    Visualize multiple inference results and save images.
    
    Args:
        results: List of dicts from CDRPredictor.predict_multiple()
        save_dir: Directory to save images
        method: 'pymol' to save images, 'interactive' for views
        show_framework: Show antibody framework
        width, height: Image dimensions
    
    Example:
        results = predictor.predict_multiple(loader, n_samples=10)
        visualize_multiple_results(results, save_dir='figures/')
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    views = []
    
    for i, result in enumerate(results):
        pdb_id = result.get('pdb_id', f'sample_{i}')
        cdr_type = result.get('cdr_type', 3)
        rmsd = result.get('rmsd', 0)
        
        save_path = os.path.join(save_dir, f'{pdb_id}_cdr{cdr_type}_rmsd{rmsd:.2f}.png')
        
        view = visualize_inference_result(
            result,
            save_path=save_path if method in ['pymol', 'both'] else None,
            method=method,
            show_framework=show_framework,
            width=width,
            height=height
        )
        
        if view is not None:
            views.append((pdb_id, rmsd, view))
    
    # Summary
    rmsds = [r['rmsd'] for r in results]
    print(f"\n{'='*50}")
    print(f"Visualization Summary")
    print(f"{'='*50}")
    print(f"Samples visualized: {len(results)}")
    print(f"Mean RMSD: {np.mean(rmsds):.3f} Å")
    print(f"Best RMSD: {np.min(rmsds):.3f} Å")
    print(f"Worst RMSD: {np.max(rmsds):.3f} Å")
    if method in ['pymol', 'both']:
        print(f"Images saved to: {save_dir}/")
    
    return views


def visualize_coords(pred_coords: np.ndarray, 
                     true_coords: np.ndarray,
                     sequence: str,
                     save_path: Optional[str] = None,
                     method: str = 'interactive',
                     width: int = 800,
                     height: int = 600):
    """
    Simple visualization from raw coordinate arrays.
    
    Args:
        pred_coords: Predicted coordinates (L, 3) or (L, 4, 3)
        true_coords: Ground truth coordinates (L, 3) or (L, 4, 3)
        sequence: CDR sequence
        save_path: Path to save image
        method: 'interactive', 'pymol', or 'both'
    
    Returns:
        py3Dmol view if interactive
    
    Example:
        view = visualize_coords(pred_ca, true_ca, "ARGSYWFDY")
        view.show()
    """
    # Create pseudo result dict
    result = {
        'pred_coords': pred_coords,
        'pred_coords_aligned': pred_coords,
        'true_coords': true_coords,
        'cdr_seq': sequence,
        'cdr_start': 1,
        'cdr_type': 3,
        'pdb_id': 'custom',
        'rmsd': 0,  # Will be computed if needed
        'has_antigen': False,
    }
    
    # Compute RMSD
    if pred_coords.ndim == 3:
        pred_ca = pred_coords[:, 1, :] if pred_coords.shape[1] >= 2 else pred_coords[:, 0, :]
        true_ca = true_coords[:, 1, :] if true_coords.shape[1] >= 2 else true_coords[:, 0, :]
    else:
        pred_ca = pred_coords
        true_ca = true_coords
    
    # Simple RMSD (assumes aligned)
    result['rmsd'] = np.sqrt(np.mean(np.sum((pred_ca - true_ca) ** 2, axis=1)))
    
    return visualize_inference_result(result, save_path=save_path, method=method,
                                       width=width, height=height)


def create_comparison_figure(results: list, 
                              save_path: str = 'cdr_comparison_grid.png',
                              cols: int = 3,
                              figsize: tuple = (15, 10)):
    """
    Create a matplotlib figure with multiple CDR comparisons.
    
    Args:
        results: List of inference results
        save_path: Path to save figure
        cols: Number of columns in grid
        figsize: Figure size
    
    Example:
        results = predictor.predict_multiple(loader, n_samples=6)
        create_comparison_figure(results, 'comparison.png')
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available")
        return None
    
    n = len(results)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize, 
                              subplot_kw={'projection': '3d'})
    
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]
    
    for i, result in enumerate(results):
        row = i // cols
        col = i % cols
        ax = axes[row][col] if isinstance(axes[row], list) else axes[row]
        
        # Get CA coordinates
        pred = result.get('pred_coords_aligned', result['pred_coords'])
        true = result['true_coords']
        
        if pred.ndim == 3:
            pred_ca = pred[:, 1, :]
            true_ca = true[:, 1, :]
        else:
            pred_ca = pred
            true_ca = true
        
        # Plot
        ax.plot(pred_ca[:, 0], pred_ca[:, 1], pred_ca[:, 2], 
                'g-o', label='Predicted', linewidth=2, markersize=4)
        ax.plot(true_ca[:, 0], true_ca[:, 1], true_ca[:, 2], 
                'b-s', label='Ground Truth', linewidth=2, markersize=4)
        
        pdb_id = result.get('pdb_id', f'Sample {i}')
        rmsd = result.get('rmsd', 0)
        ax.set_title(f'{pdb_id}\nRMSD: {rmsd:.2f} Å', fontsize=10)
        ax.legend(fontsize=8)
    
    # Hide empty subplots
    for i in range(n, rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row][col] if isinstance(axes[row], list) else axes[row]
        ax.set_visible(False)
    
    plt.suptitle('CDR Prediction Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison figure to: {save_path}")
    plt.show()
    
    return fig


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    print(__doc__)
    print("\n" + "="*60)
    print("SETUP INSTRUCTIONS FOR COLAB")
    print("="*60)
    setup_colab()
    
    print("\n" + "="*60)
    print("EXAMPLE USAGE WITH INFERENCE_CDR.PY")
    print("="*60)
    print("""
# 1. Run inference and visualize

from inference_cdr import CDRPredictor
from visualize_cdr import visualize_inference_result, visualize_multiple_results

# Load model and data
predictor = CDRPredictor('ckpts/cdr3_with_antigen/model.best.ckpt')
loader, data = predictor.load_test_data('data/test.jsonl')

# Predict a sample
result = predictor.predict_sample(loader, sample_idx=0)

# Interactive visualization (in notebook)
view = visualize_inference_result(result)
view.show()

# Save as PyMOL image
visualize_inference_result(result, save_path='cdr3_comparison.png', method='pymol')

# 2. Visualize multiple samples

results = predictor.predict_multiple(loader, n_samples=10)
visualize_multiple_results(results, save_dir='figures/', method='pymol')

# 3. Create comparison grid figure
from visualize_cdr import create_comparison_figure
create_comparison_figure(results, save_path='comparison_grid.png')

# 4. Simple visualization from arrays
from visualize_cdr import visualize_coords
view = visualize_coords(pred_coords, true_coords, sequence="ARGSYWFDY")
view.show()
""")