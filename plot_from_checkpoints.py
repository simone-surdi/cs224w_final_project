"""
Training Curves Visualization for Antigen-Aware CDR Generation
==============================================================

This script loads checkpoint files created by ab_train_with_antigen.py
and generates comprehensive comparison plots.

Usage in Colab:
---------------
# Option 1: Run as script
!python plot_from_checkpoints.py --ckpt_dirs ckpts/cdr1_ag ckpts/cdr1_no_ag ckpts/cdr2_ag ...

# Option 2: Import and use functions
from plot_from_checkpoints import load_all_experiments, plot_all_comparisons
experiments = load_all_experiments({
    'CDR1 + Antigen': 'ckpts/cdr1_with_antigen',
    'CDR1 No Antigen': 'ckpts/cdr1_no_antigen',
    ...
})
plot_all_comparisons(experiments, save_dir='plots/')

# Option 3: Quick plot from single checkpoint
from plot_from_checkpoints import quick_plot
quick_plot('ckpts/cdr1_with_antigen')
"""

import torch
import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Union
import argparse


# ============================================================================
# CHECKPOINT LOADING FUNCTIONS
# ============================================================================

def load_history_from_checkpoint(ckpt_path: str) -> Optional[Dict]:
    """
    Load training history from a checkpoint file.
    
    Supports both old format (tuple) and new format (dict).
    """
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        # New format: dict with 'history' key
        if isinstance(ckpt, dict):
            if 'history' in ckpt:
                history = ckpt['history'].copy()
                # Add single-epoch metrics if not in history
                if 'test_ppl' not in history and 'test_ppl' in ckpt:
                    history['test_ppl'] = ckpt['test_ppl']
                if 'test_rmsd' not in history and 'test_rmsd' in ckpt:
                    history['test_rmsd'] = ckpt['test_rmsd']
                return history
            else:
                # Might be just model state dict
                print(f"Warning: No 'history' key in {ckpt_path}")
                return None
        
        # Old format: tuple (model_state, optimizer_state, args, [history])
        elif isinstance(ckpt, tuple):
            if len(ckpt) > 3 and isinstance(ckpt[3], dict):
                return ckpt[3]
            else:
                print(f"Warning: Old checkpoint format without history in {ckpt_path}")
                return None
        
        else:
            print(f"Warning: Unknown checkpoint format in {ckpt_path}")
            return None
            
    except Exception as e:
        print(f"Error loading {ckpt_path}: {e}")
        return None


def load_history_from_json(json_path: str) -> Optional[Dict]:
    """Load training history from a JSON file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None


def find_best_checkpoint(ckpt_dir: str) -> Optional[str]:
    """Find the best or latest checkpoint in a directory."""
    ckpt_dir = Path(ckpt_dir)
    
    # Priority order for checkpoint files
    candidates = [
        ckpt_dir / "model.best_rmsd.ckpt",
        ckpt_dir / "model.best.ckpt",
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    
    # Find latest epoch checkpoint
    ckpt_files = list(ckpt_dir.glob("model.ckpt.*"))
    if ckpt_files:
        def get_epoch(f):
            try:
                return int(str(f).split('.')[-1])
            except:
                return -1
        ckpt_files.sort(key=get_epoch, reverse=True)
        return str(ckpt_files[0])
    
    return None


def load_experiment(ckpt_dir: str, name: Optional[str] = None) -> Optional[Dict]:
    """
    Load experiment history from a checkpoint directory.
    
    Tries in order:
    1. history.json
    2. results.json
    3. Best checkpoint
    4. Latest checkpoint
    """
    ckpt_dir = Path(ckpt_dir)
    
    if not ckpt_dir.exists():
        print(f"Directory not found: {ckpt_dir}")
        return None
    
    # Try history.json first
    history_json = ckpt_dir / "history.json"
    if history_json.exists():
        history = load_history_from_json(str(history_json))
        if history:
            print(f"Loaded history.json from {ckpt_dir}")
            return history
    
    # Try results.json
    results_json = ckpt_dir / "results.json"
    if results_json.exists():
        results = load_history_from_json(str(results_json))
        if results and 'history' in results:
            print(f"Loaded results.json from {ckpt_dir}")
            history = results['history']
            history['test_ppl'] = results.get('test', {}).get('ppl')
            history['test_rmsd'] = results.get('test', {}).get('rmsd')
            return history
    
    # Try checkpoint files
    ckpt_path = find_best_checkpoint(str(ckpt_dir))
    if ckpt_path:
        history = load_history_from_checkpoint(ckpt_path)
        if history:
            print(f"Loaded checkpoint from {ckpt_path}")
            return history
    
    print(f"No valid history found in {ckpt_dir}")
    return None


def load_all_experiments(experiment_dirs: Dict[str, str]) -> Dict[str, Dict]:
    """
    Load multiple experiments from checkpoint directories.
    
    Args:
        experiment_dirs: Dict mapping experiment names to checkpoint directories
                        e.g., {'CDR1 + Antigen': 'ckpts/cdr1_ag', ...}
    
    Returns:
        Dict mapping experiment names to history dicts
    """
    experiments = {}
    
    for name, ckpt_dir in experiment_dirs.items():
        history = load_experiment(ckpt_dir, name)
        if history:
            experiments[name] = history
    
    print(f"\nLoaded {len(experiments)}/{len(experiment_dirs)} experiments")
    return experiments


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_single_experiment(history: Dict, title: str, save_path: Optional[str] = None,
                          show: bool = True) -> plt.Figure:
    """
    Plot training curves for a single experiment.
    
    Creates 3 subplots: PPL, RMSD, and normalized metrics.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = history.get('epoch', list(range(len(history.get('val_ppl', [])))))
    
    # Plot 1: PPL (Train vs Val)
    if 'train_ppl' in history and history['train_ppl']:
        axes[0].plot(epochs[:len(history['train_ppl'])], history['train_ppl'], 
                     'b-o', label='Train PPL', linewidth=2, markersize=5)
    if 'val_ppl' in history and history['val_ppl']:
        axes[0].plot(epochs[:len(history['val_ppl'])], history['val_ppl'], 
                     'r-s', label='Val PPL', linewidth=2, markersize=5)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Perplexity (PPL)', fontsize=12)
    axes[0].set_title('Perplexity', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: RMSD
    if 'val_rmsd' in history and history['val_rmsd']:
        axes[1].plot(epochs[:len(history['val_rmsd'])], history['val_rmsd'], 
                     'g-^', linewidth=2, markersize=5, label='Val RMSD')
        min_rmsd = min(history['val_rmsd'])
        axes[1].axhline(y=min_rmsd, color='g', linestyle=':', alpha=0.5, 
                       label=f'Best: {min_rmsd:.3f} Å')
        axes[1].axhline(y=1.0, color='r', linestyle='--', alpha=0.3, label='1.0 Å')
        axes[1].axhline(y=2.0, color='orange', linestyle='--', alpha=0.3, label='2.0 Å')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('RMSD (Å)', fontsize=12)
    axes[1].set_title('Validation RMSD', fontsize=14)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Loss (if available) or normalized metrics
    if 'train_loss' in history and history['train_loss']:
        axes[2].plot(epochs[:len(history['train_loss'])], history['train_loss'], 
                     'b-o', label='Train Loss', linewidth=2, markersize=5)
        if 'val_loss' in history and history['val_loss']:
            axes[2].plot(epochs[:len(history['val_loss'])], history['val_loss'], 
                         'r-s', label='Val Loss', linewidth=2, markersize=5)
        axes[2].set_ylabel('Loss', fontsize=12)
        axes[2].set_title('Loss', fontsize=14)
    else:
        # Plot normalized metrics
        def normalize(arr):
            arr = np.array(arr)
            if arr.max() - arr.min() > 0:
                return (arr - arr.min()) / (arr.max() - arr.min())
            return arr * 0
        
        if history.get('train_ppl'):
            axes[2].plot(epochs[:len(history['train_ppl'])], normalize(history['train_ppl']), 
                         'b-o', label='Train PPL (norm)', linewidth=2, markersize=5)
        if history.get('val_ppl'):
            axes[2].plot(epochs[:len(history['val_ppl'])], normalize(history['val_ppl']), 
                         'r-s', label='Val PPL (norm)', linewidth=2, markersize=5)
        if history.get('val_rmsd'):
            axes[2].plot(epochs[:len(history['val_rmsd'])], normalize(history['val_rmsd']), 
                         'g-^', label='Val RMSD (norm)', linewidth=2, markersize=5)
        axes[2].set_ylabel('Normalized Value', fontsize=12)
        axes[2].set_title('All Metrics (Normalized)', fontsize=14)
    
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_antigen_comparison(history_with_ag: Dict, history_no_ag: Dict, 
                           cdr_name: str, save_path: Optional[str] = None,
                           show: bool = True) -> plt.Figure:
    """
    Plot side-by-side comparison: with antigen vs without antigen for one CDR.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    configs = [
        ('With Antigen', history_with_ag, 'tab:green', 'o'),
        ('Without Antigen', history_no_ag, 'tab:red', 's'),
    ]
    
    for label, history, color, marker in configs:
        if history is None:
            continue
            
        epochs = history.get('epoch', list(range(len(history.get('val_ppl', [])))))
        
        # Plot Val PPL
        if 'val_ppl' in history and history['val_ppl']:
            axes[0].plot(epochs[:len(history['val_ppl'])], history['val_ppl'], 
                        f'-{marker}', color=color, label=label, linewidth=2, markersize=5)
        
        # Plot Val RMSD
        if 'val_rmsd' in history and history['val_rmsd']:
            axes[1].plot(epochs[:len(history['val_rmsd'])], history['val_rmsd'], 
                        f'-{marker}', color=color, label=label, linewidth=2, markersize=5)
        
        # Plot Train PPL
        if 'train_ppl' in history and history['train_ppl']:
            axes[2].plot(epochs[:len(history['train_ppl'])], history['train_ppl'], 
                        f'-{marker}', color=color, label=label, linewidth=2, markersize=5)
    
    # RMSD reference lines
    axes[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='1.0 Å')
    axes[1].axhline(y=2.0, color='gray', linestyle=':', alpha=0.5, label='2.0 Å')
    
    titles = ['Validation PPL', 'Validation RMSD', 'Training PPL']
    ylabels = ['PPL', 'RMSD (Å)', 'PPL']
    
    for i in range(3):
        axes[i].set_xlabel('Epoch', fontsize=12)
        axes[i].set_ylabel(ylabels[i], fontsize=12)
        axes[i].set_title(titles[i], fontsize=14)
        axes[i].legend(fontsize=10)
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle(f'{cdr_name}: With vs Without Antigen', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_cdr_comparison(cdr_histories: Dict[str, Dict[str, Dict]], 
                        save_path: Optional[str] = None,
                        show: bool = True) -> plt.Figure:
    """
    Plot comparison across CDR1, CDR2, CDR3 with/without antigen.
    
    Args:
        cdr_histories: Dict with structure:
            {
                'CDR1': {'with_ag': history, 'no_ag': history},
                'CDR2': {'with_ag': history, 'no_ag': history},
                'CDR3': {'with_ag': history, 'no_ag': history},
            }
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    cdr_colors = {'CDR1': 'tab:blue', 'CDR2': 'tab:orange', 'CDR3': 'tab:green'}
    markers = {'with_ag': 'o', 'no_ag': 's'}
    
    for cdr_name, histories in cdr_histories.items():
        color = cdr_colors.get(cdr_name, 'gray')
        
        for ag_type, history in histories.items():
            if history is None:
                continue
                
            epochs = history.get('epoch', list(range(len(history.get('val_ppl', [])))))
            row = 0 if ag_type == 'with_ag' else 1
            marker = markers[ag_type]
            
            # Plot Val PPL
            if 'val_ppl' in history and history['val_ppl']:
                axes[row, 0].plot(epochs[:len(history['val_ppl'])], history['val_ppl'], 
                                 f'-{marker}', color=color, label=cdr_name, 
                                 linewidth=2, markersize=5)
            
            # Plot Val RMSD
            if 'val_rmsd' in history and history['val_rmsd']:
                axes[row, 1].plot(epochs[:len(history['val_rmsd'])], history['val_rmsd'], 
                                 f'-{marker}', color=color, label=cdr_name, 
                                 linewidth=2, markersize=5)
            
            # Plot Train PPL
            if 'train_ppl' in history and history['train_ppl']:
                axes[row, 2].plot(epochs[:len(history['train_ppl'])], history['train_ppl'], 
                                 f'-{marker}', color=color, label=cdr_name, 
                                 linewidth=2, markersize=5)
    
    # Configure axes
    row_titles = ['With Antigen', 'Without Antigen']
    col_titles = ['Validation PPL', 'Validation RMSD (Å)', 'Training PPL']
    
    for row in range(2):
        for col in range(3):
            axes[row, col].set_xlabel('Epoch', fontsize=11)
            axes[row, col].set_ylabel(col_titles[col].replace(' (Å)', ''), fontsize=11)
            axes[row, col].set_title(f'{row_titles[row]} - {col_titles[col]}', fontsize=12)
            axes[row, col].legend(fontsize=9, loc='best')
            axes[row, col].grid(True, alpha=0.3)
            
            if col == 1:  # RMSD plot
                axes[row, col].axhline(y=1.0, color='r', linestyle=':', alpha=0.4)
                axes[row, col].axhline(y=2.0, color='orange', linestyle=':', alpha=0.4)
    
    plt.suptitle('CDR Generation: Impact of Antigen Conditioning', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_rmsd_comparison_bar(experiments: Dict[str, Dict], 
                             save_path: Optional[str] = None,
                             show: bool = True) -> plt.Figure:
    """
    Create bar chart comparing best RMSD across experiments.
    """
    names = []
    best_rmsds = []
    colors = []
    
    color_map = {
        'CDR1': 'tab:blue',
        'CDR2': 'tab:orange', 
        'CDR3': 'tab:green'
    }
    
    for name, history in experiments.items():
        if history and 'val_rmsd' in history and history['val_rmsd']:
            names.append(name)
            best_rmsds.append(min(history['val_rmsd']))
            
            # Determine color based on CDR type
            color = 'gray'
            for cdr, c in color_map.items():
                if cdr in name.upper():
                    color = c
                    break
            colors.append(color)
    
    if not names:
        print("No RMSD data to plot")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(names))
    bars = ax.bar(x, best_rmsds, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, rmsd in zip(bars, best_rmsds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{rmsd:.2f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Best Validation RMSD (Å)', fontsize=12)
    ax.set_title('Best RMSD Comparison Across Experiments', fontsize=14, fontweight='bold')
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='1.0 Å threshold')
    ax.axhline(y=2.0, color='orange', linestyle='--', alpha=0.5, label='2.0 Å threshold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def create_summary_table(experiments: Dict[str, Dict], 
                        save_path: Optional[str] = None) -> str:
    """
    Create and print a summary table of best metrics.
    Returns the table as a string.
    """
    lines = []
    lines.append("=" * 100)
    lines.append("TRAINING SUMMARY")
    lines.append("=" * 100)
    header = f"{'Experiment':<35} {'Best Val PPL':<15} {'Best Val RMSD':<15} {'Test PPL':<12} {'Test RMSD':<12}"
    lines.append(header)
    lines.append("-" * 100)
    
    results = []
    for name, history in sorted(experiments.items()):
        if history is None:
            continue
        
        best_ppl = min(history.get('val_ppl', [float('inf')])) if history.get('val_ppl') else float('inf')
        best_rmsd = min(history.get('val_rmsd', [float('inf')])) if history.get('val_rmsd') else float('inf')
        test_ppl = history.get('test_ppl')
        test_rmsd = history.get('test_rmsd')
        
        test_ppl_str = f"{test_ppl:.3f}" if isinstance(test_ppl, (int, float)) else "N/A"
        test_rmsd_str = f"{test_rmsd:.3f}" if isinstance(test_rmsd, (int, float)) else "N/A"
        
        line = f"{name:<35} {best_ppl:<15.3f} {best_rmsd:<15.3f} {test_ppl_str:<12} {test_rmsd_str:<12}"
        lines.append(line)
        
        results.append({
            'name': name,
            'best_val_ppl': best_ppl,
            'best_val_rmsd': best_rmsd,
            'test_ppl': test_ppl,
            'test_rmsd': test_rmsd
        })
    
    lines.append("=" * 100)
    
    # Add improvement analysis
    lines.append("\nIMPROVEMENT FROM ANTIGEN CONDITIONING:")
    lines.append("-" * 50)
    
    for cdr in ['CDR1', 'CDR2', 'CDR3']:
        with_ag = None
        no_ag = None
        
        for name, history in experiments.items():
            if cdr in name.upper():
                if 'NO' in name.upper() or 'WITHOUT' in name.upper():
                    if history and history.get('val_rmsd'):
                        no_ag = min(history['val_rmsd'])
                else:
                    if history and history.get('val_rmsd'):
                        with_ag = min(history['val_rmsd'])
        
        if with_ag is not None and no_ag is not None:
            improvement = no_ag - with_ag
            pct_improvement = (improvement / no_ag) * 100
            lines.append(f"{cdr}: {no_ag:.3f} Å → {with_ag:.3f} Å (Δ = {improvement:.3f} Å, {pct_improvement:.1f}% improvement)")
        elif with_ag is not None:
            lines.append(f"{cdr}: With antigen = {with_ag:.3f} Å (no baseline)")
        elif no_ag is not None:
            lines.append(f"{cdr}: Without antigen = {no_ag:.3f} Å (no antigen model)")
    
    table_str = "\n".join(lines)
    print(table_str)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(table_str)
        print(f"\nSaved summary to: {save_path}")
    
    return table_str


# ============================================================================
# HIGH-LEVEL FUNCTIONS
# ============================================================================

def quick_plot(ckpt_dir: str, save_dir: Optional[str] = None, show: bool = True):
    """
    Quick plot from a single checkpoint directory.
    
    Usage:
        quick_plot('ckpts/cdr1_with_antigen')
    """
    history = load_experiment(ckpt_dir)
    if history:
        name = os.path.basename(ckpt_dir)
        save_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{name}_curves.png")
        plot_single_experiment(history, name, save_path, show)
        return history
    return None


def plot_all_comparisons(experiments: Dict[str, Dict], 
                        save_dir: str = 'plots',
                        show: bool = True):
    """
    Generate all comparison plots from loaded experiments.
    
    Args:
        experiments: Dict mapping experiment names to history dicts
        save_dir: Directory to save plots
        show: Whether to display plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Plot individual experiments
    print("\n" + "="*60)
    print("GENERATING INDIVIDUAL EXPERIMENT PLOTS")
    print("="*60)
    for name, history in experiments.items():
        if history:
            safe_name = name.replace(' ', '_').replace('+', 'with').replace('/', '_')
            save_path = os.path.join(save_dir, f"{safe_name}_curves.png")
            plot_single_experiment(history, name, save_path, show)
    
    # 2. Plot antigen comparisons for each CDR
    print("\n" + "="*60)
    print("GENERATING ANTIGEN COMPARISON PLOTS")
    print("="*60)
    for cdr in ['CDR1', 'CDR2', 'CDR3']:
        with_ag = None
        no_ag = None
        
        for name, history in experiments.items():
            name_upper = name.upper()
            if cdr in name_upper:
                if 'NO' in name_upper or 'WITHOUT' in name_upper:
                    no_ag = history
                elif 'ANTIGEN' in name_upper or 'AG' in name_upper:
                    with_ag = history
        
        if with_ag or no_ag:
            save_path = os.path.join(save_dir, f"{cdr}_antigen_comparison.png")
            plot_antigen_comparison(with_ag, no_ag, cdr, save_path, show)
    
    # 3. Plot CDR comparison grid
    print("\n" + "="*60)
    print("GENERATING CDR COMPARISON GRID")
    print("="*60)
    cdr_histories = {}
    for cdr in ['CDR1', 'CDR2', 'CDR3']:
        cdr_histories[cdr] = {'with_ag': None, 'no_ag': None}
        
        for name, history in experiments.items():
            name_upper = name.upper()
            if cdr in name_upper:
                if 'NO' in name_upper or 'WITHOUT' in name_upper:
                    cdr_histories[cdr]['no_ag'] = history
                elif 'ANTIGEN' in name_upper or 'AG' in name_upper:
                    cdr_histories[cdr]['with_ag'] = history
    
    has_any = any(h['with_ag'] or h['no_ag'] for h in cdr_histories.values())
    if has_any:
        save_path = os.path.join(save_dir, "CDR_comparison_all.png")
        plot_cdr_comparison(cdr_histories, save_path, show)
    
    # 4. Plot RMSD bar chart
    print("\n" + "="*60)
    print("GENERATING RMSD BAR CHART")
    print("="*60)
    save_path = os.path.join(save_dir, "RMSD_comparison_bar.png")
    plot_rmsd_comparison_bar(experiments, save_path, show)
    
    # 5. Create summary table
    print("\n" + "="*60)
    print("GENERATING SUMMARY TABLE")
    print("="*60)
    save_path = os.path.join(save_dir, "summary.txt")
    create_summary_table(experiments, save_path)
    
    print(f"\nAll plots saved to: {save_dir}/")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Plot training curves from checkpoint files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot from checkpoint directories
  python plot_from_checkpoints.py --ckpt_dirs ckpts/cdr1_ag ckpts/cdr1_no_ag
  
  # Specify custom names
  python plot_from_checkpoints.py --ckpt_dirs ckpts/cdr1_ag ckpts/cdr1_no_ag \\
                                  --names "CDR1 + Antigen" "CDR1 No Antigen"
  
  # Save to specific directory
  python plot_from_checkpoints.py --ckpt_dirs ckpts/* --save_dir plots/
        """
    )
    
    parser.add_argument('--ckpt_dirs', nargs='+', required=True,
                        help='Checkpoint directories to load')
    parser.add_argument('--names', nargs='+', default=None,
                        help='Custom names for experiments (must match number of ckpt_dirs)')
    parser.add_argument('--save_dir', default='plots',
                        help='Directory to save plots (default: plots)')
    parser.add_argument('--no_show', action='store_true',
                        help='Do not display plots (only save)')
    
    args = parser.parse_args()
    
    # Build experiment dict
    if args.names:
        if len(args.names) != len(args.ckpt_dirs):
            print("Error: Number of names must match number of checkpoint directories")
            return
        experiment_dirs = dict(zip(args.names, args.ckpt_dirs))
    else:
        experiment_dirs = {os.path.basename(d): d for d in args.ckpt_dirs}
    
    # Load experiments
    experiments = load_all_experiments(experiment_dirs)
    
    if not experiments:
        print("No experiments loaded. Check your checkpoint directories.")
        return
    
    # Generate all plots
    plot_all_comparisons(experiments, args.save_dir, show=not args.no_show)


if __name__ == '__main__':
    main()
