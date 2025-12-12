# =============================================================================
# FILE: test_antigen_aware_decoder.py
# Save in your RefineGNN folder and run from Colab
# =============================================================================

import torch
import torch.nn as nn
import numpy as np
import sys
import os


def test_all(REPO_PATH, DATA_PATH):
    """
    Run all tests for the antigen-aware decoder.
    
    Args:
        REPO_PATH: Path to RefineGNN folder
        DATA_PATH: Path to data folder with JSONL files
    """
    
    # Setup paths
    sys.path.insert(0, REPO_PATH)
    sys.path.insert(0, os.path.join(REPO_PATH, 'structgen'))
    
    print("="*70)
    print("ANTIGEN-AWARE DECODER TESTS")
    print("="*70)
    
    # Run tests
    test_imports()
    args = test_create_args()
    test_antigen_encoder(args)
    test_cross_attention(args)
    test_full_decoder(args)
    test_with_real_data(args, DATA_PATH)
    
    print("\n" + "="*70)
    print("✓ ALL TESTS COMPLETE!")
    print("="*70)


# =============================================================================
# TEST 1: Imports
# =============================================================================

def test_imports():
    print("\n" + "─"*70)
    print("TEST 1: Imports")
    print("─"*70)
    
    try:
        from structgen.protein_features import ProteinFeatures, CrossDistanceFeatures
        print("  ✓ ProteinFeatures, CrossDistanceFeatures")
    except ImportError as e:
        print(f"  ✗ ProteinFeatures: {e}")
        return False
    
    try:
        from structgen.hierarchical_with_antigen import (
            HierarchicalEncoder,
            AntigenEncoder,
            CrossAttentionLayer,
            AntigenAwareHierarchicalDecoder
        )
        print("  ✓ HierarchicalEncoder")
        print("  ✓ AntigenEncoder")
        print("  ✓ CrossAttentionLayer")
        print("  ✓ AntigenAwareHierarchicalDecoder")
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False
    
    try:
        from structgen.data_with_antigen import alphabet
        print(f"  ✓ alphabet ({len(alphabet)} amino acids)")
    except ImportError as e:
        print(f"  ✗ alphabet: {e}")
    
    print("\n  All imports successful!")
    return True


# =============================================================================
# TEST 2: Create Args
# =============================================================================

def test_create_args():
    print("\n" + "─"*70)
    print("TEST 2: Create Args")
    print("─"*70)
    
    class Args:
        # Model architecture
        hidden_size = 128
        depth = 3
        dropout = 0.1
        k_neighbors = 9
        num_rbf = 16
        vocab_size = 21  # 20 amino acids + padding
        
        # CDR generation
        cdr_type = '3'
        block_size = 8
        update_freq = 1
        
        # Antigen-specific (NEW)
        use_antigen = True
        use_distance_bias = True
        cross_k_neighbors = 8
        interface_cutoff = 10.0
    
    args = Args()
    
    print(f"  hidden_size:      {args.hidden_size}")
    print(f"  depth:            {args.depth}")
    print(f"  k_neighbors:      {args.k_neighbors}")
    print(f"  num_rbf:          {args.num_rbf}")
    print(f"  vocab_size:       {args.vocab_size}")
    print(f"  cdr_type:         {args.cdr_type}")
    print(f"  use_antigen:      {args.use_antigen}")
    print(f"  use_distance_bias:{args.use_distance_bias}")
    
    print("\n  ✓ Args created successfully!")
    return args


# =============================================================================
# TEST 3: AntigenEncoder
# =============================================================================

def test_antigen_encoder(args):
    print("\n" + "─"*70)
    print("TEST 3: AntigenEncoder")
    print("─"*70)
    
    from structgen.hierarchical_with_antigen import AntigenEncoder
    
    # Create encoder
    encoder = AntigenEncoder(args)
    if torch.cuda.is_available():
        encoder = encoder.cuda()
    
    # Count parameters
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Parameters: {n_params:,}")
    
    # Create dummy data
    B, M, N = 2, 50, 100  # Batch, Antigen length, Antibody length
    
    X_ag = torch.randn(B, M, 4, 3).cuda()      # Antigen coords
    S_ag = torch.randint(0, 21, (B, M)).cuda()  # Antigen sequence
    mask_ag = torch.ones(B, M).cuda()           # Antigen mask
    X_ab_ca = torch.randn(B, N, 3).cuda()       # Antibody CA coords
    mask_ab = torch.ones(B, N).cuda()           # Antibody mask
    
    print(f"\n  Input shapes:")
    print(f"    X_ag:    {X_ag.shape} (B, M, 4, 3)")
    print(f"    S_ag:    {S_ag.shape} (B, M)")
    print(f"    mask_ag: {mask_ag.shape} (B, M)")
    print(f"    X_ab_ca: {X_ab_ca.shape} (B, N, 3)")
    print(f"    mask_ab: {mask_ab.shape} (B, N)")
    
    # Forward pass
    # === FIX: Now expects 3 return values ===
    with torch.no_grad():
        h_ag, cross_info, mask_ag_out = encoder(X_ag, S_ag, mask_ag, X_ab_ca, mask_ab)
    
    print(f"\n  Output shapes:")
    print(f"    h_ag: {h_ag.shape} (B, M, H)")
    print(f"    mask_ag_out: {mask_ag_out.shape} (B, M)")
    print(f"    cross_info keys: {list(cross_info.keys())}")
    
    for key, val in cross_info.items():
        print(f"      {key}: {val.shape}")
    
    # Verify output
    assert h_ag.shape == (B, M, args.hidden_size), f"Wrong h_ag shape: {h_ag.shape}"
    assert cross_info['D_cross'].shape == (B, M, N), f"Wrong D_cross shape"
    assert cross_info['RBF_min'].shape == (B, M, args.num_rbf), f"Wrong RBF_min shape"
    assert mask_ag_out.shape == (B, M), f"Wrong mask_ag_out shape"
    
    print("\n  ✓ AntigenEncoder test passed!")
    return True


# =============================================================================
# TEST 4: CrossAttentionLayer
# =============================================================================

def test_cross_attention(args):
    print("\n" + "─"*70)
    print("TEST 4: CrossAttentionLayer")
    print("─"*70)
    
    from structgen.hierarchical_with_antigen import CrossAttentionLayer
    
    # Create layer
    cross_attn = CrossAttentionLayer(
        hidden_size=args.hidden_size,
        num_rbf=args.num_rbf,
        use_distance_bias=True,
        dropout=args.dropout
    )
    if torch.cuda.is_available():
        cross_attn = cross_attn.cuda()
    
    n_params = sum(p.numel() for p in cross_attn.parameters())
    print(f"  Parameters: {n_params:,}")
    
    # Create dummy data
    B, N, M, H = 2, 100, 50, args.hidden_size
    
    Q_ab = torch.randn(B, N, H).cuda()       # Antibody query
    K_V_ag = torch.randn(B, M, H).cuda()     # Antigen key/value
    mask_ag = torch.ones(B, M).cuda()        # Antigen mask
    D_cross = torch.rand(B, M, N).cuda() * 30  # Cross-distances (0-30 Å)
    
    print(f"\n  Input shapes:")
    print(f"    Q_ab:    {Q_ab.shape} (B, N, H)")
    print(f"    K_V_ag:  {K_V_ag.shape} (B, M, H)")
    print(f"    mask_ag: {mask_ag.shape} (B, M)")
    print(f"    D_cross: {D_cross.shape} (B, M, N)")
    
    # Forward WITHOUT distance bias
    with torch.no_grad():
        out_no_bias = cross_attn(Q_ab, K_V_ag, mask_ag, D_cross=None)
    
    print(f"\n  Without distance bias:")
    print(f"    Output: {out_no_bias.shape} (B, N, H)")
    
    # Forward WITH distance bias
    with torch.no_grad():
        out_with_bias = cross_attn(Q_ab, K_V_ag, mask_ag, D_cross=D_cross)
    
    print(f"\n  With distance bias:")
    print(f"    Output: {out_with_bias.shape} (B, N, H)")
    
    # Verify outputs are different (distance bias should matter)
    diff = (out_no_bias - out_with_bias).abs().mean().item()
    print(f"    Mean difference: {diff:.6f}")
    
    assert out_no_bias.shape == (B, N, H), f"Wrong output shape"
    assert out_with_bias.shape == (B, N, H), f"Wrong output shape"
    assert diff > 0, "Distance bias should change output"
    
    # Test with masked antigen positions
    mask_ag_partial = torch.ones(B, M).cuda()
    mask_ag_partial[:, M//2:] = 0  # Mask half of antigen
    
    with torch.no_grad():
        out_masked = cross_attn(Q_ab, K_V_ag, mask_ag_partial, D_cross=D_cross)
    
    print(f"\n  With partial mask (50% antigen masked):")
    print(f"    Output: {out_masked.shape} (B, N, H)")
    
    print("\n  ✓ CrossAttentionLayer test passed!")
    return True


# =============================================================================
# TEST 5: Full Decoder
# =============================================================================

def test_full_decoder(args):
    print("\n" + "─"*70)
    print("TEST 5: AntigenAwareHierarchicalDecoder (full model)")
    print("─"*70)
    
    from structgen.hierarchical_with_antigen import AntigenAwareHierarchicalDecoder
    
    # Create decoder
    decoder = AntigenAwareHierarchicalDecoder(args)
    if torch.cuda.is_available():
        decoder = decoder.cuda()
    
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Total parameters: {n_params:,}")
    
    # Count parameters by component
    param_counts = {}
    for name, module in decoder.named_children():
        count = sum(p.numel() for p in module.parameters())
        if count > 0:
            param_counts[name] = count
    
    print(f"\n  Parameters by component:")
    for name, count in sorted(param_counts.items(), key=lambda x: -x[1]):
        print(f"    {name}: {count:,}")
    
    # Create dummy data
    B, N, M = 2, 120, 60  # Batch, Antibody length, Antigen length
    
    # Antibody
    true_X = torch.randn(B, N, 4, 3).cuda()
    true_S = torch.randint(0, 21, (B, N)).cuda()
    mask = torch.ones(B, N).cuda()
    
    # CDR labels: framework=0, CDR1=1, CDR2=2, CDR3=3
    # Example: CDR3 at positions 100-110
    true_cdr = ['0' * 100 + '3' * 11 + '0' * 9] * B
    
    # Antigen
    X_ag = torch.randn(B, M, 4, 3).cuda()
    S_ag = torch.randint(0, 21, (B, M)).cuda()
    mask_ag = torch.ones(B, M).cuda()
    
    print(f"\n  Input shapes:")
    print(f"    true_X:  {true_X.shape} (B, N, 4, 3)")
    print(f"    true_S:  {true_S.shape} (B, N)")
    print(f"    mask:    {mask.shape} (B, N)")
    print(f"    true_cdr: '{true_cdr[0][:20]}...{true_cdr[0][-20:]}'")
    print(f"    X_ag:    {X_ag.shape} (B, M, 4, 3)")
    print(f"    S_ag:    {S_ag.shape} (B, M)")
    print(f"    mask_ag: {mask_ag.shape} (B, M)")
    
    # === TEST FORWARD (training) ===
    print(f"\n  Testing forward() (training)...")
    decoder.train()
    
    loss, sloss = decoder(
        true_X, true_S, true_cdr, mask,
        X_ag=X_ag, S_ag=S_ag, mask_ag=mask_ag
    )
    
    print(f"    loss:  {loss.item():.4f}")
    print(f"    sloss: {sloss.item():.4f}")
    
    # Check gradient flow
    loss.backward()
    
    grad_norms = {}
    for name, param in decoder.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    
    print(f"    Gradient norms (sample):")
    for name in list(grad_norms.keys())[:5]:
        print(f"      {name}: {grad_norms[name]:.6f}")
    
    # === TEST WITHOUT ANTIGEN (backward compatibility) ===
    print(f"\n  Testing forward() WITHOUT antigen...")
    decoder.zero_grad()
    
    loss_no_ag, sloss_no_ag = decoder(
        true_X, true_S, true_cdr, mask,
        X_ag=None, S_ag=None, mask_ag=None
    )
    
    print(f"    loss:  {loss_no_ag.item():.4f}")
    print(f"    sloss: {sloss_no_ag.item():.4f}")
    
    # === TEST LOG_PROB (evaluation) ===
    print(f"\n  Testing log_prob() (evaluation)...")
    decoder.eval()
    
    with torch.no_grad():
        result = decoder.log_prob(
            true_S, true_cdr, mask,
            X_ag=X_ag, S_ag=S_ag, mask_ag=mask_ag
        )
    
    print(f"    nll: {result.nll.item():.4f}")
    print(f"    ppl shape: {result.ppl.shape}")
    print(f"    ppl mean: {result.ppl.mean().item():.4f}")
    print(f"    X shape: {result.X.shape}")
    print(f"    X_cdr shape: {result.X_cdr.shape}")
    
    # === TEST GENERATE (sampling) ===
    print(f"\n  Testing generate() (sampling)...")
    
    with torch.no_grad():
        sequences, ppl, X_cdr = decoder.generate(
            true_S, true_cdr, mask,
            X_ag=X_ag, S_ag=S_ag, mask_ag=mask_ag,
            return_ppl=True
        )
    
    print(f"    Generated sequences:")
    for i, seq in enumerate(sequences):
        print(f"      Sample {i}: {seq}")
    print(f"    ppl: {ppl.tolist()}")
    print(f"    X_cdr shape: {X_cdr.shape}")
    
    print("\n  ✓ Full decoder test passed!")
    return True


# =============================================================================
# TEST 6: With Real Data
# =============================================================================

def test_with_real_data(args, DATA_PATH):
    print("\n" + "─"*70)
    print("TEST 6: With Real Data")
    print("─"*70)
    
    from structgen.data_with_antigen import (
        AntibodyAntigenDataset,
        StructureLoader,
        completize_with_antigen
    )
    from structgen.hierarchical_with_antigen import AntigenAwareHierarchicalDecoder
    
    # Check for data file
    data_file = os.path.join(DATA_PATH, 'train_data.jsonl')
    if not os.path.exists(data_file):
        print(f"  ⚠ Data file not found: {data_file}")
        print("    Skipping real data test")
        return False
    
    # Load dataset
    print(f"  Loading dataset from: {data_file}")
    dataset = AntibodyAntigenDataset(
        data_file,
        cdr_type='3',
        max_len=150,
        max_ag_len=200,
        require_antigen=False,
        verbose=False
    )
    
    stats = dataset.get_stats()
    print(f"  Dataset stats:")
    for key, val in stats.items():
        print(f"    {key}: {val}")
    
    # Create loader
    loader = StructureLoader(dataset.data, batch_tokens=500)
    print(f"  Batches: {len(loader)}")
    
    # Create model
    decoder = AntigenAwareHierarchicalDecoder(args)
    if torch.cuda.is_available():
        decoder = decoder.cuda()
    decoder.train()
    
    # Test on first batch
    print(f"\n  Testing on first batch...")
    
    for batch in loader:
        X_ab, S_ab, L, mask_ab, X_ag, S_ag, mask_ag = completize_with_antigen(batch)
        
        print(f"    Batch size: {len(batch)}")
        print(f"    X_ab: {X_ab.shape}")
        print(f"    S_ab: {S_ab.shape}")
        print(f"    mask_ab: {mask_ab.shape}")
        
        if X_ag is not None:
            print(f"    X_ag: {X_ag.shape}")
            print(f"    S_ag: {S_ag.shape}")
            print(f"    mask_ag: {mask_ag.shape}")
        else:
            print(f"    X_ag: None (no antigen in batch)")
        
        # Forward pass
        loss, sloss = decoder(
            X_ab, S_ab, L, mask_ab,
            X_ag=X_ag, S_ag=S_ag, mask_ag=mask_ag
        )
        
        print(f"\n    Loss: {loss.item():.4f}")
        print(f"    Sequence loss: {sloss.item():.4f}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        total_grad_norm = 0
        for param in decoder.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        print(f"    Total gradient norm: {total_grad_norm:.4f}")
        
        break  # Only test first batch
    
    print("\n  ✓ Real data test passed!")
    return True


# =============================================================================
# TEST 7: Compare With and Without Antigen
# =============================================================================

def test_compare_with_without_antigen(args, DATA_PATH):
    print("\n" + "─"*70)
    print("TEST 7: Compare With vs Without Antigen")
    print("─"*70)
    
    from structgen.data_with_antigen import (
        AntibodyAntigenDataset,
        StructureLoader,
        completize_with_antigen
    )
    from structgen.hierarchical_with_antigen import AntigenAwareHierarchicalDecoder
    
    # Load data
    data_file = os.path.join(DATA_PATH, 'train_data.jsonl')
    if not os.path.exists(data_file):
        print(f"  ⚠ Skipping - no data file")
        return False
    
    dataset = AntibodyAntigenDataset(
        data_file,
        cdr_type='3',
        max_len=150,
        max_ag_len=200,
        require_antigen=False,
        verbose=False
    )
    
    loader = StructureLoader(dataset.data, batch_tokens=300)
    
    # Create model
    decoder = AntigenAwareHierarchicalDecoder(args)
    if torch.cuda.is_available():
        decoder = decoder.cuda()
    decoder.eval()
    
    # Compare losses
    print(f"\n  Comparing losses with vs without antigen:")
    print(f"  {'─'*50}")
    print(f"  {'Batch':<8} {'With AG':<12} {'Without AG':<12} {'Diff':<12}")
    print(f"  {'─'*50}")
    
    for i, batch in enumerate(loader):
        if i >= 5:
            break
        
        X_ab, S_ab, L, mask_ab, X_ag, S_ag, mask_ag = completize_with_antigen(batch)
        
        with torch.no_grad():
            # With antigen
            loss_with, _ = decoder(
                X_ab, S_ab, L, mask_ab,
                X_ag=X_ag, S_ag=S_ag, mask_ag=mask_ag
            )
            
            # Without antigen
            loss_without, _ = decoder(
                X_ab, S_ab, L, mask_ab,
                X_ag=None, S_ag=None, mask_ag=None
            )
        
        diff = loss_with.item() - loss_without.item()
        has_ag = "✓" if X_ag is not None else "✗"
        
        print(f"  {i:<8} {loss_with.item():<12.4f} {loss_without.item():<12.4f} {diff:<12.4f} {has_ag}")
    
    print(f"  {'─'*50}")
    print("\n  ✓ Comparison test complete!")
    return True


# =============================================================================
# MAIN (for running from Colab)
# =============================================================================

if __name__ == "__main__":
    print("Run from Colab with:")
    print("  from test_antigen_aware_decoder import test_all")
    print("  test_all(REPO_PATH, DATA_PATH)")