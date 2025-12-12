import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from structgen.encoder import MPNEncoder
from structgen.data_with_antigen import alphabet
from structgen.utils import *
from structgen.protein_features import ProteinFeatures, CrossDistanceFeatures


# =============================================================================
# HIERARCHICAL ENCODER (UNCHANGED from original)
# =============================================================================

class HierarchicalEncoder(nn.Module):
    """Message passing encoder for antibody - UNCHANGED from original."""
    
    def __init__(self, args, node_in, edge_in):
        super(HierarchicalEncoder, self).__init__()
        self.node_in, self.edge_in = node_in, edge_in
        self.W_v = nn.Sequential(
                nn.Linear(self.node_in, args.hidden_size, bias=True),
                Normalize(args.hidden_size)
        )
        self.W_e = nn.Sequential(
                nn.Linear(self.edge_in, args.hidden_size, bias=True),
                Normalize(args.hidden_size)
        )
        self.layers = nn.ModuleList([
                MPNNLayer(args.hidden_size, args.hidden_size * 3, dropout=args.dropout)
                for _ in range(args.depth)
        ])
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, V, E, hS, E_idx, mask):
        h_v = self.W_v(V)
        h_e = self.W_e(E)
        nei_s = gather_nodes(hS, E_idx)
        vmask = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        h = h_v
        for layer in self.layers:
            nei_v = gather_nodes(h, E_idx)
            nei_h = torch.cat([nei_v, nei_s, h_e], dim=-1)
            h = layer(h, nei_h, mask_attend=vmask)
            h = h * mask.unsqueeze(-1)
        return h


# =============================================================================
# ANTIGEN ENCODER (NEW)
# =============================================================================

# =============================================================================
# In hierarchical_with_antigen.py - AntigenEncoder class
# =============================================================================

class AntigenEncoder(nn.Module):
    """
    Encode antigen structure using GNN with cross-distance features.
    """
    
    def __init__(self, args):
        # ... (keep __init__ the same) ...
        super(AntigenEncoder, self).__init__()
        
        self.hidden_size = args.hidden_size
        self.k_neighbors = args.k_neighbors
        
        self.features = ProteinFeatures(
            top_k=args.k_neighbors,
            num_rbf=args.num_rbf,
            features_type='full',
            direction='bidirectional'
        )
        self.node_in, self.edge_in = self.features.feature_dimensions['full']
        
        self.cross_features = CrossDistanceFeatures(
            num_rbf=args.num_rbf,
            top_k=getattr(args, 'cross_k_neighbors', 8),
            interface_cutoff=getattr(args, 'interface_cutoff', 10.0)
        )
        
        self.node_in_augmented = self.node_in + args.num_rbf + 1
        
        self.W_v = nn.Sequential(
            nn.Linear(self.node_in_augmented, args.hidden_size, bias=True),
            Normalize(args.hidden_size)
        )
        self.W_e = nn.Sequential(
            nn.Linear(self.edge_in, args.hidden_size, bias=True),
            Normalize(args.hidden_size)
        )
        self.W_s = nn.Embedding(args.vocab_size, args.hidden_size)
        
        self.layers = nn.ModuleList([
            MPNNLayer(args.hidden_size, args.hidden_size * 3, dropout=args.dropout)
            for _ in range(args.depth)
        ])
        
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    
    def forward(self, X_ag, S_ag, mask_ag, X_ab_ca, mask_ab):
        """
        Encode antigen with cross-distance features to antibody.
        
        Args:
            X_ag: Antigen coordinates [B, M, 4, 3]
            S_ag: Antigen sequence [B, M]
            mask_ag: Antigen mask [B, M]
            X_ab_ca: Antibody CA coordinates [B, N, 3]
            mask_ab: Antibody mask [B, N]
            
        Returns:
            h: Antigen node representations [B, M, H]
            cross_info: Dictionary with cross-distance information
            mask_ag: Antigen mask [B, M] (passed through for convenience)
        """
        B, M = S_ag.size(0), S_ag.size(1)
        
        # Extract CA coordinates
        X_ag_ca = X_ag[:, :, 1, :]  # [B, M, 3]
        
        # Compute cross-distance features
        cross_info = self.cross_features(X_ab_ca, X_ag_ca, mask_ab, mask_ag)
        
        # Internal protein features
        V, E, E_idx = self.features(X_ag, mask_ag)
        
        # Augment node features with cross-distance info
        V_augmented = torch.cat([
            V,                                          # [B, M, node_in]
            cross_info['RBF_min'],                      # [B, M, num_rbf]
            cross_info['interface_mask'].unsqueeze(-1)  # [B, M, 1]
        ], dim=-1)
        
        # Embed
        h_v = self.W_v(V_augmented)
        h_e = self.W_e(E)
        h_s = self.W_s(S_ag)
        nei_s = gather_nodes(h_s, E_idx)
        
        # Attention mask
        vmask = gather_nodes(mask_ag.unsqueeze(-1), E_idx).squeeze(-1)
        
        # Message passing
        h = h_v
        for layer in self.layers:
            nei_v = gather_nodes(h, E_idx)
            nei_h = torch.cat([nei_v, nei_s, h_e], dim=-1)
            h = layer(h, nei_h, mask_attend=vmask)
            h = h * mask_ag.unsqueeze(-1)
        
        # === FIX: Return 3 values including mask_ag ===
        return h, cross_info, mask_ag


# =============================================================================
# CROSS-ATTENTION (NEW)
# =============================================================================

class CrossAttentionLayer(nn.Module):
    """
    Cross-attention from antibody to antigen with optional distance bias.
    """
    
    def __init__(self, hidden_size, num_rbf=16, use_distance_bias=True, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        
        self.hidden_size = hidden_size
        self.use_distance_bias = use_distance_bias
        self.num_rbf = num_rbf
        
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        
        if use_distance_bias:
            self.W_dist = nn.Sequential(
                nn.Linear(num_rbf, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, 1),
            )
        
        self.W_out = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    
    def _rbf(self, D):
        """RBF encoding for distances."""
        D_min, D_max = 0., 30.
        D_mu = torch.linspace(D_min, D_max, self.num_rbf).to(D.device)
        D_mu = D_mu.view(1, 1, 1, -1)
        D_sigma = (D_max - D_min) / self.num_rbf
        return torch.exp(-((D.unsqueeze(-1) - D_mu) / D_sigma)**2)
    
    def forward(self, Q_ab, K_V_ag, mask_ag, D_cross=None):
        """
        Cross-attention from antibody (query) to antigen (key/value).
        
        Args:
            Q_ab: Antibody representations [B, N, H]
            K_V_ag: Antigen representations [B, M, H]
            mask_ag: Antigen mask [B, M]
            D_cross: Cross-distance matrix [B, M, N] (optional)
            
        Returns:
            out: Attended representations [B, N, H]
        """
        B, N, H = Q_ab.size()
        M = K_V_ag.size(1)
        
        # Project
        Q = self.W_q(Q_ab)      # [B, N, H]
        K = self.W_k(K_V_ag)    # [B, M, H]
        V = self.W_v(K_V_ag)    # [B, M, H]
        
        # Attention scores [B, N, M]
        att = torch.bmm(Q, K.transpose(1, 2)) / (H ** 0.5)
        
        # Add distance bias
        if self.use_distance_bias and D_cross is not None:
            # D_cross is [B, M, N], transpose to [B, N, M]
            D_cross_t = D_cross.transpose(1, 2)
            D_rbf = self._rbf(D_cross_t)  # [B, N, M, num_rbf]
            dist_bias = self.W_dist(D_rbf).squeeze(-1)  # [B, N, M]
            att = att + dist_bias
        
        # Mask invalid antigen positions
        att = att - 1e6 * (1 - mask_ag.unsqueeze(1))
        att = F.softmax(att, dim=-1)
        
        # Weighted sum
        out = torch.bmm(att, V)  # [B, N, H]
        
        # Combine with query and project
        out = torch.cat([Q_ab, out], dim=-1)
        out = self.W_out(out)
        
        return out


# =============================================================================
# ANTIGEN-AWARE HIERARCHICAL DECODER (MODIFIED)
# =============================================================================

class AntigenAwareHierarchicalDecoder(nn.Module):
    """
    Hierarchical decoder with antigen context for CDR generation.
    
    Based on original HierarchicalDecoder with these additions:
    1. AntigenEncoder to encode antigen structure
    2. CrossAttentionLayer for antibody-antigen interaction
    3. Combined attention (framework + antigen) for prediction
    """

    def __init__(self, args):
        super(AntigenAwareHierarchicalDecoder, self).__init__()
        
        # === ORIGINAL PARAMETERS ===
        self.cdr_type = args.cdr_type
        self.k_neighbors = args.k_neighbors
        self.block_size = args.block_size
        self.update_freq = args.update_freq
        self.hidden_size = args.hidden_size
        self.pos_embedding = PosEmbedding(16)
        
        # === NEW PARAMETERS ===
        self.use_antigen = getattr(args, 'use_antigen', True)
        self.use_distance_bias = getattr(args, 'use_distance_bias', True)
        
        # === ORIGINAL: Protein features ===
        self.features = ProteinFeatures(
                top_k=args.k_neighbors, num_rbf=args.num_rbf,
                features_type='full',
                direction='bidirectional'
        )
        self.node_in, self.edge_in = self.features.feature_dimensions['full']
        
        # === ORIGINAL: Output layers ===
        self.O_d0 = nn.Linear(args.hidden_size, 12)
        self.O_d = nn.Linear(args.hidden_size, 12)
        self.O_s = nn.Linear(args.hidden_size, args.vocab_size)
        self.W_s = nn.Embedding(args.vocab_size, args.hidden_size)

        # === ORIGINAL: Antibody encoders ===
        self.struct_mpn = HierarchicalEncoder(args, self.node_in, self.edge_in)
        self.seq_mpn = HierarchicalEncoder(args, self.node_in, self.edge_in)
        self.init_mpn = HierarchicalEncoder(args, 16, 32)
        
        # === ORIGINAL: Framework encoder ===
        self.rnn = nn.GRU(
                args.hidden_size, args.hidden_size, batch_first=True, 
                num_layers=1, bidirectional=True
        )
        
        # === ORIGINAL: Framework attention ===
        self.W_stc = nn.Sequential(
                nn.Linear(args.hidden_size * 2, args.hidden_size),
                nn.ReLU(),
        )
        self.W_seq = nn.Sequential(
                nn.Linear(args.hidden_size * 2, args.hidden_size),
                nn.ReLU(),
        )
        
        # === NEW: Antigen encoder ===
        if self.use_antigen:
            self.antigen_encoder = AntigenEncoder(args)
            
            # Cross-attention layers
            self.cross_attn_stc = CrossAttentionLayer(
                args.hidden_size, 
                num_rbf=args.num_rbf,
                use_distance_bias=self.use_distance_bias,
                dropout=args.dropout
            )
            self.cross_attn_seq = CrossAttentionLayer(
                args.hidden_size,
                num_rbf=args.num_rbf,
                use_distance_bias=self.use_distance_bias,
                dropout=args.dropout
            )
            
            # Combine framework + antigen attention
            self.W_final_stc = nn.Sequential(
                nn.Linear(args.hidden_size * 2, args.hidden_size),
                nn.ReLU(),
            )
            self.W_final_seq = nn.Sequential(
                nn.Linear(args.hidden_size * 2, args.hidden_size),
                nn.ReLU(),
            )

        # === ORIGINAL: Loss functions ===
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.huber_loss = nn.SmoothL1Loss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    # =========================================================================
    # ORIGINAL HELPER METHODS (UNCHANGED)
    # =========================================================================
    
    def init_struct(self, B, N, K):
        """Initialize structure features - UNCHANGED."""
        pos = torch.arange(N).cuda()
        V = self.pos_embedding(pos.view(1, N, 1))
        V = V.squeeze(2).expand(B, -1, -1)
        pos = pos.unsqueeze(0) - pos.unsqueeze(1)
        D_idx, E_idx = pos.abs().topk(k=K, dim=-1, largest=False)
        E_idx = E_idx.unsqueeze(0).expand(B, -1, -1)
        D_idx = D_idx.unsqueeze(0).expand(B, -1, -1)
        E_rbf = self.features._rbf(3 * D_idx)
        E_pos = self.features.embeddings(E_idx)
        E = torch.cat((E_pos, E_rbf), dim=-1)
        return V, E, E_idx

    def init_coords(self, S, mask):
        """Initialize coordinates - UNCHANGED."""
        B, N = S.size(0), S.size(1)
        K = min(self.k_neighbors, N)
        V, E, E_idx = self.init_struct(B, N, K)
        h = self.init_mpn(V, E, S, E_idx, mask)
        return self.predict_dist(self.O_d0(h))

    def attention(self, Q, context, cmask, W):
        """Framework attention - UNCHANGED."""
        att = torch.bmm(Q, context.transpose(1, 2))
        att = att - 1e6 * (1 - cmask.unsqueeze(1))
        att = F.softmax(att, dim=-1)
        out = torch.bmm(att, context)
        out = torch.cat([Q, out], dim=-1)
        return W(out)

    def predict_dist(self, X):
        """Predict distances - UNCHANGED."""
        X = X.view(X.size(0), X.size(1), 4, 3)
        X_ca = X[:, :, 1, :]
        dX = X_ca[:, None, :, :] - X_ca[:, :, None, :]
        D = torch.sum(dX ** 2, dim=-1)
        V = self.features._dihedrals(X)
        AD = self.features._AD_features(X[:,:,1,:])
        return X.detach().clone(), D, V, AD

    def mask_mean(self, X, mask, i):
        """Masked mean for blocks - UNCHANGED."""
        X = X[:, i:i+self.block_size]
        if X.dim() == 4:
            mask = mask[:, i:i+self.block_size].unsqueeze(-1).unsqueeze(-1)
        else:
            mask = mask[:, i:i+self.block_size].unsqueeze(-1)
        return torch.sum(X * mask, dim=1, keepdims=True) / (mask.sum(dim=1, keepdims=True) + 1e-8)

    def make_X_blocks(self, X, l, r, mask):
        """Make coordinate blocks - UNCHANGED."""
        N = X.size(1)
        lblocks = [self.mask_mean(X, mask, i) for i in range(0, l, self.block_size)]
        rblocks = [self.mask_mean(X, mask, i) for i in range(r + 1, N, self.block_size)]
        bX = torch.cat(lblocks + [X[:, l:r+1]] + rblocks, dim=1)
        return bX.detach()

    def make_S_blocks(self, LS, S, RS, l, r, mask):
        """Make sequence blocks - UNCHANGED."""
        N = S.size(1)
        hS = self.W_s(S)
        LS_blocks = [self.mask_mean(hS, mask, i) for i in range(0, l, self.block_size)]
        RS_blocks = [self.mask_mean(hS, mask, i) for i in range(r + 1, N, self.block_size)]
        bS = torch.cat(LS_blocks + [hS[:, l:r+1]] + RS_blocks, dim=1)
        lmask = [mask[:, i:i+self.block_size].amax(dim=1, keepdims=True) for i in range(0, l, self.block_size)]
        rmask = [mask[:, i:i+self.block_size].amax(dim=1, keepdims=True) for i in range(r + 1, N, self.block_size)]
        bmask = torch.cat(lmask + [mask[:, l:r+1]] + rmask, dim=1)
        return bS, bmask, len(LS_blocks), len(RS_blocks)

    def get_completion_mask(self, B, N, cdr_range):
        """Get CDR mask - UNCHANGED."""
        cmask = torch.zeros(B, N).cuda()
        for i, (l,r) in enumerate(cdr_range):
            cmask[i, l:r+1] = 1
        return cmask

    def remove_cdr_coords(self, X, cdr_range):
        """Remove CDR coords - UNCHANGED."""
        X = X.clone()
        for i, (l,r) in enumerate(cdr_range):
            X[i, l:r+1, :, :] = 0
        return X.clone()

    # =========================================================================
    # NEW HELPER METHOD: Combined attention
    # =========================================================================
    
    def apply_combined_attention(self, h, LS, smask, ag_info, mode='seq', X_ab_ca=None):
        """
        Apply both framework and antigen attention, then combine.
        
        Args:
            h: Antibody hidden states from MPN [B, N_blocked, H]
            LS: Framework context [B, N_orig, H]  
            smask: Framework mask [B, N_orig]
            ag_info: Tuple of (h_ag, cross_info, mask_ag) from antigen encoder, or None
            mode: 'seq' for sequence prediction, 'stc' for structure
            X_ab_ca: Current antibody CA coordinates [B, N_blocked, 3] for recomputing distances
            
        Returns:
            Combined hidden states [B, N_blocked, H]
        """
        # Framework attention (original)
        W_fw = self.W_seq if mode == 'seq' else self.W_stc
        h_fw = self.attention(h, LS, smask, W_fw)
        
        # Antigen attention (new)
        if self.use_antigen and ag_info is not None:
            h_ag, cross_info, mask_ag = ag_info
            
            cross_attn = self.cross_attn_seq if mode == 'seq' else self.cross_attn_stc
            W_final = self.W_final_seq if mode == 'seq' else self.W_final_stc
            
            # IMPORTANT: Don't use pre-computed D_cross because shapes don't match
            # Either recompute or use None
            D_cross = None  # Disable distance bias for now (simpler fix)
            
            h_ag_att = cross_attn(h, h_ag, mask_ag, D_cross=D_cross)
            
            # Combine framework + antigen
            h_combined = W_final(torch.cat([h_fw, h_ag_att], dim=-1))
            return h_combined
        else:
            return h_fw

    # =========================================================================
    # MODIFIED: Forward pass (training)
    # =========================================================================
    
    def forward(self, true_X, true_S, true_cdr, mask, X_ag=None, S_ag=None, mask_ag=None):
        """
        Training forward pass.
        
        MODIFIED: Added X_ag, S_ag, mask_ag parameters and antigen encoding.
        
        Args:
            true_X: Antibody coordinates [B, N, 4, 3]
            true_S: Antibody sequence [B, N]
            true_cdr: CDR labels (list of strings)
            mask: Antibody mask [B, N]
            X_ag: Antigen coordinates [B, M, 4, 3] (NEW)
            S_ag: Antigen sequence [B, M] (NEW)
            mask_ag: Antigen mask [B, M] (NEW)
        """
        B, N = mask.size(0), mask.size(1)
        K = min(self.k_neighbors, N)

        # === NEW: Encode antigen ===
        ag_info = None
        if self.use_antigen and X_ag is not None and S_ag is not None and mask_ag is not None:
            X_ab_ca = true_X[:, :, 1, :]  # [B, N, 3]
            h_ag, cross_info, mask_ag_returned = self.antigen_encoder(X_ag, S_ag, mask_ag, X_ab_ca, mask)
            ag_info = (h_ag, cross_info, mask_ag_returned)

        # === ORIGINAL: CDR range and masks ===
        cdr_range = [(cdr.index(self.cdr_type), cdr.rindex(self.cdr_type)) for cdr in true_cdr]
        T_min = min([l for l,r in cdr_range])
        T_max = max([r for l,r in cdr_range])
        cmask = self.get_completion_mask(B, N, cdr_range)
        smask = mask.clone()

        # === ORIGINAL: Encode framework ===
        S = true_S.clone() * (1 - cmask.long())
        hS, _ = self.rnn(self.W_s(S))
        LS, RS = hS[:, :, :self.hidden_size], hS[:, :, self.hidden_size:]
        hS, mask, offset, suffix = self.make_S_blocks(LS, S, RS, T_min, T_max, mask)
        cmask = torch.cat([cmask.new_zeros(B, offset), cmask[:, T_min:T_max+1], cmask.new_zeros(B, suffix)], dim=1)

        # === ORIGINAL: Ground truth ===
        true_X = self.make_X_blocks(true_X, T_min, T_max, smask)
        true_V = self.features._dihedrals(true_X)
        true_AD = self.features._AD_features(true_X[:,:,1,:])
        true_D, mask_2D = pairwise_distance(true_X, mask)
        true_D = true_D ** 2

        # === ORIGINAL: Initial prediction ===
        sloss = 0.
        X, D, V, AD = self.init_coords(hS, mask)
        X = X.detach().clone()
        dloss = self.huber_loss(D, true_D)
        vloss = self.mse_loss(V, true_V)
        aloss = self.mse_loss(AD, true_AD)

        # === MODIFIED: Iterative generation with antigen attention ===
        for t in range(T_min, T_max + 1):
            # Prepare input
            V, E, E_idx = self.features(X, mask)
            hS = self.make_S_blocks(LS, S, RS, T_min, T_max, smask)[0]

            # MODIFIED: Sequence prediction with combined attention
            h = self.seq_mpn(V, E, hS, E_idx, mask)
            h = self.apply_combined_attention(h, LS, smask, ag_info, mode='seq')
            logits = self.O_s(h[:, offset + t - T_min])
            snll = self.ce_loss(logits, true_S[:, t])
            sloss = sloss + torch.sum(snll * cmask[:, offset + t - T_min])

            # Teacher forcing
            S = S.clone()
            S[:, t] = true_S[:, t]
            S = S.clone()

            # MODIFIED: Structure refinement with combined attention
            if t % self.update_freq == 0:
                h = self.struct_mpn(V, E, hS, E_idx, mask)
                h = self.apply_combined_attention(h, LS, smask, ag_info, mode='stc')
                X, D, V, AD = self.predict_dist(self.O_d(h))
                X = X.detach().clone()
                dloss = dloss + self.huber_loss(D, true_D)
                vloss = vloss + self.mse_loss(V, true_V)
                aloss = aloss + self.mse_loss(AD, true_AD)

        # === ORIGINAL: Compute losses ===
        dloss = torch.sum(dloss * mask_2D) / mask_2D.sum()
        vloss = torch.sum(vloss * mask.unsqueeze(-1)) / mask.sum()
        aloss = torch.sum(aloss * mask.unsqueeze(-1)) / mask.sum()
        sloss = sloss.sum() / cmask.sum()
        loss = sloss + dloss + vloss + aloss
        return loss, sloss

    # =========================================================================
    # MODIFIED: Log probability (evaluation)
    # =========================================================================
    
    def log_prob(self, true_S, true_cdr, mask, X_ag=None, S_ag=None, mask_ag=None):
        """
        Compute log probability for evaluation.
        
        MODIFIED: Added antigen parameters.
        """
        B, N = mask.size(0), mask.size(1)
        K = min(self.k_neighbors, N)

        cdr_range = [(cdr.index(self.cdr_type), cdr.rindex(self.cdr_type)) for cdr in true_cdr]
        T_min = min([l for l,r in cdr_range])
        T_max = max([r for l,r in cdr_range])
        cmask = self.get_completion_mask(B, N, cdr_range)
        smask = mask.clone()

        # Initialize
        S = true_S.clone() * (1 - cmask.long())
        hS, _ = self.rnn(self.W_s(S))
        LS, RS = hS[:, :, :self.hidden_size], hS[:, :, self.hidden_size:]
        hS, mask, offset, suffix = self.make_S_blocks(LS, S, RS, T_min, T_max, mask)
        cmask = torch.cat([cmask.new_zeros(B, offset), cmask[:, T_min:T_max+1], cmask.new_zeros(B, suffix)], dim=1)

        sloss = 0.
        X = self.init_coords(hS, mask)[0]
        X = X.detach().clone()

        # === NEW: Encode antigen ===
        ag_info = None
        if self.use_antigen and X_ag is not None and S_ag is not None and mask_ag is not None:
            X_ab_ca = X[:, :, 1, :]  # Use predicted coordinates
            h_ag, cross_info, mask_ag_returned = self.antigen_encoder(X_ag, S_ag, mask_ag, X_ab_ca, mask)
            ag_info = (h_ag, cross_info, mask_ag_returned)

        for t in range(T_min, T_max + 1):
            V, E, E_idx = self.features(X, mask)
            hS = self.make_S_blocks(LS, S, RS, T_min, T_max, smask)[0]

            # MODIFIED: Combined attention
            h = self.seq_mpn(V, E, hS, E_idx, mask)
            h = self.apply_combined_attention(h, LS, smask, ag_info, mode='seq')
            logits = self.O_s(h[:, offset + t - T_min])
            snll = self.ce_loss(logits, true_S[:, t])
            sloss = sloss + snll * cmask[:, offset + t - T_min]

            # Teacher forcing
            S = S.clone()
            S[:, t] = true_S[:, t]
            S = S.clone()

            # Structure refinement
            if t % self.update_freq == 0:
                h = self.struct_mpn(V, E, hS, E_idx, mask)
                h = self.apply_combined_attention(h, LS, smask, ag_info, mode='stc')
                X = self.predict_dist(self.O_d(h))[0]
                X = X.detach().clone()

        ppl = sloss / cmask.sum(dim=-1)
        sloss = sloss.sum() / cmask.sum()
        return ReturnType(nll=sloss, ppl=ppl, X=X, X_cdr=X[:, offset:offset+T_max-T_min+1])

    # =========================================================================
    # MODIFIED: Generate (sampling)
    # =========================================================================
    
    def generate(self, true_S, true_cdr, mask, X_ag=None, S_ag=None, mask_ag=None, return_ppl=False):
        """
        Generate CDR sequence by sampling.
        
        MODIFIED: Added antigen parameters.
        """
        B, N = mask.size(0), mask.size(1)
        K = min(self.k_neighbors, N)

        cdr_range = [(cdr.index(self.cdr_type), cdr.rindex(self.cdr_type)) for cdr in true_cdr]
        T_min = min([l for l,r in cdr_range])
        T_max = max([r for l,r in cdr_range])
        cmask = self.get_completion_mask(B, N, cdr_range)
        smask = mask.clone()

        # Initialize
        S = true_S.clone() * (1 - cmask.long())
        hS, _ = self.rnn(self.W_s(S))
        LS, RS = hS[:, :, :self.hidden_size], hS[:, :, self.hidden_size:]
        hS, mask, offset, suffix = self.make_S_blocks(LS, S, RS, T_min, T_max, mask)
        cmask = torch.cat([cmask.new_zeros(B, offset), cmask[:, T_min:T_max+1], cmask.new_zeros(B, suffix)], dim=1)

        X = self.init_coords(hS, mask)[0]
        X = X.detach().clone()
        sloss = 0

        # === NEW: Encode antigen ===
        ag_info = None
        if self.use_antigen and X_ag is not None and S_ag is not None and mask_ag is not None:
            X_ab_ca = X[:, :, 1, :]
            h_ag, cross_info, mask_ag_returned = self.antigen_encoder(X_ag, S_ag, mask_ag, X_ab_ca, mask)
            ag_info = (h_ag, cross_info, mask_ag_returned)

        for t in range(T_min, T_max + 1):
            V, E, E_idx = self.features(X, mask)
            hS = self.make_S_blocks(LS, S, RS, T_min, T_max, smask)[0]

            # MODIFIED: Combined attention
            h = self.seq_mpn(V, E, hS, E_idx, mask)
            h = self.apply_combined_attention(h, LS, smask, ag_info, mode='seq')
            logits = self.O_s(h[:, offset + t - T_min])
            prob = F.softmax(logits, dim=-1)
            S[:, t] = torch.multinomial(prob, num_samples=1).squeeze(-1)
            sloss = sloss + self.ce_loss(logits, S[:, t]) * cmask[:, offset + t - T_min]

            # Structure refinement
            h = self.struct_mpn(V, E, hS, E_idx, mask)
            h = self.apply_combined_attention(h, LS, smask, ag_info, mode='stc')
            X = self.predict_dist(self.O_d(h))[0]
            X = X.detach().clone()

        S = S.tolist()
        S = [''.join([alphabet[S[i][j]] for j in range(cdr_range[i][0], cdr_range[i][1] + 1)]) for i in range(B)]
        ppl = torch.exp(sloss / cmask.sum(dim=-1))
        return (S, ppl, X[:, offset:offset+T_max-T_min+1]) if return_ppl else S