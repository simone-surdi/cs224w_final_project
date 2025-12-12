# =============================================================================
# COMPLETE DATA LOADING MODULE FOR ANTIBODY-ANTIGEN PAIRS
# =============================================================================
# File: structgen/data_with_antigen.py
#
# Contains:
# - Original classes: AntibodyDataset, CDRDataset, StructureDataset, StructureLoader
# - New classes: AntibodyAntigenDataset
# - Original functions: completize, featurize
# - New functions: completize_with_antigen
# =============================================================================

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import random

# =============================================================================
# CONSTANTS
# =============================================================================

DUMMY = {
    'pdb': None, 
    'seq': '#' * 10,
    'coords': {
        "N": np.zeros((10, 3)) + np.nan,
        "CA": np.zeros((10, 3)) + np.nan,
        "C": np.zeros((10, 3)) + np.nan,
        "O": np.zeros((10, 3)) + np.nan,
    }
}

alphabet = '#ACDEFGHIKLMNPQRSTVWY'  # 0 is padding

# Amino acid properties
HYDROPATHY = {'#': 0, "I":4.5, "V":4.2, "L":3.8, "F":2.8, "C":2.5, "M":1.9, "A":1.8, "W":-0.9, "G":-0.4, "T":-0.7, "S":-0.8, "Y":-1.3, "P":-1.6, "H":-3.2, "N":-3.5, "D":-3.5, "Q":-3.5, "E":-3.5, "K":-3.9, "R":-4.5}
VOLUME = {'#': 0, "G":60.1, "A":88.6, "S":89.0, "C":108.5, "D":111.1, "P":112.7, "N":114.1, "T":116.1, "E":138.4, "V":140.0, "Q":143.8, "H":153.2, "M":162.9, "I":166.7, "L":166.7, "K":168.6, "R":173.4, "F":189.9, "Y":193.6, "W":227.8}
CHARGE = {**{'R':1, 'K':1, 'D':-1, 'E':-1, 'H':0.1}, **{x:0 for x in 'ABCFGIJLMNOPQSTUVWXYZ#'}}
POLARITY = {**{x:1 for x in 'RNDQEHKSTY'}, **{x:0 for x in "ACGILMFPWV#"}}
ACCEPTOR = {**{x:1 for x in 'DENQHSTY'}, **{x:0 for x in "RKWACGILMFPV#"}}
DONOR = {**{x:1 for x in 'RKWNQHSTY'}, **{x:0 for x in "DEACGILMFPV#"}}
PMAP = lambda x: [HYDROPATHY.get(x, 0) / 5, VOLUME.get(x, 0) / 200, CHARGE.get(x, 0), POLARITY.get(x, 0), ACCEPTOR.get(x, 0), DONOR.get(x, 0)]


# =============================================================================
# ORIGINAL DATASET CLASSES
# =============================================================================

class AntibodyDataset():
    """
    Original antibody dataset (without antigen).
    
    Loads antibody sequences with CDR annotations.
    """

    def __init__(self, jsonl_file, cdr_type='3', max_len=130):
        alphabet_set = set([a for a in alphabet])
        self.data = []
        
        with open(jsonl_file) as f:
            lines = f.readlines()
            for i in range(len(lines)):
                entry = json.loads(lines[i])
                if entry['cdr'] is None or cdr_type not in entry['cdr']:
                    continue

                last_cdr = entry['cdr'].rindex(cdr_type)
                if last_cdr >= max_len - 1:
                    entry['seq'] = entry['seq'][last_cdr - max_len + 10 : last_cdr + 10]
                    entry['cdr'] = entry['cdr'][last_cdr - max_len + 10 : last_cdr + 10]
                    for key, val in entry['coords'].items():
                        entry['coords'][key] = np.asarray(val)[last_cdr - max_len + 10 : last_cdr + 10]
                else:
                    entry['seq'] = entry['seq'][:max_len]
                    entry['cdr'] = entry['cdr'][:max_len]
                    for key, val in entry['coords'].items():
                        entry['coords'][key] = np.asarray(val)[:max_len]

                if entry is not None and len(entry['seq']) > 0:
                    self.data.append(entry)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CDRDataset():
    """
    Dataset that extracts only CDR regions from antibody sequences.
    
    Unlike AntibodyDataset which keeps the full sequence with CDR annotations,
    this dataset extracts just the CDR residues and their coordinates,
    while also storing the "context" (framework with CDR masked out).
    """

    def __init__(self, jsonl_file, hcdr):
        """
        Args:
            jsonl_file: Path to JSONL file
            hcdr: List/string of CDR types to extract (e.g., '123' or ['1','2','3'])
        """
        alphabet_set = set([a for a in alphabet])
        self.cdrs = []  # CDR data
        self.atgs = []  # Antigen data (placeholder, set to None)

        with open(jsonl_file) as f:
            lines = f.readlines()
            for i in range(len(lines)):
                # Extract each requested CDR type
                for cdr_type in hcdr:
                    entry = self.get_cdr(lines[i], cdr_type)
                    if entry is not None and len(entry['seq']) > 0:
                        self.cdrs.append(entry)
                        self.atgs.append(None)

    def get_cdr(self, s, cdr_type):
        """
        Extract CDR region from a JSON line.
        
        Creates:
        - entry['seq']: Just the CDR residues
        - entry['chain']: Full original sequence
        - entry['context']: Full sequence with CDR replaced by '#' (padding)
        - entry['coords']: Coordinates of CDR residues only
        
        Args:
            s: JSON string (one line from file)
            cdr_type: Which CDR to extract ('1', '2', or '3')
            
        Returns:
            Modified entry dict or None if invalid
        """
        entry = json.loads(s)
        seq = entry['seq']
        
        if seq is None or len(cdr_type) == 0:
            return None

        if 'cdr' in entry:
            cdr = entry['cdr']
            
            # Store original full sequence
            entry['chain'] = entry['seq']
            
            # Create context: replace CDR positions with '#' (padding character)
            # Example: seq="ABCDEFGH", cdr="00111100" → context="AB####GH"
            entry['context'] = ''.join([
                (alphabet[0] if y == cdr_type else x)  # alphabet[0] = '#'
                for x, y in zip(seq, cdr)
            ])
            
            # Extract only CDR residues
            # Example: seq="ABCDEFGH", cdr="00111100" → seq="CDEF"
            entry['seq'] = ''.join([
                x for x, y in zip(seq, cdr) if y == cdr_type
            ])
            
            # Create boolean mask for CDR positions
            cdr_mask = np.array([(y == cdr_type) for y in cdr])
        else:
            # No CDR annotation - treat entire sequence as CDR
            cdr_mask = np.array([True] * len(seq))

        # Filter coordinates to only CDR positions
        for key, val in entry['coords'].items():
            val = np.asarray(val)
            val = val[:len(cdr_mask)]  # Ensure same length
            # Apply mask to keep only CDR coordinates
            entry['coords'][key] = val[cdr_mask] if len(cdr_mask) <= len(val) else val

        return entry

    def __len__(self):
        return len(self.cdrs)

    def __getitem__(self, idx):
        return (self.cdrs[idx], self.atgs[idx])


class StructureDataset():
    """
    Simple dataset for loading protein structures without CDR annotations.
    Just loads sequences and coordinates, filtering by alphabet and length.
    """

    def __init__(self, jsonl_file, max_length=100):
        alphabet_set = set([a for a in alphabet])
        
        with open(jsonl_file) as f:
            self.data = []
            lines = f.readlines()
            
            for i, line in enumerate(lines):
                entry = json.loads(line)
                seq = entry['seq']
                
                # Convert coordinate lists to numpy arrays
                for key, val in entry['coords'].items():
                    entry['coords'][key] = np.asarray(val)

                # Validate: check all characters are in alphabet
                bad_chars = set([s for s in seq]).difference(alphabet_set)
                
                # Only keep valid entries within length limit
                if len(bad_chars) == 0 and len(seq) <= max_length:
                    self.data.append(entry)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# =============================================================================
# NEW: ANTIBODY-ANTIGEN DATASET
# =============================================================================

class AntibodyAntigenDataset():
    """
    Dataset for loading antibody-antigen pairs.
    
    Expected JSONL format:
    {
        "pdb": "7d5p",
        "seq": "EVQLVES...",                    # Antibody sequence
        "cdr": "00001111000033330000",          # CDR annotation
        "coords": {"N": [...], "CA": [...], "C": [...], "O": [...]},
        
        # Antigen (from processing script)
        "seq_agchain_A": "MKTLLI...",           # Antigen sequence
        "coords_agchain_A": {"N": [...], "CA": [...], "C": [...], "O": [...]}
    }
    """

    def __init__(self, jsonl_file, cdr_type='3', max_len=130, max_ag_len=200, 
                 require_antigen=False, verbose=True):
        """
        Args:
            jsonl_file: Path to JSONL file
            cdr_type: Which CDR to model ('1', '2', or '3')
            max_len: Maximum antibody length
            max_ag_len: Maximum antigen length
            require_antigen: If True, skip entries without antigen
            verbose: Print loading statistics
        """
        self.data = []
        self.cdr_type = cdr_type
        self.max_len = max_len
        self.max_ag_len = max_ag_len
        
        # Statistics
        stats = {
            'total': 0,
            'no_cdr': 0,
            'with_antigen': 0,
            'without_antigen': 0,
            'errors': 0,
        }
        
        with open(jsonl_file) as f:
            lines = f.readlines()
            
            for line in lines:
                stats['total'] += 1
                
                try:
                    entry = json.loads(line)
                    
                    # === VALIDATE ANTIBODY ===
                    if entry.get('cdr') is None or cdr_type not in entry['cdr']:
                        stats['no_cdr'] += 1
                        continue
                    
                    # === PROCESS ANTIBODY ===
                    entry = self._process_antibody(entry, cdr_type, max_len)
                    if entry is None:
                        stats['errors'] += 1
                        continue
                    
                    # === PROCESS ANTIGEN ===
                    entry = self._process_antigen(entry, max_ag_len)
                    
                    # Track antigen presence
                    if entry.get('ag_seq') is not None:
                        stats['with_antigen'] += 1
                    else:
                        stats['without_antigen'] += 1
                    
                    # Skip if antigen required but not found
                    if require_antigen and entry.get('ag_seq') is None:
                        continue
                    
                    if len(entry['seq']) > 0:
                        self.data.append(entry)
                        
                except Exception as e:
                    stats['errors'] += 1
                    if verbose:
                        print(f"Error processing line: {e}")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Dataset loaded from: {jsonl_file}")
            print(f"{'='*60}")
            print(f"  Total lines:        {stats['total']}")
            print(f"  No CDR{cdr_type}:          {stats['no_cdr']}")
            print(f"  With antigen:       {stats['with_antigen']}")
            print(f"  Without antigen:    {stats['without_antigen']}")
            print(f"  Errors:             {stats['errors']}")
            print(f"  Final dataset size: {len(self.data)}")
            print(f"{'='*60}\n")

    def _process_antibody(self, entry, cdr_type, max_len):
        """Process and truncate antibody sequence/coordinates."""
        try:
            last_cdr = entry['cdr'].rindex(cdr_type)
            
            if last_cdr >= max_len - 1:
                start = last_cdr - max_len + 10
                end = last_cdr + 10
                
                entry['seq'] = entry['seq'][start:end]
                entry['cdr'] = entry['cdr'][start:end]
                
                for key in ['N', 'CA', 'C', 'O']:
                    if key in entry['coords']:
                        coords = entry['coords'][key]
                        if isinstance(coords, list):
                            coords = np.asarray(coords)
                        entry['coords'][key] = coords[start:end]
            else:
                entry['seq'] = entry['seq'][:max_len]
                entry['cdr'] = entry['cdr'][:max_len]
                
                for key in ['N', 'CA', 'C', 'O']:
                    if key in entry['coords']:
                        coords = entry['coords'][key]
                        if isinstance(coords, list):
                            coords = np.asarray(coords)
                        entry['coords'][key] = coords[:max_len]
            
            return entry
            
        except Exception as e:
            return None

    def _process_antigen(self, entry, max_ag_len):
        """Find and process antigen chain(s) from entry."""
        # Find antigen chain(s)
        ag_chains = []
        for key in entry.keys():
            if key.startswith('seq_agchain_'):
                chain_id = key.replace('seq_agchain_', '')
                ag_chains.append(chain_id)
        
        if not ag_chains:
            entry['ag_seq'] = None
            entry['ag_coords'] = None
            entry['ag_chain_id'] = None
            return entry
        
        # Use first antigen chain
        chain_id = ag_chains[0]
        seq_key = f'seq_agchain_{chain_id}'
        coords_key = f'coords_agchain_{chain_id}'
        
        if seq_key in entry and coords_key in entry:
            ag_seq = entry[seq_key]
            ag_coords = entry[coords_key]
            
            if ag_seq and len(ag_seq) > 0:
                ag_seq = ag_seq[:max_ag_len]
                
                processed_coords = {}
                for atom in ['N', 'CA', 'C', 'O']:
                    if atom in ag_coords:
                        coords = ag_coords[atom]
                        if isinstance(coords, list):
                            coords = np.asarray(coords)
                        processed_coords[atom] = coords[:max_ag_len]
                
                if all(atom in processed_coords for atom in ['N', 'CA', 'C', 'O']):
                    entry['ag_seq'] = ag_seq
                    entry['ag_coords'] = processed_coords
                    entry['ag_chain_id'] = chain_id
                    return entry
        
        entry['ag_seq'] = None
        entry['ag_coords'] = None
        entry['ag_chain_id'] = None
        return entry

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_stats(self):
        """Return dataset statistics."""
        n_with_ag = sum(1 for e in self.data if e.get('ag_seq') is not None)
        ag_lens = [len(e['ag_seq']) for e in self.data if e.get('ag_seq') is not None]
        ab_lens = [len(e['seq']) for e in self.data]
        
        return {
            'total': len(self.data),
            'with_antigen': n_with_ag,
            'without_antigen': len(self.data) - n_with_ag,
            'avg_antibody_len': np.mean(ab_lens) if ab_lens else 0,
            'avg_antigen_len': np.mean(ag_lens) if ag_lens else 0,
            'max_antibody_len': max(ab_lens) if ab_lens else 0,
            'max_antigen_len': max(ag_lens) if ag_lens else 0,
        }


# =============================================================================
# STRUCTURE LOADER
# =============================================================================

class StructureLoader():
    """
    Custom data loader that batches by TOKEN COUNT.
    Works with all dataset formats.
    """

    def __init__(self, dataset, batch_tokens, binder_data=None, interval_sort=0):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.batch_tokens = batch_tokens
        self.binder_data = binder_data

        if interval_sort > 0:
            cdr_type = str(interval_sort)
            self.lengths = [dataset[i]['cdr'].count(cdr_type) for i in range(self.size)]
            self.intervals = [
                (dataset[i]['cdr'].index(cdr_type), dataset[i]['cdr'].rindex(cdr_type)) 
                for i in range(self.size)
            ]
            sorted_ix = sorted(range(self.size), key=self.intervals.__getitem__)
        else:
            sorted_ix = np.argsort(self.lengths)

        # Cluster into batches
        clusters, batch = [], []
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_tokens:
                batch.append(ix)
            else:
                if batch:
                    clusters.append(batch)
                batch = [ix]
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            if self.binder_data:
                abatch = [self.binder_data[i] for i in b_idx]
                yield (batch, abatch)
            else:
                yield batch


class StructureLoaderCombined():
    """Loader that sorts by TOTAL length (antibody + antigen)."""
    
    def __init__(self, dataset, batch_tokens, interval_sort=0):
        self.dataset = dataset
        self.size = len(dataset)
        self.batch_tokens = batch_tokens
        
        # Get lengths
        self.ab_lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.ag_lengths = [
            len(dataset[i]['ag_seq']) if dataset[i].get('ag_seq') else 0 
            for i in range(self.size)
        ]
        
        # TOTAL = AB + AG
        self.total_lengths = [
            self.ab_lengths[i] + self.ag_lengths[i] 
            for i in range(self.size)
        ]
        
        # Sort by total length
        if interval_sort > 0:
            cdr_type = str(interval_sort)
            self.intervals = [
                (dataset[i]['cdr'].index(cdr_type), dataset[i]['cdr'].rindex(cdr_type)) 
                for i in range(self.size)
            ]
            sorted_ix = sorted(range(self.size), key=self.intervals.__getitem__)
        else:
            sorted_ix = np.argsort(self.total_lengths)
        
        # Greedy batching
        clusters, batch = [], []
        max_total_in_batch = 0
        
        for ix in sorted_ix:
            total_len = self.total_lengths[ix]
            new_batch_size = len(batch) + 1
            new_max_total = max(max_total_in_batch, total_len)
            new_cost = new_max_total * new_batch_size
            
            if new_cost <= self.batch_tokens:
                batch.append(ix)
                max_total_in_batch = new_max_total
            else:
                if batch:
                    clusters.append(batch)
                batch = [ix]
                max_total_in_batch = total_len
        
        if len(batch) > 0:
            clusters.append(batch)
        
        self.clusters = clusters
        self._print_stats()
    
    def _print_stats(self):
        print(f"\n{'='*60}")
        print(f"StructureLoaderCombined (Sort by AB + AG)")
        print(f"{'='*60}")
        print(f"Dataset size: {self.size}")
        print(f"Number of batches: {len(self.clusters)}")
        print(f"Batch token limit: {self.batch_tokens}")
        
        print(f"\nLength statistics:")
        print(f"  Antibody:  min={min(self.ab_lengths)}, max={max(self.ab_lengths)}, avg={np.mean(self.ab_lengths):.1f}")
        if max(self.ag_lengths) > 0:
            ag_nonzero = [l for l in self.ag_lengths if l > 0]
            print(f"  Antigen:   min={min(ag_nonzero)}, max={max(ag_nonzero)}, avg={np.mean(ag_nonzero):.1f}")
        print(f"  Total:     min={min(self.total_lengths)}, max={max(self.total_lengths)}, avg={np.mean(self.total_lengths):.1f}")
        
        batch_sizes = [len(c) for c in self.clusters]
        print(f"\nBatch sizes: min={min(batch_sizes)}, max={max(batch_sizes)}, avg={np.mean(batch_sizes):.1f}")
        
        print(f"\nFirst 5 batches:")
        for i, cluster in enumerate(self.clusters[:5]):
            ab_lens = [self.ab_lengths[j] for j in cluster]
            ag_lens = [self.ag_lengths[j] for j in cluster]
            total_lens = [self.total_lengths[j] for j in cluster]
            max_total = max(total_lens)
            cost = max_total * len(cluster)
            
            print(f"  Batch {i}: {len(cluster)} samples, max_total={max_total}, cost={cost}")
            print(f"    AB: {ab_lens}")
            print(f"    AG: {ag_lens}")
        
        print(f"{'='*60}\n")
    
    def __len__(self):
        return len(self.clusters)
    
    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            yield [self.dataset[i] for i in b_idx]



# =============================================================================
# ORIGINAL BATCH PROCESSING FUNCTIONS
# =============================================================================

def completize(batch):
    """
    Original completize function (antibody only).
    For backward compatibility.
    
    Args:
        batch: List of entry dicts with 'seq', 'cdr', 'coords'
        
    Returns:
        X: Coordinates [B, N, 4, 3]
        S: Sequence [B, N]
        L: CDR labels (list of strings)
        mask: Valid mask [B, N]
    """
    B = len(batch)
    L = [b['cdr'] for b in batch]
    L_max = max([len(b['seq']) for b in batch])
    
    X = np.zeros([B, L_max, 4, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    mask = np.zeros([B, L_max], dtype=np.float32)

    for i, b in enumerate(batch):
        x = np.stack([b['coords'][c] for c in ['N', 'CA', 'C', 'O']], 1)
        X[i, :len(x), :, :] = x
        l = len(b['seq'])
        indices = np.asarray([alphabet.index(a) for a in b['seq']], dtype=np.int32)
        S[i, :l] = indices
        mask[i, :l] = 1.

    mask = mask * np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[np.isnan(X)] = 0.

    S = torch.from_numpy(S).long().cuda()
    X = torch.from_numpy(X).float().cuda()
    mask = torch.from_numpy(mask).float().cuda()
    
    return X, S, L, mask


def featurize(batch, context=True):
    """
    Original featurize function with amino acid properties.
    
    Args:
        batch: List of entry dicts
        context: Whether to extract context information
        
    Returns:
        (X, S, P, mask): Main tensors
        context: (cS, cmask, crange) if context=True
    """
    B = len(batch)
    L_max = max([len(b['seq']) for b in batch])
    
    X = np.zeros([B, L_max, 4, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    P = np.zeros([B, L_max, 6])
    mask = np.zeros([B, L_max], dtype=np.float32)

    for i, b in enumerate(batch):
        x = np.stack([b['coords'][c] for c in ['N', 'CA', 'C', 'O']], 1)
        if len(x) <= L_max:
            X[i, :len(x), :, :] = x

        l = len(b['seq'])
        indices = np.asarray([alphabet.index(a) for a in b['seq']], dtype=np.int32)
        S[i, :l] = indices
        P[i, :l] = np.array([PMAP(a) for a in b['seq']])
        mask[i, :l] = 1.

    mask = mask * np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[np.isnan(X)] = 0.

    S = torch.from_numpy(S).long().cuda()
    X = torch.from_numpy(X).float().cuda()
    P = torch.from_numpy(P).float().cuda()
    mask = torch.from_numpy(mask).float().cuda()

    if context:
        L_max = max([len(b['context']) for b in batch])
        cS = np.zeros([B, L_max], dtype=np.int32)
        cmask = np.zeros([B, L_max], dtype=np.float32)
        crange = [None] * B
        
        for i, b in enumerate(batch):
            l = len(b['context'])
            indices = np.asarray([alphabet.index(a) for a in b['context']], dtype=np.int32)
            cS[i, :l] = indices
            cmask[i, :l] = 1.
            crange[i] = (b['context'].index('#'), b['context'].rindex('#'))

        cmask = torch.from_numpy(cmask).float().cuda()
        cS = torch.from_numpy(cS).long().cuda()
        context = (cS, cmask, crange)

    return (X, S, P, mask), context


# =============================================================================
# NEW: BATCH PROCESSING WITH ANTIGEN
# =============================================================================

def completize_with_antigen(batch):
    """
    Extended completize function that also processes antigen.
    
    Args:
        batch: List of entry dicts with antibody and optional antigen
        
    Returns:
        X_ab: Antibody coordinates [B, N, 4, 3]
        S_ab: Antibody sequence [B, N]
        L: CDR labels (list of strings)
        mask_ab: Antibody mask [B, N]
        X_ag: Antigen coordinates [B, M, 4, 3] or None
        S_ag: Antigen sequence [B, M] or None
        mask_ag: Antigen mask [B, M] or None
    """
    B = len(batch)
    
    # =========================================================================
    # ANTIBODY
    # =========================================================================
    L = [b['cdr'] for b in batch]
    L_max = max([len(b['seq']) for b in batch])
    
    X_ab = np.zeros([B, L_max, 4, 3])
    S_ab = np.zeros([B, L_max], dtype=np.int32)
    mask_ab = np.zeros([B, L_max], dtype=np.float32)

    for i, b in enumerate(batch):
        coords = b['coords']
        x = np.stack([coords[c] for c in ['N', 'CA', 'C', 'O']], 1)
        X_ab[i, :len(x), :, :] = x
        
        l = len(b['seq'])
        indices = np.asarray([alphabet.index(a) for a in b['seq']], dtype=np.int32)
        S_ab[i, :l] = indices
        mask_ab[i, :l] = 1.

    mask_ab = mask_ab * np.isfinite(np.sum(X_ab, (2, 3))).astype(np.float32)
    X_ab[np.isnan(X_ab)] = 0.

    S_ab = torch.from_numpy(S_ab).long().cuda()
    X_ab = torch.from_numpy(X_ab).float().cuda()
    mask_ab = torch.from_numpy(mask_ab).float().cuda()

    # =========================================================================
    # ANTIGEN
    # =========================================================================
    has_antigen = any(b.get('ag_seq') is not None for b in batch)
    
    if has_antigen:
        ag_lens = [len(b['ag_seq']) if b.get('ag_seq') else 0 for b in batch]
        M_max = max(ag_lens) if max(ag_lens) > 0 else 1
        
        X_ag = np.zeros([B, M_max, 4, 3])
        S_ag = np.zeros([B, M_max], dtype=np.int32)
        mask_ag = np.zeros([B, M_max], dtype=np.float32)
        
        for i, b in enumerate(batch):
            if b.get('ag_seq') is not None and b.get('ag_coords') is not None:
                ag_coords = b['ag_coords']
                ag_seq = b['ag_seq']
                
                try:
                    x_ag = np.stack([ag_coords[c] for c in ['N', 'CA', 'C', 'O']], 1)
                    X_ag[i, :len(x_ag), :, :] = x_ag
                except Exception as e:
                    continue
                
                l_ag = len(ag_seq)
                ag_indices = []
                for a in ag_seq:
                    if a in alphabet:
                        ag_indices.append(alphabet.index(a))
                    else:
                        ag_indices.append(0)
                
                S_ag[i, :l_ag] = np.asarray(ag_indices, dtype=np.int32)
                mask_ag[i, :l_ag] = 1.
        
        mask_ag = mask_ag * np.isfinite(np.sum(X_ag, (2, 3))).astype(np.float32)
        X_ag[np.isnan(X_ag)] = 0.
        
        S_ag = torch.from_numpy(S_ag).long().cuda()
        X_ag = torch.from_numpy(X_ag).float().cuda()
        mask_ag = torch.from_numpy(mask_ag).float().cuda()
    else:
        X_ag, S_ag, mask_ag = None, None, None

    return X_ab, S_ab, L, mask_ab, X_ag, S_ag, mask_ag


def featurize_with_antigen(batch, context=True):
    """
    Extended featurize function with antigen support.
    Includes amino acid property features.
    
    Args:
        batch: List of entry dicts
        context: Whether to extract context information
        
    Returns:
        (X_ab, S_ab, P_ab, mask_ab): Antibody tensors
        context_data: (cS, cmask, crange) or None
        antigen_data: (X_ag, S_ag, P_ag, mask_ag) or None
    """
    B = len(batch)
    
    # =========================================================================
    # ANTIBODY
    # =========================================================================
    L_max = max([len(b['seq']) for b in batch])
    
    X_ab = np.zeros([B, L_max, 4, 3])
    S_ab = np.zeros([B, L_max], dtype=np.int32)
    P_ab = np.zeros([B, L_max, 6])
    mask_ab = np.zeros([B, L_max], dtype=np.float32)

    for i, b in enumerate(batch):
        x = np.stack([b['coords'][c] for c in ['N', 'CA', 'C', 'O']], 1)
        if len(x) <= L_max:
            X_ab[i, :len(x), :, :] = x

        l = len(b['seq'])
        indices = np.asarray([alphabet.index(a) for a in b['seq']], dtype=np.int32)
        S_ab[i, :l] = indices
        P_ab[i, :l] = np.array([PMAP(a) for a in b['seq']])
        mask_ab[i, :l] = 1.

    mask_ab = mask_ab * np.isfinite(np.sum(X_ab, (2, 3))).astype(np.float32)
    X_ab[np.isnan(X_ab)] = 0.

    S_ab = torch.from_numpy(S_ab).long().cuda()
    X_ab = torch.from_numpy(X_ab).float().cuda()
    P_ab = torch.from_numpy(P_ab).float().cuda()
    mask_ab = torch.from_numpy(mask_ab).float().cuda()

    # =========================================================================
    # CONTEXT
    # =========================================================================
    context_data = None
    if context and 'context' in batch[0]:
        L_max_ctx = max([len(b['context']) for b in batch])
        cS = np.zeros([B, L_max_ctx], dtype=np.int32)
        cmask = np.zeros([B, L_max_ctx], dtype=np.float32)
        crange = [None] * B
        
        for i, b in enumerate(batch):
            l = len(b['context'])
            indices = np.asarray([alphabet.index(a) for a in b['context']], dtype=np.int32)
            cS[i, :l] = indices
            cmask[i, :l] = 1.
            crange[i] = (b['context'].index('#'), b['context'].rindex('#'))

        cmask = torch.from_numpy(cmask).float().cuda()
        cS = torch.from_numpy(cS).long().cuda()
        context_data = (cS, cmask, crange)

    # =========================================================================
    # ANTIGEN
    # =========================================================================
    has_antigen = any(b.get('ag_seq') is not None for b in batch)
    
    if has_antigen:
        ag_lens = [len(b['ag_seq']) if b.get('ag_seq') else 0 for b in batch]
        M_max = max(ag_lens) if max(ag_lens) > 0 else 1
        
        X_ag = np.zeros([B, M_max, 4, 3])
        S_ag = np.zeros([B, M_max], dtype=np.int32)
        P_ag = np.zeros([B, M_max, 6])
        mask_ag = np.zeros([B, M_max], dtype=np.float32)
        
        for i, b in enumerate(batch):
            if b.get('ag_seq') is not None and b.get('ag_coords') is not None:
                ag_coords = b['ag_coords']
                ag_seq = b['ag_seq']
                
                try:
                    x_ag = np.stack([ag_coords[c] for c in ['N', 'CA', 'C', 'O']], 1)
                    X_ag[i, :len(x_ag), :, :] = x_ag
                except:
                    continue
                
                l_ag = len(ag_seq)
                ag_indices = []
                ag_props = []
                for a in ag_seq:
                    if a in alphabet:
                        ag_indices.append(alphabet.index(a))
                        ag_props.append(PMAP(a))
                    else:
                        ag_indices.append(0)
                        ag_props.append([0, 0, 0, 0, 0, 0])
                
                S_ag[i, :l_ag] = np.asarray(ag_indices, dtype=np.int32)
                P_ag[i, :l_ag] = np.array(ag_props)
                mask_ag[i, :l_ag] = 1.
        
        mask_ag = mask_ag * np.isfinite(np.sum(X_ag, (2, 3))).astype(np.float32)
        X_ag[np.isnan(X_ag)] = 0.
        
        S_ag = torch.from_numpy(S_ag).long().cuda()
        X_ag = torch.from_numpy(X_ag).float().cuda()
        P_ag = torch.from_numpy(P_ag).float().cuda()
        mask_ag = torch.from_numpy(mask_ag).float().cuda()
        
        antigen_data = (X_ag, S_ag, P_ag, mask_ag)
    else:
        antigen_data = None

    return (X_ab, S_ab, P_ab, mask_ab), context_data, antigen_data


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_rmsd(pred, true, mask):
    """
    Compute RMSD between predicted and true coordinates.
    
    Args:
        pred: Predicted coordinates [B, N, 3]
        true: True coordinates [B, N, 3]
        mask: Valid position mask [B, N]
        
    Returns:
        RMSD value (scalar tensor)
    """
    diff = pred - true
    sq_diff = torch.sum(diff ** 2, dim=-1)
    masked_sq_diff = sq_diff * mask
    mse = masked_sq_diff.sum() / mask.sum().clamp(min=1)
    rmsd = torch.sqrt(mse)
    return rmsd


def pairwise_distance(X, mask):
    """
    Compute pairwise CA distances.
    
    Args:
        X: Coordinates [B, N, 4, 3]
        mask: Valid position mask [B, N]
        
    Returns:
        D: Pairwise distances [B, N, N]
        mask_2D: Valid pair mask [B, N, N]
    """
    X_ca = X[:, :, 1, :]
    dX = X_ca.unsqueeze(1) - X_ca.unsqueeze(2)
    D = torch.sqrt(torch.sum(dX ** 2, dim=-1) + 1e-8)
    mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)
    return D, mask_2D