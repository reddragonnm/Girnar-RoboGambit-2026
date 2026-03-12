"""
EGTB Size Estimator for 6x6 Chess (no rooks)
Piece types: K, Q, B, N, P
Both sides always have exactly 1 king.
Board: 36 squares. Pawns restricted to ranks 2-5 (24 squares).
"""

from math import comb, factorial
from itertools import combinations_with_replacement

SQUARES = 36
PAWN_SQUARES = 24  # ranks 2-5 on a 6x6 board (4 ranks x 6 files)
PIECE_TYPES = ['Q', 'B', 'N', 'P']

def config_label(white_pieces, black_pieces):
    """Generate a human-readable label like KQBvKN"""
    w = 'K' + ''.join(white_pieces)
    b = 'K' + ''.join(black_pieces)
    return f"{w}v{b}"

def count_positions(white_pieces, black_pieces):
    """
    Count maximum positions for a given material configuration.
    
    Kings: King1 on 36 squares, King2 on 35 squares.
    Other pieces placed on remaining squares (non-king squares).
    Pawns are restricted to 24 pawn-legal squares.
    Identical pieces on the same side: divide by permutations.
    Multiply by 2 for side to move.
    """
    # Separate pawns and non-pawns for each side
    w_non_pawns = [p for p in white_pieces if p != 'P']
    w_pawns = [p for p in white_pieces if p == 'P']
    b_non_pawns = [p for p in black_pieces if p != 'P']
    b_pawns = [p for p in black_pieces if p == 'P']
    
    n_extra = len(white_pieces) + len(black_pieces)  # total non-king pieces
    
    # We place pieces sequentially on distinct squares
    # King1: 36 choices
    # King2: 35 choices
    # Then non-pawn extras go on remaining general squares
    # Pawn extras go on remaining pawn-legal squares
    #
    # For maximum index size, we compute the product of available squares
    # for each piece placed sequentially, then divide by factorials for
    # identical pieces on the same side.
    
    # Approach: 
    # Place King1 (36), King2 (35).
    # Remaining general squares: 34, 33, 32, ... for non-pawn pieces
    # Pawn squares: some subset of the 24 pawn squares minus any occupied by kings
    #
    # For an UPPER BOUND (index space), we use:
    # Kings: 36 * 35
    # Non-pawn pieces: placed on remaining squares (34, 33, 32, ...)
    # Pawn pieces: placed on pawn-legal squares (24, 23, 22, ...)
    #   But we need to be careful: a king might be on a pawn square,
    #   reducing available pawn squares. For INDEX SPACE (upper bound),
    #   we typically ignore this interaction and use the maximum.
    #
    # Actually, for tablebase indexing, the standard approach is:
    # - Total positions = (squares for K1) * (squares for K2) * 
    #   product of (squares for each additional piece) / (permutations of identical pieces)
    #   * 2 (side to move)
    #
    # For non-pawn pieces: use remaining general squares (36 - 2 = 34 for first, etc.)
    # For pawn pieces: use pawn squares (24 for first, 23 for second, etc.)
    # But pawns and non-pawns share the board, so placing a non-pawn on a pawn square
    # would reduce pawn squares. For index space we use the simple upper bound.
    
    # Simple sequential placement:
    # Kings take 2 of the 36 squares.
    # Non-pawn extras are placed on remaining squares (34 available, decrementing).
    # Pawn extras are placed on pawn-legal squares (24 available, decrementing).
    # 
    # But actually, non-pawns placed on pawn squares reduce pawn square count.
    # For an INDEX SPACE (which is what tablebases use), we want an upper bound
    # that's easy to compute. The standard approach:
    #
    # positions = 36 * 35 * C(34, n_non_pawns) * n_non_pawns! / dup_non_pawns
    #                     * C(24, n_pawns) * n_pawns! / dup_pawns * 2
    #
    # Wait, let me think more carefully. In tablebases, the index maps to a position.
    # The index space is:
    #
    # For pieces placed on GENERAL squares (36 sq): use falling factorial
    # For pieces placed on PAWN squares (24 sq): use falling factorial  
    # Divide by duplicates.
    #
    # More precisely:
    # K1: 36 positions
    # K2: 35 positions (any square except K1)
    # Non-pawn piece 1: 34 positions (any except K1, K2)
    # Non-pawn piece 2: 33 positions
    # ...
    # Then pawns go on pawn squares. But kings/non-pawns might occupy pawn squares.
    # For index space upper bound:
    # Pawn 1: 24 positions
    # Pawn 2: 23 positions
    # ...
    # This slightly overcounts (a pawn might overlap with a king or non-pawn 
    # on a pawn square), but it's the standard index space.
    #
    # Actually, I think the cleanest way for tablebases:
    # All non-pawn pieces (including kings) are placed on the 36 squares.
    # Total non-pawn pieces = 2 (kings) + len(w_non_pawns) + len(b_non_pawns)
    # These use: 36 * 35 * 34 * ... (falling factorial)
    # Then pawns are placed on the 24 pawn squares:
    # 24 * 23 * 22 * ...
    # Divide by duplicates on each side, multiply by 2 for side to move.

    n_total_non_pawns = 2 + len(w_non_pawns) + len(b_non_pawns)  # including kings
    n_total_pawns = len(w_pawns) + len(b_pawns)
    
    # Falling factorial for non-pawn pieces on 36 squares
    positions = 1
    for i in range(n_total_non_pawns):
        positions *= (SQUARES - i)
    
    # Falling factorial for pawns on 24 pawn squares
    for i in range(n_total_pawns):
        positions *= (PAWN_SQUARES - i)
    
    # Divide by permutations of identical pieces on the same side
    # Count duplicates per side
    from collections import Counter
    
    w_counts = Counter(white_pieces)
    b_counts = Counter(black_pieces)
    
    for piece, count in w_counts.items():
        positions //= factorial(count)
    for piece, count in b_counts.items():
        positions //= factorial(count)
    
    # Side to move
    positions *= 2
    
    return positions

def generate_configs(n_extra):
    """
    Generate all distinct material configurations with n_extra non-king pieces.
    Each config is (white_pieces, black_pieces) where white has >= material.
    
    We need to distribute n_extra pieces among white and black,
    choosing from piece types Q, B, N, P (with repetition allowed per side).
    
    Two configs are the same if swapping colors gives the same material.
    Convention: white has "more" material (by some canonical ordering).
    """
    configs = []
    
    # Generate all ways to pick n_extra pieces (each from PIECE_TYPES)
    # and split them between white and black
    
    # First, generate all multisets of n_extra pieces
    all_piece_combos = list(combinations_with_replacement(PIECE_TYPES, n_extra))
    
    # For each multiset, generate all ways to split into white/black
    seen = set()
    
    for pieces in all_piece_combos:
        pieces = list(pieces)
        # Generate all subsets to assign to white (rest goes to black)
        # Need to handle duplicates carefully
        from itertools import combinations
        
        # Generate all possible splits
        n = len(pieces)
        for r in range(n + 1):  # r pieces go to white
            # Generate all ways to choose r pieces from the multiset for white
            for white_indices in combinations(range(n), r):
                white = tuple(sorted([pieces[i] for i in white_indices]))
                black = tuple(sorted([pieces[i] for i in range(n) if i not in white_indices]))
                
                # Canonical form: white >= black (lexicographically, or by some ordering)
                # We want to avoid counting KXvKY and KYvKX as different
                # Convention: (larger_side, smaller_side) where we compare canonically
                
                # For canonical ordering, use: the side with more pieces is "white"
                # If equal count, use lexicographic order on the sorted tuple
                key1 = (white, black)
                key2 = (black, white)
                
                if key1 in seen or key2 in seen:
                    continue
                
                # Determine canonical form
                if len(white) > len(black):
                    canonical = (white, black)
                elif len(white) < len(black):
                    canonical = (black, white)
                else:
                    # Same number of pieces - use lex order
                    canonical = (min(white, black), max(white, black)) if white != black else (white, black)
                    # Actually we want white >= black
                    canonical = (max(white, black), min(white, black))
                
                if canonical not in seen:
                    seen.add(canonical)
                    configs.append(canonical)
    
    return sorted(configs)

def format_size(nbytes):
    """Format bytes as human-readable size"""
    if nbytes < 1024:
        return f"{nbytes} B"
    elif nbytes < 1024**2:
        return f"{nbytes/1024:.1f} KB"
    elif nbytes < 1024**3:
        return f"{nbytes/1024**2:.1f} MB"
    else:
        return f"{nbytes/1024**3:.2f} GB"

def main():
    grand_totals = {}
    cumulative = 0
    
    for n_pieces in [3, 4, 5]:
        n_extra = n_pieces - 2  # pieces beyond the two kings
        configs = generate_configs(n_extra)
        
        print(f"\n{'='*80}")
        print(f"  {n_pieces}-PIECE TABLEBASES ({n_extra} extra piece{'s' if n_extra > 1 else ''})")
        print(f"{'='*80}")
        print(f"{'Config':<18} {'Positions':>18} {'Size':>12}  Notes")
        print(f"{'-'*18} {'-'*18} {'-'*12}  {'-'*30}")
        
        total_positions = 0
        total_bytes = 0
        
        for white, black in configs:
            label = config_label(list(white), list(black))
            positions = count_positions(list(white), list(black))
            size_bytes = positions  # 1 byte per position
            
            # Note about pawns
            notes = ""
            all_pieces = list(white) + list(black)
            if 'P' in all_pieces:
                notes = f"(pawns on 24 sq)"
            
            # Note about identical pieces
            from collections import Counter
            w_c = Counter(white)
            b_c = Counter(black)
            dups = []
            for p, c in w_c.items():
                if c > 1:
                    dups.append(f"W:{p}x{c}")
            for p, c in b_c.items():
                if c > 1:
                    dups.append(f"B:{p}x{c}")
            if dups:
                notes += f" (÷{','.join(dups)})"
            
            print(f"{label:<18} {positions:>18,} {format_size(size_bytes):>12}  {notes}")
            
            total_positions += positions
            total_bytes += size_bytes
        
        print(f"{'-'*18} {'-'*18} {'-'*12}")
        print(f"{'TOTAL ' + str(n_pieces) + '-piece':<18} {total_positions:>18,} {format_size(total_bytes):>12}  ({len(configs)} tables)")
        
        grand_totals[n_pieces] = (total_positions, total_bytes, len(configs))
        cumulative += total_bytes
        print(f"{'Cumulative':<18} {'':>18} {format_size(cumulative):>12}")
    
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    cum = 0
    for n_pieces in [3, 4, 5]:
        pos, byt, ntables = grand_totals[n_pieces]
        cum += byt
        print(f"  {n_pieces}-piece: {ntables:>4} tables, {pos:>20,} positions, {format_size(byt):>12}, cumulative: {format_size(cum):>12}")
    
    print(f"\n  Total tables: {sum(v[2] for v in grand_totals.values())}")
    print(f"  Total positions: {sum(v[0] for v in grand_totals.values()):,}")
    print(f"  Total size (1 byte/pos): {format_size(sum(v[1] for v in grand_totals.values()))}")


if __name__ == '__main__':
    main()
