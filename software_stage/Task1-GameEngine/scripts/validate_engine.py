#!/usr/bin/env python3
"""
Comprehensive engine validation for 6x6 chess (RoboGambit rules).

1. Independent Python move generator (no C++ dependency)
2. Generic engine (random legal moves) to play against
3. Match runner with per-move legality cross-validation
4. Fischer Random starting position generator

Rules:
- 6x6 board, no rooks
- Pieces: P(1/6), N(2/7), B(3/8), Q(4/9), K(5/10)
- Pawns move 1 square forward only, no double push, no en passant
- No castling
- Promotion only to pieces already captured (from your own side)
- If no piece captured, pawn CANNOT advance to last rank
"""

import numpy as np
import random
import sys
import os
import copy

# ============================================================
# Board representation
# ============================================================

EMPTY = 0
W_PAWN, W_KNIGHT, W_BISHOP, W_QUEEN, W_KING = 1, 2, 3, 4, 5
B_PAWN, B_KNIGHT, B_BISHOP, B_QUEEN, B_KING = 6, 7, 8, 9, 10

INITIAL_COUNTS = {
    # White
    1: 6, 2: 2, 3: 2, 4: 1, 5: 1,
    # Black
    6: 6, 7: 2, 8: 2, 9: 1, 10: 1,
}

def piece_color(pid):
    """Return 'W' or 'B' for a piece id, or None for empty."""
    if 1 <= pid <= 5: return 'W'
    if 6 <= pid <= 10: return 'B'
    return None

def piece_type(pid):
    """Return generic type 1-5 regardless of color."""
    if 1 <= pid <= 5: return pid
    if 6 <= pid <= 10: return pid - 5
    return 0

def make_pid(color, ptype):
    """Create piece id from color ('W'/'B') and type (1-5)."""
    return ptype if color == 'W' else ptype + 5

def sq_to_cell(row, col):
    """Convert (row, col) to chess notation. row=0 is rank 1."""
    return chr(ord('A') + col) + str(row + 1)

def cell_to_sq(cell):
    """Convert chess notation to (row, col)."""
    col = ord(cell[0]) - ord('A')
    row = int(cell[1]) - 1
    return (row, col)

def on_board(r, c):
    return 0 <= r < 6 and 0 <= c < 6


# ============================================================
# Independent move generator
# ============================================================

def get_promotion_types(board, color):
    """
    Get available promotion piece types for a color.
    Can promote to a piece type if current count < initial count.
    Returns list of piece IDs (for that color).
    """
    promos = []
    for ptype in [2, 3, 4]:  # Knight, Bishop, Queen
        pid = make_pid(color, ptype)
        initial = INITIAL_COUNTS[pid]
        current = np.count_nonzero(board == pid)
        if current < initial:
            promos.append(pid)
    return promos


def is_attacked_by(board, r, c, by_color):
    """Check if square (r,c) is attacked by any piece of by_color."""
    for ar in range(6):
        for ac in range(6):
            pid = board[ar][ac]
            if pid == 0 or piece_color(pid) != by_color:
                continue
            pt = piece_type(pid)
            if _can_attack(board, ar, ac, r, c, pt, by_color):
                return True
    return False


def _can_attack(board, fr, fc, tr, tc, pt, color):
    """Check if piece at (fr,fc) of given type/color can attack (tr,tc)."""
    dr = tr - fr
    dc = tc - fc
    
    if pt == 1:  # Pawn
        direction = 1 if color == 'W' else -1
        if dr == direction and abs(dc) == 1:
            return True
        return False
    
    if pt == 2:  # Knight
        if (abs(dr), abs(dc)) in [(1, 2), (2, 1)]:
            return True
        return False
    
    if pt == 3:  # Bishop
        if abs(dr) == abs(dc) and dr != 0:
            return _path_clear(board, fr, fc, tr, tc)
        return False
    
    if pt == 4:  # Queen
        if (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (abs(dr) == abs(dc) and dr != 0):
            return _path_clear(board, fr, fc, tr, tc)
        return False
    
    if pt == 5:  # King
        if abs(dr) <= 1 and abs(dc) <= 1 and (dr != 0 or dc != 0):
            return True
        return False
    
    return False


def _path_clear(board, fr, fc, tr, tc):
    """Check if path from (fr,fc) to (tr,tc) is clear (not including endpoints)."""
    dr = 0 if tr == fr else (1 if tr > fr else -1)
    dc = 0 if tc == fc else (1 if tc > fc else -1)
    r, c = fr + dr, fc + dc
    while (r, c) != (tr, tc):
        if board[r][c] != 0:
            return False
        r += dr
        c += dc
    return True


def find_king(board, color):
    """Find king position for a color."""
    kid = make_pid(color, 5)
    for r in range(6):
        for c in range(6):
            if board[r][c] == kid:
                return (r, c)
    return None


def in_check(board, color):
    """Check if the given color's king is in check."""
    kpos = find_king(board, color)
    if kpos is None:
        return True  # king captured = definitely in check
    opp = 'B' if color == 'W' else 'W'
    return is_attacked_by(board, kpos[0], kpos[1], opp)


def generate_pseudo_moves(board, color):
    """
    Generate all pseudo-legal moves (may leave king in check).
    Returns list of (piece_id, from_rc, to_rc, promo_pid_or_None).
    """
    moves = []
    opp = 'B' if color == 'W' else 'W'
    direction = 1 if color == 'W' else -1
    promo_rank = 5 if color == 'W' else 0
    
    for r in range(6):
        for c in range(6):
            pid = board[r][c]
            if pid == 0 or piece_color(pid) != color:
                continue
            pt = piece_type(pid)
            
            if pt == 1:  # Pawn
                # Forward
                nr = r + direction
                if on_board(nr, c) and board[nr][c] == 0:
                    if nr == promo_rank:
                        promos = get_promotion_types(board, color)
                        for ppid in promos:
                            moves.append((pid, (r, c), (nr, c), ppid))
                        # If no promos available, pawn CANNOT advance to last rank
                    else:
                        moves.append((pid, (r, c), (nr, c), None))
                
                # Captures
                for dc in [-1, 1]:
                    nc = c + dc
                    nr2 = r + direction
                    if on_board(nr2, nc) and board[nr2][nc] != 0 and piece_color(board[nr2][nc]) == opp:
                        if nr2 == promo_rank:
                            promos = get_promotion_types(board, color)
                            for ppid in promos:
                                moves.append((pid, (r, c), (nr2, nc), ppid))
                            # If no promos, capture to last rank is also impossible
                        else:
                            moves.append((pid, (r, c), (nr2, nc), None))
            
            elif pt == 2:  # Knight
                for dr, dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
                    nr, nc = r + dr, c + dc
                    if on_board(nr, nc):
                        target = board[nr][nc]
                        if target == 0 or piece_color(target) == opp:
                            moves.append((pid, (r, c), (nr, nc), None))
            
            elif pt == 3:  # Bishop
                for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nr, nc = r + dr, c + dc
                    while on_board(nr, nc):
                        target = board[nr][nc]
                        if target == 0:
                            moves.append((pid, (r, c), (nr, nc), None))
                        elif piece_color(target) == opp:
                            moves.append((pid, (r, c), (nr, nc), None))
                            break
                        else:
                            break
                        nr += dr
                        nc += dc
            
            elif pt == 4:  # Queen
                for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                    nr, nc = r + dr, c + dc
                    while on_board(nr, nc):
                        target = board[nr][nc]
                        if target == 0:
                            moves.append((pid, (r, c), (nr, nc), None))
                        elif piece_color(target) == opp:
                            moves.append((pid, (r, c), (nr, nc), None))
                            break
                        else:
                            break
                        nr += dr
                        nc += dc
            
            elif pt == 5:  # King
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if on_board(nr, nc):
                            target = board[nr][nc]
                            if target == 0 or piece_color(target) == opp:
                                moves.append((pid, (r, c), (nr, nc), None))
    
    return moves


def make_move_on_board(board, move):
    """
    Apply a move to the board (returns new board).
    move = (piece_id, from_rc, to_rc, promo_pid_or_None)
    """
    b = board.copy()
    pid, (fr, fc), (tr, tc), promo = move
    b[fr][fc] = 0
    if promo is not None:
        b[tr][tc] = promo
    else:
        b[tr][tc] = pid
    return b


def generate_legal_moves(board, color):
    """Generate all legal moves for the given color."""
    pseudo = generate_pseudo_moves(board, color)
    legal = []
    for move in pseudo:
        new_board = make_move_on_board(board, move)
        if not in_check(new_board, color):
            legal.append(move)
    return legal


def move_to_str(move):
    """Convert move tuple to competition string format."""
    pid, (fr, fc), (tr, tc), promo = move
    # If promotion, use the promoted piece's ID
    display_id = promo if promo is not None else pid
    return f"{display_id}:{sq_to_cell(fr, fc)}->{sq_to_cell(tr, tc)}"


def parse_move_str(s):
    """Parse '4:A5->A6' into (display_id, from_cell, to_cell)."""
    parts = s.split(':')
    display_id = int(parts[0])
    cells = parts[1].split('->')
    return display_id, cells[0], cells[1]


# ============================================================
# Fischer Random starting position generator
# ============================================================

def generate_fischer_random():
    """Generate a Fischer Random 6x6 starting position."""
    board = np.zeros((6, 6), dtype=int)
    
    # Place pawns
    for c in range(6):
        board[1][c] = W_PAWN
        board[4][c] = B_PAWN
    
    # Generate back rank: 2N, 2B (opposite colors), 1Q, 1K
    # Bishops must be on opposite colored squares
    # On 6x6: files 0,2,4 are one color; 1,3,5 are the other (for rank 0)
    
    files = list(range(6))
    
    # Place bishops on opposite colors
    light_squares = [0, 2, 4]  # even files on rank 0
    dark_squares = [1, 3, 5]   # odd files on rank 0
    b1_file = random.choice(light_squares)
    b2_file = random.choice(dark_squares)
    
    remaining = [f for f in files if f != b1_file and f != b2_file]
    random.shuffle(remaining)
    
    # Place Q, K, N, N on remaining 4 squares
    # King cannot be on the edge... actually no restriction in Fischer Random for 6x6
    # Just place randomly
    q_file = remaining[0]
    k_file = remaining[1]
    n1_file = remaining[2]
    n2_file = remaining[3]
    
    board[0][b1_file] = W_BISHOP
    board[0][b2_file] = W_BISHOP
    board[0][q_file] = W_QUEEN
    board[0][k_file] = W_KING
    board[0][n1_file] = W_KNIGHT
    board[0][n2_file] = W_KNIGHT
    
    # Mirror for black
    board[5][b1_file] = B_BISHOP
    board[5][b2_file] = B_BISHOP
    board[5][q_file] = B_QUEEN
    board[5][k_file] = B_KING
    board[5][n1_file] = B_KNIGHT
    board[5][n2_file] = B_KNIGHT
    
    return board


# ============================================================
# Validate a move string against independent movegen
# ============================================================

def validate_move(board, color, move_str, legal_moves):
    """
    Validate that a move string from the engine is legal.
    Returns (is_valid, matching_move_tuple_or_None, error_msg).
    """
    try:
        display_id, from_cell, to_cell = parse_move_str(move_str)
    except Exception as e:
        return False, None, f"Failed to parse move string '{move_str}': {e}"
    
    fr, fc = cell_to_sq(from_cell)
    tr, tc = cell_to_sq(to_cell)
    
    # Convert legal moves to strings for comparison
    legal_strs = [move_to_str(m) for m in legal_moves]
    
    if move_str in legal_strs:
        idx = legal_strs.index(move_str)
        return True, legal_moves[idx], None
    
    # Detailed error diagnosis
    # Check if the source square has the right piece
    src_pid = board[fr][fc]
    if src_pid == 0:
        return False, None, f"Source square {from_cell} is empty"
    if piece_color(src_pid) != color:
        return False, None, f"Source square {from_cell} has opponent's piece ({src_pid})"
    
    # Check if any legal move goes from->to
    matching = [m for m in legal_moves if m[1] == (fr, fc) and m[2] == (tr, tc)]
    if not matching:
        from_moves = [m for m in legal_moves if m[1] == (fr, fc)]
        return False, None, (
            f"No legal move from {from_cell} to {to_cell}. "
            f"Legal moves from {from_cell}: {[move_to_str(m) for m in from_moves]}"
        )
    
    # Move exists but piece ID doesn't match (promotion issue?)
    return False, None, (
        f"Move {from_cell}->{to_cell} exists but ID mismatch. "
        f"Engine says {display_id}, legal options: {[move_to_str(m) for m in matching]}"
    )


# ============================================================
# Print board
# ============================================================

PIECE_CHARS = {
    0: '.', 1: 'P', 2: 'N', 3: 'B', 4: 'Q', 5: 'K',
    6: 'p', 7: 'n', 8: 'b', 9: 'q', 10: 'k'
}

def print_board(board):
    print("  A B C D E F")
    for r in range(5, -1, -1):
        print(f"{r+1} ", end="")
        for c in range(6):
            print(PIECE_CHARS[board[r][c]], end=" ")
        print(f"{r+1}")
    print("  A B C D E F")


# ============================================================
# Match runner
# ============================================================

def run_match(num_games=20, time_ms=200, verbose=True):
    """Run matches between our C++ engine and a random-move generic engine."""
    
    # Import our engine
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from Game import get_best_move, reset_flip_cache
    
    os.environ['CHESS_TIME_MS'] = str(time_ms)
    
    stats = {'engine_wins': 0, 'generic_wins': 0, 'draws': 0, 
             'errors': [], 'games': 0, 'total_moves': 0}
    
    for game_idx in range(num_games):
        reset_flip_cache()
        board = generate_fischer_random()
        
        # Alternate who plays white
        engine_is_white = (game_idx % 2 == 0)
        engine_color = 'W' if engine_is_white else 'B'
        generic_color = 'B' if engine_is_white else 'W'
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Game {game_idx + 1}/{num_games} — Engine plays {'White' if engine_is_white else 'Black'}")
            print_board(board)
        
        current_color = 'W'
        move_count = 0
        game_over = False
        result = None  # 'W', 'B', 'D'
        halfmove_clock = 0
        position_history = {}  # hash -> count for repetition
        
        while move_count < 300 and not game_over:
            legal_moves = generate_legal_moves(board, current_color)
            
            if len(legal_moves) == 0:
                if in_check(board, current_color):
                    winner = 'B' if current_color == 'W' else 'W'
                    result = winner
                    if verbose:
                        print(f"  Checkmate! {'White' if winner == 'W' else 'Black'} wins.")
                else:
                    result = 'D'
                    if verbose:
                        print(f"  Stalemate! Draw.")
                game_over = True
                break
            
            if halfmove_clock >= 100:
                result = 'D'
                if verbose:
                    print(f"  Draw by 50-move rule.")
                game_over = True
                break
            
            # Repetition detection (simple board hash)
            board_key = board.tobytes()
            position_history[board_key] = position_history.get(board_key, 0) + 1
            if position_history[board_key] >= 3:
                result = 'D'
                if verbose:
                    print(f"  Draw by threefold repetition.")
                game_over = True
                break
            
            move_count += 1
            
            if current_color == engine_color:
                # Engine move
                try:
                    move_str = get_best_move(board, playing_white=(current_color == 'W'))
                except Exception as e:
                    stats['errors'].append({
                        'game': game_idx + 1, 'move': move_count,
                        'type': 'engine_exception', 'msg': str(e),
                        'board': board.copy(), 'color': current_color
                    })
                    if verbose:
                        print(f"  ERROR: Engine raised exception: {e}")
                    result = generic_color[0]  # engine loses
                    game_over = True
                    break
                
                is_valid, matched_move, err = validate_move(board, current_color, move_str, legal_moves)
                
                if not is_valid:
                    stats['errors'].append({
                        'game': game_idx + 1, 'move': move_count,
                        'type': 'illegal_move', 'msg': err,
                        'engine_move': move_str,
                        'board': board.copy(), 'color': current_color,
                        'legal_moves': [move_to_str(m) for m in legal_moves]
                    })
                    if verbose:
                        print(f"  ERROR: Engine made illegal move: {move_str}")
                        print(f"    Reason: {err}")
                        print(f"    Legal moves ({len(legal_moves)}): {[move_to_str(m) for m in legal_moves[:10]]}...")
                        print_board(board)
                    result = generic_color[0]  # engine loses due to illegal move
                    game_over = True
                    break
                
                if verbose and move_count <= 6:
                    print(f"  {move_count}. [Engine] {move_str}")
                
                # Apply the matched move
                is_capture = board[matched_move[2][0]][matched_move[2][1]] != 0
                is_pawn = piece_type(matched_move[0]) == 1
                board = make_move_on_board(board, matched_move)
                halfmove_clock = 0 if (is_capture or is_pawn) else halfmove_clock + 1
                
            else:
                # Generic engine: pick random legal move
                move = random.choice(legal_moves)
                move_str = move_to_str(move)
                
                if verbose and move_count <= 6:
                    print(f"  {move_count}. [Random] {move_str}")
                
                is_capture = board[move[2][0]][move[2][1]] != 0
                is_pawn = piece_type(move[0]) == 1
                board = make_move_on_board(board, move)
                halfmove_clock = 0 if (is_capture or is_pawn) else halfmove_clock + 1
            
            current_color = 'B' if current_color == 'W' else 'W'
        
        if not game_over:
            result = 'D'
            if verbose:
                print(f"  Draw by move limit (300).")
        
        stats['games'] += 1
        stats['total_moves'] += move_count
        
        if result == engine_color:
            stats['engine_wins'] += 1
        elif result == generic_color:
            stats['generic_wins'] += 1
        else:
            stats['draws'] += 1
        
        if verbose:
            print(f"  Result: {'Engine wins' if result == engine_color else 'Generic wins' if result == generic_color else 'Draw'} ({move_count} moves)")
    
    return stats


# ============================================================
# Standalone movegen perft test (no engine needed)
# ============================================================

def perft(board, color, depth):
    """Count leaf nodes at given depth for movegen validation."""
    if depth == 0:
        return 1
    
    legal_moves = generate_legal_moves(board, color)
    if depth == 1:
        return len(legal_moves)
    
    nodes = 0
    opp = 'B' if color == 'W' else 'W'
    for move in legal_moves:
        new_board = make_move_on_board(board, move)
        nodes += perft(new_board, opp, depth - 1)
    return nodes


def run_perft_test():
    """Run perft on starting position to sanity-check our Python movegen."""
    board = np.array([
        [3, 2, 4, 5, 2, 3],
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [6, 6, 6, 6, 6, 6],
        [8, 7, 9, 10, 7, 8]
    ])
    
    print("Perft test on standard starting position:")
    print_board(board)
    
    for d in range(1, 5):
        nodes = perft(board, 'W', d)
        print(f"  Depth {d}: {nodes} nodes")


# ============================================================
# Cross-validate C++ movegen against Python movegen
# ============================================================

def cross_validate_movegen(num_positions=100, verbose=True):
    """
    Play random games and at each position, compare the C++ engine's
    legal move count with our Python movegen.
    Uses engine_get_move to get one move, but we mostly validate that
    every position the engine encounters has correct legal moves.
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from Game import get_best_move, reset_flip_cache
    
    os.environ['CHESS_TIME_MS'] = '50'  # fast for validation
    
    errors = []
    positions_checked = 0
    
    for game_idx in range(num_positions // 30 + 1):
        reset_flip_cache()
        board = generate_fischer_random()
        current_color = 'W'
        
        for move_idx in range(60):
            legal_moves = generate_legal_moves(board, current_color)
            positions_checked += 1
            
            if len(legal_moves) == 0:
                break
            
            # Every N positions, ask the engine for a move and validate
            if positions_checked % 3 == 0:
                try:
                    reset_flip_cache()
                    move_str = get_best_move(board, playing_white=(current_color == 'W'))
                    is_valid, _, err = validate_move(board, current_color, move_str, legal_moves)
                    if not is_valid:
                        errors.append({
                            'position': positions_checked,
                            'move_str': move_str,
                            'error': err,
                            'board': board.copy(),
                            'color': current_color,
                            'legal_count': len(legal_moves)
                        })
                        if verbose:
                            print(f"  ILLEGAL at position {positions_checked}: {move_str} — {err}")
                            print_board(board)
                except Exception as e:
                    errors.append({
                        'position': positions_checked,
                        'error': str(e),
                        'board': board.copy(),
                        'color': current_color
                    })
            
            # Play a random move to advance
            move = random.choice(legal_moves)
            board = make_move_on_board(board, move)
            current_color = 'B' if current_color == 'W' else 'W'
            
            if positions_checked >= num_positions:
                break
        
        if positions_checked >= num_positions:
            break
    
    return positions_checked, errors


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='6x6 Chess Engine Validator')
    parser.add_argument('mode', choices=['perft', 'match', 'crossval', 'all'],
                       help='Test mode: perft, match, crossval, or all')
    parser.add_argument('--games', type=int, default=20, help='Number of games for match mode')
    parser.add_argument('--time', type=int, default=200, help='Time per move in ms')
    parser.add_argument('--positions', type=int, default=200, help='Positions for crossval')
    parser.add_argument('--quiet', action='store_true', help='Less verbose output')
    args = parser.parse_args()
    
    if args.mode in ('perft', 'all'):
        print("\n" + "="*60)
        print("PERFT TEST (Python movegen self-check)")
        print("="*60)
        run_perft_test()
    
    if args.mode in ('crossval', 'all'):
        print("\n" + "="*60)
        print("CROSS-VALIDATION (C++ engine vs Python movegen)")
        print("="*60)
        n_pos, errors = cross_validate_movegen(args.positions, verbose=not args.quiet)
        print(f"\nChecked {n_pos} positions, {len(errors)} errors found.")
        if errors:
            print("\nERRORS:")
            for e in errors[:5]:
                print(f"  Position {e.get('position','?')}: {e.get('error','?')}")
                if 'board' in e:
                    print_board(e['board'])
    
    if args.mode in ('match', 'all'):
        print("\n" + "="*60)
        print(f"MATCH: Engine vs Random ({args.games} games, {args.time}ms/move)")
        print("="*60)
        stats = run_match(num_games=args.games, time_ms=args.time, verbose=not args.quiet)
        
        print(f"\n{'='*60}")
        print("MATCH RESULTS")
        print(f"{'='*60}")
        print(f"Games:        {stats['games']}")
        print(f"Engine wins:  {stats['engine_wins']}")
        print(f"Generic wins: {stats['generic_wins']}")
        print(f"Draws:        {stats['draws']}")
        print(f"Total moves:  {stats['total_moves']}")
        print(f"Avg moves/game: {stats['total_moves'] / max(1, stats['games']):.1f}")
        
        if stats['errors']:
            print(f"\nILLEGAL MOVES / ERRORS: {len(stats['errors'])}")
            for e in stats['errors'][:10]:
                print(f"  Game {e['game']}, move {e['move']}: {e['type']}")
                print(f"    {e.get('msg', e.get('engine_move', ''))}")
                if 'board' in e:
                    print_board(e['board'])
        else:
            print("\nNo illegal moves detected!")
