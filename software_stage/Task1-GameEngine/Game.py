"""
RoboGambit 2025-26 — Task 1: Autonomous Game Engine
Organised by Aries and Robotics Club, IIT Delhi

Board: 6x6 NumPy array
  - 0  : Empty cell
  - 1  : White Pawn
  - 2  : White Knight
  - 3  : White Bishop
  - 4  : White Queen
  - 5  : White King
  - 6  : Black Pawn
  - 7  : Black Knight
  - 8  : Black Bishop
  - 9  : Black Queen
  - 10 : Black King

Board coordinates:
  - Bpttom-left  = A1  (index [0][0])
  - Columns   = A–F (left to right)
  - Rows      = 6-1 (top to bottom)(from white's perspective)

Move output format:  "<piece_id>:<source_cell>-><target_cell>"
  e.g.  "1:B3->B4"   (White Pawn moves from B3 to B4)
"""

import numpy as np
from typing import Optional
import ctypes
import os
import subprocess
import re

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMPTY = 0

# Piece IDs
WHITE_PAWN   = 1
WHITE_KNIGHT = 2
WHITE_BISHOP = 3
WHITE_QUEEN  = 4
WHITE_KING   = 5
BLACK_PAWN   = 6
BLACK_KNIGHT = 7
BLACK_BISHOP = 8
BLACK_QUEEN  = 9
BLACK_KING   = 10

WHITE_PIECES = {WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_QUEEN, WHITE_KING}
BLACK_PIECES = {BLACK_PAWN, BLACK_KNIGHT, BLACK_BISHOP, BLACK_QUEEN, BLACK_KING}

BOARD_SIZE = 6

PIECE_VALUES = {
    WHITE_PAWN:   100,
    WHITE_KNIGHT: 300,
    WHITE_BISHOP: 320,
    WHITE_QUEEN:  900,
    WHITE_KING:  20000,
    BLACK_PAWN:  -100,
    BLACK_KNIGHT:-300,
    BLACK_BISHOP:-320,
    BLACK_QUEEN: -900,
    BLACK_KING: -20000,
}
# Column index -> letter
COL_TO_FILE = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}
FILE_TO_COL = {v: k for k, v in COL_TO_FILE.items()}

# Default search time (milliseconds). Override with CHESS_TIME_MS env var.
DEFAULT_TIME_MS = int(os.environ.get("CHESS_TIME_MS", "1000"))

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def idx_to_cell(row: int, col: int) -> str:
    """Convert (row, col) zero-indexed to board notation e.g. (0,0) -> 'A1'."""
    return f"{COL_TO_FILE[col]}{row + 1}"

def cell_to_idx(cell: str):
    """Convert board notation e.g. 'A1' -> (row=0, col=0)."""
    col = FILE_TO_COL[cell[0].upper()]
    row = int(cell[1]) - 1
    return row, col

def in_bounds(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

def is_white(piece: int) -> bool:
    return piece in WHITE_PIECES

def is_black(piece: int) -> bool:
    return piece in BLACK_PIECES

def same_side(p1: int, p2: int) -> bool:
    return (is_white(p1) and is_white(p2)) or (is_black(p1) and is_black(p2))

# ---------------------------------------------------------------------------
# Board orientation detection and flipping
# ---------------------------------------------------------------------------

_flip_cached: Optional[bool] = None   # orientation decision, set once

def _detect_flip(board: np.ndarray) -> bool:
    """
    Detect whether the board is in 'flipped' orientation.

    Our C++ engine convention:
        White pieces start on low ranks (1-2), advance toward rank 6.
        Black pieces start on high ranks (5-6), advance toward rank 1.

    If the evaluator sends the board with White predominantly on HIGH rows
    (rows 3-5, i.e. ranks 4-6) and Black on LOW rows (rows 0-2, i.e. ranks
    1-3), we need to flip the board vertically before passing to the engine,
    and then un-flip the resulting move coordinates.

    Detection uses the centre-of-mass of each side's pieces.
    """
    w_rows = []
    b_rows = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = int(board[r][c])
            if 1 <= p <= 5:
                w_rows.append(r)
            elif 6 <= p <= 10:
                b_rows.append(r)

    if not w_rows or not b_rows:
        return False  # Can't determine; assume standard

    return float(np.mean(w_rows)) > float(np.mean(b_rows))


def _needs_flip(board: np.ndarray) -> bool:
    """
    Return whether the board needs flipping.

    The decision is made once on the first call (using the opening position
    where piece centre-of-mass is reliable) and cached for all subsequent
    calls so that mid-game piece movement can never cause the orientation
    to toggle.
    """
    global _flip_cached
    if _flip_cached is None:
        _flip_cached = _detect_flip(board)
    return _flip_cached


def reset_flip_cache():
    """Reset the cached orientation and game history (call when starting a new game)."""
    global _flip_cached, _game_hashes
    _flip_cached = None
    _game_hashes = []
    # Also tell C++ engine to clear its history
    if _engine_initialized and _engine_lib:
        _engine_lib.engine_clear_game_history()


def _flip_board(board: np.ndarray) -> np.ndarray:
    """Flip the board vertically (reverse row order)."""
    return board[::-1].copy()


def _flip_move_string(move_str: str) -> str:
    """
    Flip rank digits in a move string.

    The vertical flip maps rank r -> (7 - r) for ranks 1-6.
    E.g. "1:A2->A3" becomes "1:A5->A4".
    """
    # Pattern: <id>:<file><rank>-><file><rank>
    # We flip each rank digit that appears after a file letter
    def flip_rank(m):
        return str(7 - int(m.group(1)))

    return re.sub(r'([A-F])([1-6])', lambda m: m.group(1) + str(7 - int(m.group(2))), move_str)

# ---------------------------------------------------------------------------
# C++ Engine wrapper (loads libchess6x6.so via ctypes)
# ---------------------------------------------------------------------------

_engine_lib = None
_engine_initialized = False
_game_hashes: list = []  # Zobrist hash history for cross-call repetition detection


def _get_engine_dir() -> str:
    """Return the directory containing this script (and the shared lib)."""
    return os.path.dirname(os.path.abspath(__file__))


def _ensure_engine():
    """Lazy-load and initialize the C++ engine shared library."""
    global _engine_lib, _engine_initialized

    if _engine_initialized:
        return

    engine_dir = _get_engine_dir()
    lib_path = os.path.join(engine_dir, "libchess6x6.so")

    # Auto-build if missing
    if not os.path.exists(lib_path):
        print("[Game] Building engine shared library ...")
        subprocess.run(["make", "lib"], cwd=engine_dir, check=True)

    _engine_lib = ctypes.CDLL(lib_path)

    # ---------- function signatures ----------
    _engine_lib.engine_init.restype = None
    _engine_lib.engine_init.argtypes = []

    _engine_lib.engine_get_move.restype = ctypes.c_int
    _engine_lib.engine_get_move.argtypes = [
        ctypes.POINTER(ctypes.c_int),   # board_arr (36 ints)
        ctypes.c_int,                    # side_to_move
        ctypes.c_int,                    # time_limit_ms
        ctypes.POINTER(ctypes.c_int),    # captured_white (nullable)
        ctypes.POINTER(ctypes.c_int),    # captured_black (nullable)
        ctypes.c_char_p,                 # result_buf
        ctypes.c_int                     # buf_size
    ]

    _engine_lib.engine_get_nodes.restype = ctypes.c_int
    _engine_lib.engine_get_nodes.argtypes = []
    _engine_lib.engine_get_depth.restype = ctypes.c_int
    _engine_lib.engine_get_depth.argtypes = []
    _engine_lib.engine_get_score.restype = ctypes.c_int
    _engine_lib.engine_get_score.argtypes = []

    _engine_lib.engine_static_eval.restype = ctypes.c_int
    _engine_lib.engine_static_eval.argtypes = [
        ctypes.POINTER(ctypes.c_int),   # board_arr (36 ints)
        ctypes.c_int,                    # side_to_move
    ]

    _engine_lib.engine_set_game_history.restype = None
    _engine_lib.engine_set_game_history.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),  # hashes array
        ctypes.c_int,                      # count
    ]

    _engine_lib.engine_clear_game_history.restype = None
    _engine_lib.engine_clear_game_history.argtypes = []

    _engine_lib.engine_get_hash.restype = None
    _engine_lib.engine_get_hash.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),  # hash_out (2 x uint32)
    ]

    _engine_lib.engine_cleanup.restype = None
    _engine_lib.engine_cleanup.argtypes = []

    _engine_lib.engine_stop_search.restype = None
    _engine_lib.engine_stop_search.argtypes = []

    _engine_lib.engine_init()

    _engine_initialized = True


def _engine_search(board: np.ndarray, side_to_move: int,
                   time_limit_ms: int = DEFAULT_TIME_MS) -> Optional[str]:
    """
    Call the C++ engine and return the best move string, or None.
    Board must already be in the engine's expected orientation
    (White on low ranks).

    Also passes game position history for cross-call repetition detection
    and records the new position hash afterward.
    """
    global _game_hashes
    _ensure_engine()

    flat = board.flatten().astype(np.int32)
    assert len(flat) == 36, f"Board must be 6x6 (36 elements), got {len(flat)}"

    board_ptr = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    buf = ctypes.create_string_buffer(64)

    # Pass game history for repetition detection
    if _game_hashes:
        HashArray = ctypes.c_uint64 * len(_game_hashes)
        hash_arr = HashArray(*_game_hashes)
        _engine_lib.engine_set_game_history(hash_arr, len(_game_hashes))
    else:
        _engine_lib.engine_set_game_history(None, 0)

    result = _engine_lib.engine_get_move(
        board_ptr, side_to_move, time_limit_ms,
        None, None,   # captured counts inferred from board
        buf, 64
    )

    # Retrieve the Zobrist hash of the position we just searched
    # (the position *before* our move — i.e. the current board state)
    hash_out = (ctypes.c_uint32 * 2)()
    _engine_lib.engine_get_hash(hash_out)
    current_hash = hash_out[0] | (hash_out[1] << 32)
    _game_hashes.append(current_hash)

    if result < 0:
        return None  # No legal moves

    return buf.value.decode("ascii")


def _engine_stats():
    """Return (nodes, depth, score) from the last search."""
    _ensure_engine()
    return (
        _engine_lib.engine_get_nodes(),
        _engine_lib.engine_get_depth(),
        _engine_lib.engine_get_score(),
    )


def engine_eval(board: np.ndarray, playing_white_to_move: bool) -> int:
    """
    Return the C++ engine's static evaluation in centipawns from White's
    perspective. Uses the full tapered eval (material + PST + mobility +
    pawn structure + king safety), NOT search.
    """
    _ensure_engine()
    flipped = _needs_flip(board)
    eng_board = _flip_board(board) if flipped else board
    side = 0 if playing_white_to_move else 1
    flat = eng_board.flatten().astype(np.int32)
    board_ptr = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    return _engine_lib.engine_static_eval(board_ptr, side)

# ---------------------------------------------------------------------------
# Move generation helpers (Python fallback / validation)
# ---------------------------------------------------------------------------

# Initial piece counts per side for RoboGambit Fischer Random (6x6)
_INITIAL_COUNTS = {
    WHITE_KNIGHT: 2, WHITE_BISHOP: 2, WHITE_QUEEN: 1,
    BLACK_KNIGHT: 2, BLACK_BISHOP: 2, BLACK_QUEEN: 1,
}


def _get_promotion_types(board: np.ndarray, color_is_white: bool):
    """Return list of piece IDs a pawn of the given color can promote to.

    A pawn can promote to piece type T only if the current count of T on the
    board is less than the initial count (i.e., at least one T was captured).
    """
    if color_is_white:
        candidates = [WHITE_KNIGHT, WHITE_BISHOP, WHITE_QUEEN]
    else:
        candidates = [BLACK_KNIGHT, BLACK_BISHOP, BLACK_QUEEN]
    promos = []
    for pid in candidates:
        current = int(np.count_nonzero(board == pid))
        initial = _INITIAL_COUNTS.get(pid, 1)
        if current < initial:
            promos.append(pid)
    return promos


def get_pawn_moves(board: np.ndarray, row: int, col: int, piece: int):
    """
    Generate pseudo-legal pawn moves.

    Engine convention (after any necessary flip):
        White pawns advance toward higher rows (row+1).
        Black pawns advance toward lower  rows (row-1).
    Captures are diagonal in the forward direction.
    Only one square forward; no double-push, no en passant.
    Promotion: when reaching the last rank, generate one move per available
    promotion type (using the promoted piece's ID).  If no promotions are
    available the pawn cannot advance to the last rank.
    """
    moves = []
    w = is_white(piece)
    if w:
        direction = 1
        enemy = BLACK_PIECES
        promo_row = 5  # rank 6
    else:
        direction = -1
        enemy = WHITE_PIECES
        promo_row = 0  # rank 1

    nr = row + direction

    # Helper: add move(s), handling promotion if target is the promo rank
    def _add(target_row, target_col):
        if target_row == promo_row:
            for promo_id in _get_promotion_types(board, w):
                moves.append((promo_id, row, col, target_row, target_col))
        else:
            moves.append((piece, row, col, target_row, target_col))

    # Forward move
    if in_bounds(nr, col) and board[nr][col] == EMPTY:
        _add(nr, col)

    # Diagonal captures
    for dc in (-1, 1):
        nc = col + dc
        if in_bounds(nr, nc) and int(board[nr][nc]) in enemy:
            _add(nr, nc)

    return moves


def get_knight_moves(board: np.ndarray, row: int, col: int, piece: int):
    moves = []
    offsets = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
               (1, -2),  (1, 2),  (2, -1),  (2, 1)]
    for dr, dc in offsets:
        nr, nc = row + dr, col + dc
        if in_bounds(nr, nc):
            target = int(board[nr][nc])
            if target == EMPTY or not same_side(piece, target):
                moves.append((piece, row, col, nr, nc))
    return moves


def get_sliding_moves(board: np.ndarray, row: int, col: int, piece: int, directions):
    """Generic sliding piece (bishop / queen directions)."""
    moves = []
    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        while in_bounds(nr, nc):
            target = int(board[nr][nc])
            if target == EMPTY:
                moves.append((piece, row, col, nr, nc))
            elif not same_side(piece, target):
                moves.append((piece, row, col, nr, nc))
                break  # captured; can't go further
            else:
                break  # own piece blocks
            nr += dr
            nc += dc
    return moves


def get_bishop_moves(board: np.ndarray, row: int, col: int, piece: int):
    diagonals = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    return get_sliding_moves(board, row, col, piece, diagonals)


def get_queen_moves(board: np.ndarray, row: int, col: int, piece: int):
    all_dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                (0, 1),   (1, -1), (1, 0),  (1, 1)]
    return get_sliding_moves(board, row, col, piece, all_dirs)


def get_king_moves(board: np.ndarray, row: int, col: int, piece: int):
    moves = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if in_bounds(nr, nc):
                target = int(board[nr][nc])
                if target == EMPTY or not same_side(piece, target):
                    moves.append((piece, row, col, nr, nc))
    return moves


MOVE_GENERATORS = {
    WHITE_PAWN:   get_pawn_moves,
    WHITE_KNIGHT: get_knight_moves,
    WHITE_BISHOP: get_bishop_moves,
    WHITE_QUEEN:  get_queen_moves,
    WHITE_KING:   get_king_moves,
    BLACK_PAWN:   get_pawn_moves,
    BLACK_KNIGHT: get_knight_moves,
    BLACK_BISHOP: get_bishop_moves,
    BLACK_QUEEN:  get_queen_moves,
    BLACK_KING:   get_king_moves,
}


def _find_king(board: np.ndarray, white: bool):
    """Return (row, col) of the king for the given side, or None."""
    kid = WHITE_KING if white else BLACK_KING
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if int(board[r][c]) == kid:
                return (r, c)
    return None


def _is_square_attacked(board: np.ndarray, row: int, col: int, by_white: bool) -> bool:
    """
    Return True if the square (row, col) is attacked by any piece of the
    given colour.  Used for king-safety legality checks.
    """
    if by_white:
        epawn, eknight, ebishop, equeen, eking = (
            WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_QUEEN, WHITE_KING)
        pawn_dir = -1   # white pawns attack from one row below the target
    else:
        epawn, eknight, ebishop, equeen, eking = (
            BLACK_PAWN, BLACK_KNIGHT, BLACK_BISHOP, BLACK_QUEEN, BLACK_KING)
        pawn_dir = 1    # black pawns attack from one row above the target

    # Pawn attacks
    pr = row + pawn_dir
    for dc in (-1, 1):
        pc = col + dc
        if 0 <= pr < BOARD_SIZE and 0 <= pc < BOARD_SIZE:
            if int(board[pr][pc]) == epawn:
                return True

    # Knight attacks
    for dr, dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
        nr, nc = row + dr, col + dc
        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
            if int(board[nr][nc]) == eknight:
                return True

    # Sliding: diagonals (bishop + queen)
    for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        nr, nc = row + dr, col + dc
        while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
            p = int(board[nr][nc])
            if p != EMPTY:
                if p == ebishop or p == equeen:
                    return True
                break
            nr += dr; nc += dc

    # Sliding: straights (queen only — no rooks in RoboGambit)
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = row + dr, col + dc
        while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
            p = int(board[nr][nc])
            if p != EMPTY:
                if p == equeen:
                    return True
                break
            nr += dr; nc += dc

    # King adjacency
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                if int(board[nr][nc]) == eking:
                    return True

    return False


def is_in_check(board: np.ndarray, white: bool) -> bool:
    """Return True if the given side's king is in check."""
    kpos = _find_king(board, white)
    if kpos is None:
        return False
    return _is_square_attacked(board, kpos[0], kpos[1], by_white=not white)


def _is_move_legal(board: np.ndarray, move, playing_white: bool) -> bool:
    """
    Check whether a pseudo-legal move is truly legal, i.e. it does not
    leave (or keep) the moving side's king in check.
    """
    piece, sr, sc, tr, tc = move
    after = apply_move(board, piece, sr, sc, tr, tc)
    return not is_in_check(after, playing_white)


def get_pseudo_legal_moves(board: np.ndarray, playing_white: bool):
    """
    Return list of (piece_id, src_row, src_col, dst_row, dst_col) for all
    pseudo-legal moves (does NOT filter king safety).
    """
    moves = []
    our_pieces = WHITE_PIECES if playing_white else BLACK_PIECES
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = int(board[r][c])
            if p in our_pieces:
                gen = MOVE_GENERATORS.get(p)
                if gen:
                    moves.extend(gen(board, r, c, p))
    return moves


def get_all_moves(board: np.ndarray, playing_white: bool):
    """
    Return list of (piece_id, src_row, src_col, dst_row, dst_col) for all
    fully-legal moves (filters out moves that leave the king in check).
    """
    pseudo = get_pseudo_legal_moves(board, playing_white)
    return [m for m in pseudo if _is_move_legal(board, m, playing_white)]


def get_legal_targets(board: np.ndarray, row: int, col: int,
                      piece: int, playing_white: bool):
    """
    Return list of (dst_row, dst_col) that are legal destinations for the
    piece at (row, col).  Used by the GUI for move highlighting.
    """
    gen = MOVE_GENERATORS.get(piece)
    if gen is None:
        return []
    raw = gen(board, row, col, piece)
    return [
        (m[3], m[4])
        for m in raw
        if _is_move_legal(board, m, playing_white)
    ]

# ---------------------------------------------------------------------------
# Board evaluation heuristic  (simple material — used only as fallback)
# ---------------------------------------------------------------------------

def evaluate(board: np.ndarray) -> float:
    """
    Static board evaluation from White's perspective.
    Positive  -> advantage for White
    Negative  -> advantage for Black
    """
    score = 0.0
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = int(board[row][col])
            if piece != EMPTY:
                score += PIECE_VALUES.get(piece, 0)
    return score

# ---------------------------------------------------------------------------
# Apply / undo moves
# ---------------------------------------------------------------------------

def apply_move(board: np.ndarray, piece, src_row, src_col, dst_row, dst_col) -> np.ndarray:
    new_board = board.copy()
    new_board[src_row][src_col] = EMPTY
    new_board[dst_row][dst_col] = piece
    return new_board

# ---------------------------------------------------------------------------
# Format move string
# ---------------------------------------------------------------------------

def format_move(piece: int, src_row: int, src_col: int,
                dst_row: int, dst_col: int, src_piece: int = None) -> str:
    """Return move in format: '<piece_id>:<source>-><target>[=<promoted_id>]'.

    For promotion moves, src_piece is the original pawn ID and piece is the
    promoted piece ID.  The output uses the promoted piece ID as the leading
    ID and appends '=<promoted_id>'.
    """
    src_cell = idx_to_cell(src_row, src_col)
    dst_cell = idx_to_cell(dst_row, dst_col)
    is_promo = (src_piece is not None and src_piece != piece)
    if not is_promo and src_piece is None:
        # Detect promotion: piece is not a pawn but move comes from pawn rank
        # For white: pawn on row 4 promoting to row 5; for black: pawn on row 1 promoting to row 0
        if piece in (WHITE_KNIGHT, WHITE_BISHOP, WHITE_QUEEN) and src_row == 4 and dst_row == 5:
            is_promo = True
        elif piece in (BLACK_KNIGHT, BLACK_BISHOP, BLACK_QUEEN) and src_row == 1 and dst_row == 0:
            is_promo = True
    s = f"{piece}:{src_cell}->{dst_cell}"
    if is_promo:
        s += f"={piece}"
    return s

# ---------------------------------------------------------------------------
# Minimax fallback (used only if C++ engine is unavailable)
# ---------------------------------------------------------------------------

def _minimax(board: np.ndarray, depth: int, alpha: float, beta: float,
             maximising: bool) -> float:
    """Simple alpha-beta minimax for fallback."""
    if depth == 0:
        return evaluate(board)

    moves = get_all_moves(board, playing_white=maximising)
    if not moves:
        return evaluate(board)

    if maximising:
        val = -1e9
        for m in moves:
            child = apply_move(board, *m)
            val = max(val, _minimax(child, depth - 1, alpha, beta, False))
            alpha = max(alpha, val)
            if alpha >= beta:
                break
        return val
    else:
        val = 1e9
        for m in moves:
            child = apply_move(board, *m)
            val = min(val, _minimax(child, depth - 1, alpha, beta, True))
            beta = min(beta, val)
            if alpha >= beta:
                break
        return val


def _fallback_search(board: np.ndarray, playing_white: bool):
    """
    Pure-Python fallback if C++ engine fails.  Very shallow (depth 3)
    but guarantees a legal move is returned.
    """
    moves = get_all_moves(board, playing_white)
    if not moves:
        return None

    best_score = -1e9 if playing_white else 1e9
    best_move = moves[0]

    for m in moves:
        child = apply_move(board, *m)
        score = _minimax(child, 2, -1e9, 1e9, not playing_white)
        if playing_white and score > best_score:
            best_score = score
            best_move = m
        elif not playing_white and score < best_score:
            best_score = score
            best_move = m

    return best_move

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_best_move(board: np.ndarray, playing_white: bool = True
                  ) -> Optional[str]:
    """
    Given the current board state, return the best move string.

    Parameters
    ----------
    board        : 6x6 NumPy array representing the current game state.
    playing_white: True if the engine is playing as White, False for Black.

    Returns
    -------
    Move string in the format '<piece_id>:<src_cell>-><dst_cell>', or
    None if no legal moves are available.
    """
    side_to_move = 0 if playing_white else 1

    # --- Orientation handling ---
    flipped = _needs_flip(board)
    engine_board = _flip_board(board) if flipped else board

    # --- Try C++ engine (strong search) ---
    try:
        move_str = _engine_search(engine_board, side_to_move, DEFAULT_TIME_MS)
        if move_str is None:
            return None  # checkmate or stalemate

        if flipped:
            move_str = _flip_move_string(move_str)
        return move_str

    except Exception as exc:
        print(f"[Game] C++ engine failed ({exc}); using Python fallback")

    # --- Fallback: pure-Python search ---
    best = _fallback_search(board, playing_white)
    if best is None:
        return None
    return format_move(*best)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ---- Test 1: "Flipped" layout (Game.py example — White on high ranks) ----
    board_flipped = np.array([
        [ 7,  8,  9, 10,  8,  7],   # Row 0 / Rank 1 — Black back rank
        [ 6,  6,  6,  6,  6,  6],   # Row 1 / Rank 2 — Black pawns
        [ 0,  0,  0,  0,  0,  0],   # Row 2 / Rank 3
        [ 0,  0,  0,  0,  0,  0],   # Row 3 / Rank 4
        [ 1,  1,  1,  1,  1,  1],   # Row 4 / Rank 5 — White pawns
        [ 2,  3,  4,  5,  3,  2],   # Row 5 / Rank 6 — White back rank
    ], dtype=np.int32)

    print("=== Test 1: Flipped layout (White on high ranks) ===")
    print("Board:\n", board_flipped)
    print("needs_flip:", _needs_flip(board_flipped))
    move = get_best_move(board_flipped, playing_white=True)
    print("Best move for White:", move)
    try:
        nodes, depth, score = _engine_stats()
        print(f"  nodes={nodes}  depth={depth}  score={score}cp")
    except Exception:
        pass

    # ---- Test 2: Standard layout (White on low ranks — engine convention) ----
    board_standard = np.array([
        [ 3,  2,  4,  5,  2,  3],   # Row 0 / Rank 1 — White back rank
        [ 1,  1,  1,  1,  1,  1],   # Row 1 / Rank 2 — White pawns
        [ 0,  0,  0,  0,  0,  0],   # Row 2 / Rank 3
        [ 0,  0,  0,  0,  0,  0],   # Row 3 / Rank 4
        [ 6,  6,  6,  6,  6,  6],   # Row 4 / Rank 5 — Black pawns
        [ 8,  7,  9, 10,  7,  8],   # Row 5 / Rank 6 — Black back rank
    ], dtype=np.int32)

    print("\n=== Test 2: Standard layout (White on low ranks) ===")
    print("Board:\n", board_standard)
    print("needs_flip:", _needs_flip(board_standard))
    move = get_best_move(board_standard, playing_white=True)
    print("Best move for White:", move)
    try:
        nodes, depth, score = _engine_stats()
        print(f"  nodes={nodes}  depth={depth}  score={score}cp")
    except Exception:
        pass

    # ---- Test 3: Black to move ----
    print("\n=== Test 3: Standard layout, Black to move ===")
    move = get_best_move(board_standard, playing_white=False)
    print("Best move for Black:", move)
    try:
        nodes, depth, score = _engine_stats()
        print(f"  nodes={nodes}  depth={depth}  score={score}cp")
    except Exception:
        pass
