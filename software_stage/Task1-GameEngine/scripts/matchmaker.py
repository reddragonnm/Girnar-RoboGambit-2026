#!/usr/bin/env python3
"""
Matchmaker: A/B testing system for 6x6 chess engine versions.

Workflow:
  1. Tune/change eval params
  2. Build: make lib
  3. Snapshot: python3 scripts/matchmaker.py snapshot <name>
  4. Match:   python3 scripts/matchmaker.py match <name_a> <name_b> --games 100

Snapshots are stored in snapshots/ directory as .so files.

Examples:
  # Save current engine as a snapshot
  python3 scripts/matchmaker.py snapshot baseline

  # After tuning, save new version
  make lib
  python3 scripts/matchmaker.py snapshot tuned_v2

  # Compare them (100 games, 200ms/move)
  python3 scripts/matchmaker.py match baseline tuned_v2 --games 100 --time 200

  # List all snapshots
  python3 scripts/matchmaker.py list
"""

import argparse
import ctypes
import math
import numpy as np
import os
import random
import shutil
import sys
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SNAPSHOT_DIR = os.path.join(PROJECT_DIR, "snapshots")
LIB_NAME = "libchess6x6.so"

# Reuse the independent movegen and Fischer Random generator from validate_engine
sys.path.insert(0, SCRIPT_DIR)
from validate_engine import (
    generate_fischer_random, generate_legal_moves, make_move_on_board,
    move_to_str, validate_move, in_check, print_board, piece_type, piece_color,
    sq_to_cell, cell_to_sq, parse_move_str
)


import subprocess

# ============================================================
# FEN conversion
# ============================================================

PIECE_TO_CHAR = {
    0: None,
    1: 'P', 2: 'N', 3: 'B', 4: 'Q', 5: 'K',
    6: 'p', 7: 'n', 8: 'b', 9: 'q', 10: 'k'
}

def board_to_fen(board, side_to_move):
    """Convert 6x6 NumPy board to FEN string."""
    fen_rows = []
    for r in range(5, -1, -1):
        row_str = ""
        empty_count = 0
        for c in range(6):
            pid = board[r][c]
            char = PIECE_TO_CHAR.get(pid)
            if char is None:
                empty_count += 1
            else:
                if empty_count > 0:
                    row_str += str(empty_count)
                    empty_count = 0
                row_str += char
        if empty_count > 0:
            row_str += str(empty_count)
        fen_rows.append(row_str)
    
    fen = "/".join(fen_rows)
    side = 'w' if side_to_move == 0 else 'b'
    # No castling, no en passant, 0 halfmove, 1 fullmove
    return f"{fen} {side} - - 0 1"


# ============================================================
# Engine wrapper (loads a specific .so file)
# ============================================================

class EngineInstance:
    """Wrapper around a specific .so build of the engine."""
    # ... (existing code) ...

    def __init__(self, lib_path, name="engine"):
        self.name = name
        self.lib_path = lib_path

        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Library not found: {lib_path}")

        self.lib = ctypes.CDLL(lib_path)

        # Function signatures
        self.lib.engine_init.restype = None
        self.lib.engine_init.argtypes = []

        self.lib.engine_get_move.restype = ctypes.c_int
        self.lib.engine_get_move.argtypes = [
            ctypes.POINTER(ctypes.c_int),   # board_arr
            ctypes.c_int,                    # side_to_move
            ctypes.c_int,                    # time_limit_ms
            ctypes.POINTER(ctypes.c_int),    # captured_white
            ctypes.POINTER(ctypes.c_int),    # captured_black
            ctypes.c_char_p,                 # result_buf
            ctypes.c_int                     # buf_size
        ]

        self.lib.engine_get_nodes.restype = ctypes.c_int
        self.lib.engine_get_depth.restype = ctypes.c_int
        self.lib.engine_get_score.restype = ctypes.c_int

        self.lib.engine_cleanup.restype = None
        self.lib.engine_cleanup.argtypes = []

        self.lib.engine_clear_game_history.restype = None
        self.lib.engine_clear_game_history.argtypes = []

        self.lib.engine_get_hash.restype = None
        self.lib.engine_get_hash.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
        
        self.lib.engine_set_game_history.restype = None
        self.lib.engine_set_game_history.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_int]

        # Initialize engine
        self.lib.engine_init()

    def get_move(self, board, side_to_move, time_limit_ms):
        """Get best move from the engine."""
        flat = board.flatten().astype(np.int32)
        board_ptr = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        buf = ctypes.create_string_buffer(64)

        result = self.lib.engine_get_move(
            board_ptr, side_to_move, time_limit_ms,
            None, None, buf, 64
        )

        if result < 0:
            return None

        return buf.value.decode('ascii')

    def clear_history(self):
        self.lib.engine_clear_game_history()

    def get_hash(self):
        out = (ctypes.c_uint32 * 2)()
        self.lib.engine_get_hash(out)
        return (out[1] << 32) | out[0]
        
    def set_game_history(self, hashes):
        if not hashes:
            self.lib.engine_set_game_history(None, 0)
        else:
            arr = (ctypes.c_uint64 * len(hashes))(*hashes)
            self.lib.engine_set_game_history(arr, len(hashes))

    @property
    def nodes(self):
        return self.lib.engine_get_nodes()

    @property
    def depth(self):
        return self.lib.engine_get_depth()

    @property
    def score(self):
        return self.lib.engine_get_score()

    def cleanup(self):
        self.lib.engine_cleanup()


class UCIEngineInstance:
    """Wrapper around a UCI engine binary (like fairy-stockfish)."""

    def __init__(self, binary_path, name="uci_engine", variant="chess6x6", skill_level=None):
        self.name = name
        self.binary_path = binary_path
        self.variant = variant
        self.skill_level = skill_level
        
        # Start engine
        self.process = subprocess.Popen(
            [binary_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Initialize UCI
        self._send("uci")
        while True:
            line = self.process.stdout.readline().strip()
            if line == "uciok":
                break
        
        # Load variant configuration
        variants_file = os.path.join(PROJECT_DIR, "variants.ini")
        if os.path.exists(variants_file):
            self._send(f"setoption name VariantPath value {os.path.abspath(variants_file)}")
        
        self._send(f"setoption name UCI_Variant value {variant}")
        self._send("setoption name Use NNUE value false")
        
        if self.skill_level is not None:
            self._send(f"setoption name Skill Level value {self.skill_level}")
            
        self._send("isready")
        while True:
            line = self.process.stdout.readline().strip()
            if line == "readyok":
                break
        
        self.last_score = 0
        self.last_depth = 0
        self.last_nodes = 0

    def _send(self, cmd):
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(cmd + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError):
                # Process likely crashed
                self.process = None

    def get_move(self, board, side_to_move, time_limit_ms):
        if not self.process:
            return None
        fen = board_to_fen(board, side_to_move)
        self._send(f"position fen {fen}")
        self._send(f"go movetime {time_limit_ms}")
        
        bestmove = None
        while True:
            line = self.process.stdout.readline().strip()
            if not line:
                break
            if line.startswith("bestmove"):
                bestmove = line.split()[1]
                break
            if line.startswith("info"):
                # Parse metrics
                parts = line.split()
                try:
                    if "depth" in parts:
                        self.last_depth = int(parts[parts.index("depth") + 1])
                    if "nodes" in parts:
                        self.last_nodes = int(parts[parts.index("nodes") + 1])
                    if "score" in parts:
                        if "cp" in parts:
                            self.last_score = int(parts[parts.index("cp") + 1])
                        elif "mate" in parts:
                            m = int(parts[parts.index("mate") + 1])
                            self.last_score = 10000 if m > 0 else -10000
                except:
                    pass
        
        if not bestmove or bestmove == "(none)":
            return None
            
        return self._convert_uci_to_comp(board, side_to_move, bestmove)

    def _convert_uci_to_comp(self, board, side_to_move, uci_move):
        # E.g., 'b2b3' or 'b5b6q'
        from_cell = uci_move[0:2].upper()
        to_cell = uci_move[2:4].upper()
        promo_char = uci_move[4:].lower() if len(uci_move) > 4 else ""
        
        # cell_to_sq returns (row, col)
        fr, fc = cell_to_sq(from_cell)
        tr, tc = cell_to_sq(to_cell)
        pid = int(board[fr][fc])
        
        # If the move is a pawn move to the last rank, it's a promotion
        is_pawn = (pid == 1 or pid == 6)
        promo_rank = 5 if side_to_move == 0 else 0
        
        if is_pawn and tr == promo_rank:
            if promo_char:
                # Map char to local piece ID
                ptype_map = {'p': 1, 'n': 2, 'b': 3, 'q': 4, 'k': 5}
                ptype = ptype_map.get(promo_char, 0)
                display_id = ptype if side_to_move == 0 else ptype + 5
            else:
                # If no promo_char but it's a promotion rank, it's an illegal move for our validator
                # unless we pick a default. But FS should provide it.
                display_id = pid 
        else:
            display_id = pid
            
        return f"{display_id}:{from_cell}->{to_cell}"

    def clear_history(self):
        if not self.process:
            return
        self._send("ucinewgame")
        self._send("isready")
        while True:
            line = self.process.stdout.readline().strip()
            if line == "readyok":
                break

    def set_game_history(self, hashes):
        # UCI doesn't support setting Zobrist hashes directly
        pass

    def get_hash(self):
        # We can't easily get the specific Zobrist hash used by local engine from UCI
        # This will disable repetition detection in the matchmaker for UCI engines
        # but we handle repetition in the game loop anyway.
        return 0

    @property
    def nodes(self): return self.last_nodes
    @property
    def depth(self): return self.last_depth
    @property
    def score(self): return self.last_score

    def cleanup(self):
        if self.process:
            self._send("quit")
            self.process.terminate()
            self.process = None


# ============================================================
# Snapshot management
# ============================================================

def ensure_snapshot_dir():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def snapshot_path(name):
    return os.path.join(SNAPSHOT_DIR, f"{name}.so")


def cmd_snapshot(name):
    """Save current libchess6x6.so as a named snapshot."""
    ensure_snapshot_dir()
    src = os.path.join(PROJECT_DIR, LIB_NAME)
    if not os.path.exists(src):
        print(f"ERROR: {LIB_NAME} not found. Run 'make lib' first.")
        return False

    dst = snapshot_path(name)
    if os.path.exists(dst):
        print(f"WARNING: Snapshot '{name}' already exists, overwriting.")

    shutil.copy2(src, dst)
    size = os.path.getsize(dst)
    print(f"Snapshot '{name}' saved ({size:,} bytes)")
    print(f"  Path: {dst}")
    return True


def cmd_list():
    """List all saved snapshots."""
    ensure_snapshot_dir()
    snapshots = sorted([f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.so')])
    if not snapshots:
        print("No snapshots found. Use 'snapshot <name>' to create one.")
        return

    print(f"{'Name':<25} {'Size':>10} {'Date':<20}")
    print("-" * 58)
    for f in snapshots:
        path = os.path.join(SNAPSHOT_DIR, f)
        name = f[:-3]  # remove .so
        size = os.path.getsize(path)
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        print(f"{name:<25} {size:>10,} {mtime.strftime('%Y-%m-%d %H:%M'):<20}")


# ============================================================
# Statistical analysis
# ============================================================

def elo_diff(wins, losses, draws):
    """Estimate Elo difference from match results."""
    total = wins + losses + draws
    if total == 0:
        return 0.0, float('inf')

    score = (wins + draws * 0.5) / total

    # Clamp to avoid log(0)
    score = max(0.001, min(0.999, score))

    elo = -400 * math.log10(1.0 / score - 1.0)

    # Standard error estimation (trinomial)
    # Var(score) = (w(1-s)^2 + l*s^2 + d(0.5-s)^2) / (n*(n-1))
    w_frac = wins / total
    l_frac = losses / total
    d_frac = draws / total
    var = (w_frac * (1 - score) ** 2 + l_frac * score ** 2 +
           d_frac * (0.5 - score) ** 2) / total

    if var <= 0:
        return elo, float('inf')

    se = math.sqrt(var)
    # Convert score SE to Elo SE
    elo_se = se / (score * (1 - score) * math.log(10) / 400)

    return elo, elo_se


def los(wins, losses):
    """Likelihood of Superiority — probability that engine A is stronger than B."""
    total = wins + losses
    if total == 0:
        return 0.5

    # Using the normal approximation to the binomial
    # H0: p = 0.5 (equally strong), we want P(p > 0.5)
    p = wins / total if total > 0 else 0.5
    if total == 0:
        return 0.5

    # Wald statistic
    se = math.sqrt(0.25 / total)  # SE under H0: p=0.5
    z = (p - 0.5) / se

    # Normal CDF approximation
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def sprt_llr(wins, losses, draws, elo0=0, elo1=5):
    """
    Sequential Probability Ratio Test — Log-Likelihood Ratio.
    H0: Elo diff = elo0, H1: Elo diff = elo1.
    Returns LLR value. Compare against bounds:
      lower = ln(beta / (1-alpha))
      upper = ln((1-beta) / alpha)
    For alpha=beta=0.05: lower ~ -2.94, upper ~ 2.94
    """
    total = wins + losses + draws
    if total == 0:
        return 0.0

    score = (wins + draws * 0.5) / total
    score = max(0.001, min(0.999, score))

    # Expected scores under H0 and H1
    s0 = 1.0 / (1.0 + 10.0 ** (-elo0 / 400.0))
    s1 = 1.0 / (1.0 + 10.0 ** (-elo1 / 400.0))

    # LLR = sum of log-likelihood ratios
    # Approximate with the trinomial model
    w = wins / total
    l = losses / total
    d = draws / total

    # Under each hypothesis, expected w/d/l fractions
    # For a trinomial with score s, we approximate:
    # Using draws ratio from actual data
    draw_ratio = d if d > 0 else 0.01

    def get_wdl(s, d_ratio):
        w = s - d_ratio / 2
        l = 1 - s - d_ratio / 2
        w = max(0.001, w)
        l = max(0.001, l)
        return w, d_ratio, l

    w0, d0, l0 = get_wdl(s0, draw_ratio)
    w1, d1, l1 = get_wdl(s1, draw_ratio)

    # LLR per game
    llr = 0.0
    if w > 0:
        llr += wins * math.log(w1 / w0)
    if d > 0:
        llr += draws * math.log(d1 / d0) if d1 != d0 else 0
    if l > 0:
        llr += losses * math.log(l1 / l0)

    return llr


# ============================================================
# Match runner (head-to-head)
# ============================================================

def play_game(engine_w, engine_b, board, time_w, time_b, max_moves=300, verbose=False):
    """
    Play a single game between two engines.
    Returns: ('W', 'B', or 'D'), move_count, list_of_errors
    """
    engines = {'W': engine_w, 'B': engine_b}
    times = {'W': time_w, 'B': time_b}
    current_color = 'W'
    move_count = 0
    halfmove_clock = 0
    position_history = {}
    game_history_hashes = []
    errors = []

    while move_count < max_moves:
        legal_moves = generate_legal_moves(board, current_color)

        if len(legal_moves) == 0:
            if in_check(board, current_color):
                winner = 'B' if current_color == 'W' else 'W'
                return winner, move_count, errors
            else:
                return 'D', move_count, errors

        if halfmove_clock >= 100:
            return 'D', move_count, errors

        board_key = board.tobytes()
        position_history[board_key] = position_history.get(board_key, 0) + 1
        if position_history[board_key] >= 3:
            return 'D', move_count, errors

        move_count += 1
        side = 0 if current_color == 'W' else 1
        eng = engines[current_color]
        time_ms = times[current_color]

        eng.set_game_history(game_history_hashes)

        try:
            move_str = eng.get_move(board, side, time_ms)
            board_hash = eng.get_hash()
        except Exception as e:
            errors.append(f"{eng.name} ({current_color}) exception at move {move_count}: {e}")
            # Engine that crashed loses
            return ('B' if current_color == 'W' else 'W'), move_count, errors

        if move_str is None:
            errors.append(f"{eng.name} ({current_color}) returned None at move {move_count}")
            return ('B' if current_color == 'W' else 'W'), move_count, errors

        is_valid, matched_move, err = validate_move(board, current_color, move_str, legal_moves)

        if not is_valid:
            errors.append(
                f"{eng.name} ({current_color}) illegal move at move {move_count}: "
                f"{move_str} — {err}"
            )
            # Illegal move = loss
            return ('B' if current_color == 'W' else 'W'), move_count, errors

        if verbose and move_count <= 10:
            print(f"  {move_count}. [{eng.name}/{current_color}] {move_str}")

        game_history_hashes.append(board_hash)

        # Apply move
        is_capture = board[matched_move[2][0]][matched_move[2][1]] != 0
        is_pawn = piece_type(matched_move[0]) == 1
        board = make_move_on_board(board, matched_move)
        
        if is_capture or is_pawn:
            halfmove_clock = 0
            game_history_hashes = []
            position_history.clear()
        else:
            halfmove_clock += 1
            
        current_color = 'B' if current_color == 'W' else 'W'

    return 'D', move_count, errors


def cmd_match(name_a, name_b, num_games=100, time_ms=200, 
              time_a=None, time_b=None, skill_a=None, skill_b=None,
              verbose=False, sprt_elo0=0, sprt_elo1=5):
    """Run a head-to-head match between two engine snapshots or binaries."""

    # Resolve library paths — 'current' means the working libchess6x6.so
    def resolve_path(name):
        if name == 'current':
            p = os.path.join(PROJECT_DIR, LIB_NAME)
        elif name == 'fairy':
            # Support calling fairy-stockfish by name
            p = os.path.join(PROJECT_DIR, "fairy-stockfish")
            if not os.path.exists(p):
                p = shutil.which("fairy-stockfish")
            return p
        else:
            p = snapshot_path(name)
        if not os.path.exists(p) and name != 'fairy':
            print(f"ERROR: Engine '{name}' not found at {p}")
            sys.exit(1)
        return p

    path_a = resolve_path(name_a)
    path_b = resolve_path(name_b)

    # Use specific times if provided, otherwise fallback to global time_ms
    eff_time_a = time_a if time_a is not None else time_ms
    eff_time_b = time_b if time_b is not None else time_ms

    print(f"Match: {name_a} vs {name_b}")
    print(f"  {name_a}: {path_a} (Time: {eff_time_a}ms, Skill: {skill_a})")
    print(f"  {name_b}: {path_b} (Time: {eff_time_b}ms, Skill: {skill_b})")
    print(f"  Games: {num_games}")
    print(f"  SPRT bounds: H0=Elo{sprt_elo0}, H1=Elo{sprt_elo1}")
    print()

    # Load engines
    if name_a == 'fairy':
        eng_a = UCIEngineInstance(path_a, name=name_a, skill_level=skill_a)
    else:
        eng_a = EngineInstance(path_a, name=name_a)
        
    if name_b == 'fairy':
        eng_b = UCIEngineInstance(path_b, name=name_b, skill_level=skill_b)
    else:
        eng_b = EngineInstance(path_b, name=name_b)

    # Stats from perspective of engine A
    wins_a = 0
    losses_a = 0
    draws = 0
    total_moves = 0
    all_errors = []

    # SPRT bounds (alpha=beta=0.05)
    sprt_lower = math.log(0.05 / 0.95)   # ~ -2.944
    sprt_upper = math.log(0.95 / 0.05)   # ~ +2.944

    start_time = time.time()

    for game_idx in range(num_games):
        if game_idx % 2 == 0:
            base_board = generate_fischer_random()
        board = base_board.copy()

        # Alternate colors: even games A=white, odd games A=black
        if game_idx % 2 == 0:
            engine_w, engine_b = eng_a, eng_b
            time_w, time_b = eff_time_a, eff_time_b
            a_is_white = True
        else:
            engine_w, engine_b = eng_b, eng_a
            time_w, time_b = eff_time_b, eff_time_a
            a_is_white = False

        # Reset engines for a new game
        for eng in [engine_w, engine_b]:
            if isinstance(eng, UCIEngineInstance):
                # For UCI engines, we need to ensure they are alive and clear their state
                if eng.process is None:
                    # Re-instantiate if crashed
                    eng.__init__(eng.binary_path, name=eng.name, variant=eng.variant, skill_level=eng.skill_level)
                else:
                    eng.clear_history()
            else:
                # Local engine snapshots need cleanup and re-init to reset TT/state
                eng.cleanup()
                eng.lib.engine_init()
                eng.clear_history()

        result, moves, errors = play_game(
            engine_w, engine_b, board, time_w, time_b, verbose=verbose
        )
        total_moves += moves
        all_errors.extend(errors)

        # Translate result to A's perspective
        if result == 'D':
            draws += 1
            result_str = "Draw"
        elif (result == 'W' and a_is_white) or (result == 'B' and not a_is_white):
            wins_a += 1
            result_str = f"{name_a} wins"
        else:
            losses_a += 1
            result_str = f"{name_b} wins"

        # Progress report every game pair (2 games)
        games_done = game_idx + 1
        if games_done % 2 == 0 or games_done == num_games:
            elo, elo_se = elo_diff(wins_a, losses_a, draws)
            score_pct = (wins_a + draws * 0.5) / games_done * 100

            print(
                f"  [{games_done:>4}/{num_games}] "
                f"+{wins_a} -{losses_a} ={draws}  "
                f"Score: {score_pct:.1f}%  "
                f"Elo: {elo:+.1f} +/- {elo_se:.1f}  "
                f"({moves}mv, {result_str})"
            )

        # SPRT check every game pair
        if games_done >= 10 and games_done % 2 == 0:
            llr = sprt_llr(wins_a, losses_a, draws, sprt_elo0, sprt_elo1)
            if llr >= sprt_upper:
                print(f"\n  SPRT: H1 accepted (LLR={llr:.3f} >= {sprt_upper:.3f})")
                print(f"  Conclusion: {name_a} is likely stronger than {name_b} by ~Elo{sprt_elo1}")
                break
            elif llr <= sprt_lower:
                print(f"\n  SPRT: H0 accepted (LLR={llr:.3f} <= {sprt_lower:.3f})")
                print(f"  Conclusion: {name_a} is NOT stronger than {name_b}")
                break

    elapsed = time.time() - start_time
    games_done = wins_a + losses_a + draws

    # Final report
    print(f"\n{'='*60}")
    print(f"MATCH RESULTS: {name_a} vs {name_b}")
    print(f"{'='*60}")
    print(f"Games:       {games_done}")
    print(f"  {name_a} wins: {wins_a}")
    print(f"  {name_b} wins: {losses_a}")
    print(f"  Draws:     {draws}")

    if games_done > 0:
        score_pct = (wins_a + draws * 0.5) / games_done * 100
        elo, elo_se = elo_diff(wins_a, losses_a, draws)
        l = los(wins_a, losses_a)
        avg_moves = total_moves / games_done

        print(f"\nScore:       {score_pct:.1f}% (from {name_a}'s perspective)")
        print(f"Elo diff:    {elo:+.1f} +/- {elo_se:.1f} (95% CI: [{elo - 1.96*elo_se:+.1f}, {elo + 1.96*elo_se:+.1f}])")
        print(f"LOS:         {l*100:.1f}% ({name_a} is stronger)")
        print(f"Avg moves:   {avg_moves:.1f}/game")
        print(f"Time:        {elapsed:.1f}s ({elapsed/games_done:.2f}s/game)")

        llr = sprt_llr(wins_a, losses_a, draws, sprt_elo0, sprt_elo1)
        print(f"SPRT LLR:    {llr:.3f} (bounds: [{sprt_lower:.3f}, {sprt_upper:.3f}])")

    if all_errors:
        print(f"\nERRORS ({len(all_errors)}):")
        for e in all_errors[:20]:
            print(f"  {e}")
    else:
        print(f"\nNo errors detected.")

    # Cleanup
    eng_a.cleanup()
    eng_b.cleanup()


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Matchmaker: A/B testing for 6x6 chess engine versions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/matchmaker.py snapshot baseline
  python3 scripts/matchmaker.py list
  python3 scripts/matchmaker.py match baseline current --games 100
  python3 scripts/matchmaker.py match current fairy --games 200 --time 300 --skill-b 0 --time-b 10
        """
    )
    sub = parser.add_subparsers(dest='command')

    # snapshot
    p_snap = sub.add_parser('snapshot', help='Save current engine as a named snapshot')
    p_snap.add_argument('name', help='Snapshot name (e.g., "baseline", "tuned_v2")')

    # list
    sub.add_parser('list', help='List all saved snapshots')

    # match
    p_match = sub.add_parser('match', help='Play a match between two engine versions')
    p_match.add_argument('engine_a', help='First engine (snapshot name or "current")')
    p_match.add_argument('engine_b', help='Second engine (snapshot name or "current")')
    p_match.add_argument('--games', type=int, default=100, help='Number of games (default: 100)')
    p_match.add_argument('--time', type=int, default=200, help='Time per move in ms (default: 200)')
    p_match.add_argument('--time-a', type=int, help='Override time for engine A')
    p_match.add_argument('--time-b', type=int, help='Override time for engine B')
    p_match.add_argument('--skill-a', type=int, help='Skill level for engine A (if UCI)')
    p_match.add_argument('--skill-b', type=int, help='Skill level for engine B (if UCI)')
    p_match.add_argument('--verbose', '-v', action='store_true', help='Show first 10 moves of each game')
    p_match.add_argument('--elo0', type=int, default=0, help='SPRT null hypothesis Elo (default: 0)')
    p_match.add_argument('--elo1', type=int, default=5, help='SPRT alternative hypothesis Elo (default: 5)')

    args = parser.parse_args()

    if args.command == 'snapshot':
        cmd_snapshot(args.name)
    elif args.command == 'list':
        cmd_list()
    elif args.command == 'match':
        cmd_match(
            args.engine_a, args.engine_b,
            num_games=args.games, time_ms=args.time,
            time_a=args.time_a, time_b=args.time_b,
            skill_a=args.skill_a, skill_b=args.skill_b,
            verbose=args.verbose,
            sprt_elo0=args.elo0, sprt_elo1=args.elo1
        )
    else:
        parser.print_help()
