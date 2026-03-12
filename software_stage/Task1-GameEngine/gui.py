#!/usr/bin/env python3
"""
RoboGambit 6x6 Chess — GUI Visualiser
======================================
Interactive board viewer that lets you:
  - Watch Engine vs Engine (auto-play)
  - Play as White or Black against the engine
  - Step through moves with arrow keys
  - See live evaluation, search depth, and node counts

Usage:
    python3 gui.py                    # Engine vs Engine auto-play
    python3 gui.py --white            # Human plays White
    python3 gui.py --black            # Human plays Black
    python3 gui.py --time 3000        # Set engine time per move (ms)
    python3 gui.py --flip             # Start with flipped board view

Requires: tkinter (standard library)
"""

import tkinter as tk
from tkinter import messagebox
import numpy as np
import argparse
import threading
import time
import copy
import os
import sys
import subprocess
import random

# Import our engine
from Game import (
    get_best_move, get_all_moves, apply_move, evaluate, engine_eval,
    idx_to_cell, cell_to_idx, format_move, _needs_flip, reset_flip_cache,
    _engine_stats, _ensure_engine, _engine_lib, _engine_initialized,
    _get_promotion_types,
    EMPTY, BOARD_SIZE,
    WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_QUEEN, WHITE_KING,
    BLACK_PAWN, BLACK_KNIGHT, BLACK_BISHOP, BLACK_QUEEN, BLACK_KING,
    WHITE_PIECES, BLACK_PIECES,
    is_white, is_black, same_side,
)

def generate_fischer_random_board():
    """Generate a Fischer Random 6x6 starting position."""
    board = np.zeros((6, 6), dtype=np.int32)
    # Pawns
    for c in range(6):
        board[1][c] = WHITE_PAWN
        board[4][c] = BLACK_PAWN

    # Back ranks
    light_squares = [0, 2, 4]
    dark_squares = [1, 3, 5]
    b1_file = random.choice(light_squares)
    b2_file = random.choice(dark_squares)

    remaining = [f for f in range(6) if f != b1_file and f != b2_file]
    random.shuffle(remaining)

    q_file = remaining[0]
    k_file = remaining[1]
    n1_file = remaining[2]
    n2_file = remaining[3]

    board[0][b1_file] = WHITE_BISHOP
    board[0][b2_file] = WHITE_BISHOP
    board[0][q_file] = WHITE_QUEEN
    board[0][k_file] = WHITE_KING
    board[0][n1_file] = WHITE_KNIGHT
    board[0][n2_file] = WHITE_KNIGHT

    board[5][b1_file] = BLACK_BISHOP
    board[5][b2_file] = BLACK_BISHOP
    board[5][q_file] = BLACK_QUEEN
    board[5][k_file] = BLACK_KING
    board[5][n1_file] = BLACK_KNIGHT
    board[5][n2_file] = BLACK_KNIGHT

    return board


# ---------------------------------------------------------------------------
# Fairy Stockfish Engine Wrapper
# ---------------------------------------------------------------------------

class FairyEngine:
    def __init__(self, binary_path, variant="chess6x6", skill_level=20):
        _popen_kwargs = {}
        if sys.platform == "win32":
            _popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        self.process = subprocess.Popen(
            [binary_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            **_popen_kwargs,
        )
        self._send("uci")
        while True:
            line = self.process.stdout.readline().strip()
            if line == "uciok":
                break
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        variants_file = os.path.join(script_dir, "variants.ini")
        if os.path.exists(variants_file):
            self._send(f"setoption name VariantPath value {os.path.abspath(variants_file)}")
            
        self._send(f"setoption name UCI_Variant value {variant}")
        self._send("setoption name Use NNUE value false")
        self._send(f"setoption name Skill Level value {skill_level}")
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
                self.process = None

    def get_move(self, board, side_to_move, time_limit_ms):
        if not self.process:
            return None
        
        PIECE_TO_CHAR = {
            0: None, 1: 'P', 2: 'N', 3: 'B', 4: 'Q', 5: 'K',
            6: 'p', 7: 'n', 8: 'b', 9: 'q', 10: 'k'
        }
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
        fen_str = f"{fen} {side} - - 0 1"
        
        self._send(f"position fen {fen_str}")
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
            
        from_cell = bestmove[0:2].upper()
        to_cell = bestmove[2:4].upper()
        promo_char = bestmove[4:].lower() if len(bestmove) > 4 else ""
        
        fr, fc = cell_to_idx(from_cell)
        tr, tc = cell_to_idx(to_cell)
        pid = int(board[fr][fc])
        
        is_pawn = (pid == 1 or pid == 6)
        promo_rank = 5 if side_to_move == 0 else 0
        
        if is_pawn and tr == promo_rank:
            if promo_char:
                ptype_map = {'p': 1, 'n': 2, 'b': 3, 'q': 4, 'k': 5}
                ptype = ptype_map.get(promo_char, 0)
                display_id = ptype if side_to_move == 0 else ptype + 5
            else:
                display_id = pid
            return f"{display_id}:{from_cell}->{to_cell}={display_id}"
        else:
            display_id = pid
            
        return f"{display_id}:{from_cell}->{to_cell}"

    def quit(self):
        if self.process:
            self._send("quit")
            self.process.terminate()
            self.process = None


# ---------------------------------------------------------------------------
# Piece rendering (Unicode chess symbols)
# ---------------------------------------------------------------------------

PIECE_UNICODE = {
    WHITE_KING:   "\u2654",  # 
    WHITE_QUEEN:  "\u2655",  # 
    WHITE_BISHOP: "\u2657",  # 
    WHITE_KNIGHT: "\u2658",  # 
    WHITE_PAWN:   "\u2659",  # 
    BLACK_KING:   "\u265A",  # 
    BLACK_QUEEN:  "\u265B",  # 
    BLACK_BISHOP: "\u265D",  # 
    BLACK_KNIGHT: "\u265E",  # 
    BLACK_PAWN:   "\u265F",  # 
}

PIECE_NAME = {
    WHITE_PAWN: "Pawn", WHITE_KNIGHT: "Knight", WHITE_BISHOP: "Bishop",
    WHITE_QUEEN: "Queen", WHITE_KING: "King",
    BLACK_PAWN: "Pawn", BLACK_KNIGHT: "Knight", BLACK_BISHOP: "Bishop",
    BLACK_QUEEN: "Queen", BLACK_KING: "King",
}

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

LIGHT_SQ    = "#F0D9B5"
DARK_SQ     = "#B58863"
HIGHLIGHT   = "#FFFF66"
MOVE_DOT    = "#6BCB77"
LAST_FROM   = "#AAD576"
LAST_TO     = "#D5E8A0"
CHECK_SQ    = "#FF6B6B"
SEL_SQ      = "#7EC8E3"
BG_COLOUR   = "#2C2C2C"
TEXT_COLOUR  = "#E0E0E0"
PANEL_BG    = "#3A3A3A"

# ---------------------------------------------------------------------------
# Board logic helpers
# ---------------------------------------------------------------------------

def find_king(board, white):
    """Return (row, col) of the king, or None."""
    kid = WHITE_KING if white else BLACK_KING
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == kid:
                return (r, c)
    return None


def is_in_check(board, white):
    """Rough check detection: is the king attacked by any opposing piece?"""
    kpos = find_king(board, white)
    if kpos is None:
        return False
    kr, kc = kpos
    enemy_pieces = BLACK_PIECES if white else WHITE_PIECES

    # Check from knights
    for dr, dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
        nr, nc = kr+dr, kc+dc
        if 0 <= nr < 6 and 0 <= nc < 6:
            p = int(board[nr][nc])
            eknight = BLACK_KNIGHT if white else WHITE_KNIGHT
            if p == eknight:
                return True

    # Check from pawns
    pawn_dir = 1 if white else -1  # direction enemy pawns attack FROM
    epawn = BLACK_PAWN if white else WHITE_PAWN
    for dc in (-1, 1):
        nr, nc = kr + pawn_dir, kc + dc
        if 0 <= nr < 6 and 0 <= nc < 6 and int(board[nr][nc]) == epawn:
            return True

    # Check from sliding pieces (bishop/queen diags, queen straights)
    ebishop = BLACK_BISHOP if white else WHITE_BISHOP
    equeen  = BLACK_QUEEN  if white else WHITE_QUEEN
    # Diagonals
    for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        nr, nc = kr+dr, kc+dc
        while 0 <= nr < 6 and 0 <= nc < 6:
            p = int(board[nr][nc])
            if p != EMPTY:
                if p == ebishop or p == equeen:
                    return True
                break
            nr += dr; nc += dc
    # Straights (queen only — no rooks in RoboGambit)
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = kr+dr, kc+dc
        while 0 <= nr < 6 and 0 <= nc < 6:
            p = int(board[nr][nc])
            if p != EMPTY:
                if p == equeen:
                    return True
                break
            nr += dr; nc += dc

    # King adjacency (can't happen in legal play but check anyway)
    eking = BLACK_KING if white else WHITE_KING
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            if dr==0 and dc==0: continue
            nr, nc = kr+dr, kc+dc
            if 0 <= nr < 6 and 0 <= nc < 6 and int(board[nr][nc]) == eking:
                return True

    return False


def game_result(board, white_to_move):
    """
    Return a result string or None if game continues.
    Uses the engine's move gen to check for no legal moves.
    """
    move_str = get_best_move(board.astype(np.int32), playing_white=white_to_move)
    if move_str is None:
        if is_in_check(board, white_to_move):
            return "Black wins by checkmate!" if white_to_move else "White wins by checkmate!"
        else:
            return "Draw by stalemate!"
    return None


# ---------------------------------------------------------------------------
# GUI Application
# ---------------------------------------------------------------------------

class ChessGUI:
    SQ_SIZE = 90
    MARGIN  = 40       # space for rank/file labels
    PANEL_W = 300      # side panel width

    def __init__(self, root, human_side=None, engine_time=5000, start_flipped=False, fairy_skill=None, helper_engine=None, helper_skill=None):
        """
        human_side: None = engine vs engine, 'white', or 'black'
        """
        self.root = root
        self.root.title("RoboGambit 6x6 Chess")
        self.root.configure(bg=BG_COLOUR)
        self.root.resizable(False, False)

        self.human_side = human_side   # None | 'white' | 'black'
        self.engine_time = engine_time
        self.view_flipped = start_flipped
        
        self.fairy_skill = fairy_skill
        self.fairy_engine = None
        self.helper_type = helper_engine
        self.helper_skill = helper_skill
        self.helper_fairy_engine = None
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        fsf_name = "fairy-stockfish.exe" if sys.platform == "win32" else "fairy-stockfish"
        fsf_path = os.path.join(script_dir, fsf_name)
        if not os.path.exists(fsf_path):
            import shutil
            fsf_path = shutil.which(fsf_name) or fsf_path
            
        if self.fairy_skill is not None:
            self.fairy_engine = FairyEngine(fsf_path, skill_level=self.fairy_skill)
            
        if self.helper_type == "fairy":
            # If the main engine is also fairy and has the same skill level, we could reuse it,
            # but for safety and avoiding state corruption during background think vs hint, 
            # we initialize a separate instance for hints.
            skill = self.helper_skill if self.helper_skill is not None else 20
            self.helper_fairy_engine = FairyEngine(fsf_path, skill_level=skill)

        # Game state
        self.board = generate_fischer_random_board()

        self.white_to_move = True
        self.move_history = []       # list of (move_str, board_before)
        self.game_over = False
        self.result_text = ""
        self.halfmove_clock = 0
        self.position_history = {}   # board_bytes -> count for 3-fold repetition
        self.position_history[self.board.tobytes()] = 1  # track starting position

        # Selection state (for human play)
        self.selected = None         # (row, col) of selected piece
        self.legal_targets = []      # list of (row, col) targets for selected piece
        self.last_move = None        # ((fr, fc), (tr, tc))
        self.hint_arrow = None       # ((fr, fc), (tr, tc))

        # Engine thinking state
        self.engine_thinking = False
        self._closing = False          # Set when window close is requested
        self._engine_thread = None     # Reference to the engine background thread

        # Handle window close gracefully (stop engine thread first)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ---- Layout ----
        board_w = self.MARGIN + BOARD_SIZE * self.SQ_SIZE + self.MARGIN
        board_h = self.MARGIN + BOARD_SIZE * self.SQ_SIZE + self.MARGIN + 35
        total_w = board_w + self.PANEL_W

        self.canvas = tk.Canvas(root, width=board_w, height=board_h,
                                bg=BG_COLOUR, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT)
        self.canvas.bind("<Button-1>", self._on_click)

        # Side panel
        self.panel = tk.Frame(root, width=self.PANEL_W, bg=PANEL_BG)
        self.panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.panel.pack_propagate(False)

        self._build_panel()
        self._draw_board()

        # Key bindings
        self.root.bind("<f>", lambda e: self._toggle_flip())
        self.root.bind("<r>", lambda e: self._reset_game())
        self.root.bind("<u>", lambda e: self._undo_move())
        self.root.bind("<h>", lambda e: self._show_hint())
        self.root.bind("<n>", lambda e: self._show_engine_hint())

        # Pre-load engine in background
        threading.Thread(target=_ensure_engine, daemon=True).start()

        # If engine vs engine, start auto-play after a short delay
        if self.human_side is None:
            self.root.after(500, self._auto_play_step)
        elif self.human_side == 'black':
            # Engine plays first as white
            self.root.after(500, self._trigger_engine_move)

    # ----- Side panel -----

    def _build_panel(self):
        pad = dict(padx=10, pady=4, anchor="w")
        font_h  = ("Segoe UI", 14, "bold")
        font_n  = ("Segoe UI", 11)
        font_sm = ("Segoe UI", 9)

        tk.Label(self.panel, text="RoboGambit 6x6", font=("Segoe UI", 16, "bold"),
                 bg=PANEL_BG, fg="#F5C542").pack(pady=(15, 5))

        sep = tk.Frame(self.panel, height=2, bg="#555"); sep.pack(fill=tk.X, padx=10, pady=5)

        # Turn indicator
        self.turn_var = tk.StringVar(value="White to move")
        tk.Label(self.panel, textvariable=self.turn_var, font=font_h,
                 bg=PANEL_BG, fg=TEXT_COLOUR).pack(**pad)

        # Evaluation
        self.eval_var = tk.StringVar(value="Eval: --")
        self.last_engine_eval = None  # cached search score (white perspective)
        tk.Label(self.panel, textvariable=self.eval_var, font=font_n,
                 bg=PANEL_BG, fg=TEXT_COLOUR).pack(**pad)

        # Search info
        self.info_var = tk.StringVar(value="Depth: --  Nodes: --")
        tk.Label(self.panel, textvariable=self.info_var, font=font_sm,
                 bg=PANEL_BG, fg="#AAAAAA").pack(**pad)
                 
        # Hint info
        self.hint_var = tk.StringVar(value="")
        tk.Label(self.panel, textvariable=self.hint_var, font=font_sm,
                 bg=PANEL_BG, fg="#42F5B6").pack(**pad)

        sep2 = tk.Frame(self.panel, height=2, bg="#555"); sep2.pack(fill=tk.X, padx=10, pady=5)

        # Move list
        tk.Label(self.panel, text="Move History", font=font_h,
                 bg=PANEL_BG, fg=TEXT_COLOUR).pack(**pad)

        list_frame = tk.Frame(self.panel, bg=PANEL_BG)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.move_listbox = tk.Listbox(
            list_frame, font=("Consolas", 10), bg="#1E1E1E", fg="#D4D4D4",
            selectbackground="#444", borderwidth=0, highlightthickness=0,
            yscrollcommand=scrollbar.set
        )
        self.move_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.move_listbox.yview)

        # Result
        self.result_var = tk.StringVar(value="")
        tk.Label(self.panel, textvariable=self.result_var, font=("Segoe UI", 12, "bold"),
                 bg=PANEL_BG, fg="#FF6B6B").pack(pady=5)

        sep3 = tk.Frame(self.panel, height=2, bg="#555"); sep3.pack(fill=tk.X, padx=10, pady=5)

        # Buttons
        btn_frame = tk.Frame(self.panel, bg=PANEL_BG)
        btn_frame.pack(pady=10)
        btn_style = dict(font=font_sm, width=10, relief=tk.FLAT, cursor="hand2")

        tk.Button(btn_frame, text="Flip (F)", command=self._toggle_flip,
                  bg="#555", fg="white", **btn_style).grid(row=0, column=0, padx=3, pady=3)
        tk.Button(btn_frame, text="Reset (R)", command=self._reset_game,
                  bg="#555", fg="white", **btn_style).grid(row=0, column=1, padx=3, pady=3)
        tk.Button(btn_frame, text="Undo (U)", command=self._undo_move,
                  bg="#555", fg="white", **btn_style).grid(row=1, column=0, padx=3, pady=3)
        tk.Button(btn_frame, text="Hint (H)", command=self._show_hint,
                  bg="#555", fg="white", **btn_style).grid(row=1, column=1, padx=3, pady=3)
        tk.Button(btn_frame, text="Engine Hint (N)", command=self._show_engine_hint,
                  bg="#3A6B35", fg="white", **btn_style).grid(row=2, column=0, columnspan=2, padx=3, pady=3)

        # Mode label
        mode = "Engine vs Engine" if self.human_side is None else f"Human ({self.human_side}) vs Engine"
        tk.Label(self.panel, text=f"Mode: {mode}", font=font_sm,
                 bg=PANEL_BG, fg="#888").pack(side=tk.BOTTOM, pady=8)

    # ----- Drawing -----

    def _sq_coords(self, row, col):
        """Return canvas (x1, y1, x2, y2) for a board square."""
        # Flipped = 180° rotation: both rank and file reversed
        if self.view_flipped:
            dr = row
            dc = BOARD_SIZE - 1 - col
        else:
            dr = BOARD_SIZE - 1 - row
            dc = col
        x1 = self.MARGIN + dc * self.SQ_SIZE
        y1 = self.MARGIN + dr * self.SQ_SIZE
        return x1, y1, x1 + self.SQ_SIZE, y1 + self.SQ_SIZE

    def _draw_board(self):
        self.canvas.delete("all")
        S = self.SQ_SIZE
        M = self.MARGIN

        # Squares
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                x1, y1, x2, y2 = self._sq_coords(r, c)
                light = (r + c) % 2 == 0
                colour = LIGHT_SQ if light else DARK_SQ

                # Highlight last move
                if self.last_move:
                    if (r, c) == self.last_move[0]:
                        colour = LAST_FROM
                    elif (r, c) == self.last_move[1]:
                        colour = LAST_TO

                # Highlight selected square
                if self.selected and (r, c) == self.selected:
                    colour = SEL_SQ

                # King in check
                if not self.game_over:
                    kw = find_king(self.board, self.white_to_move)
                    if kw and (r, c) == kw and is_in_check(self.board, self.white_to_move):
                        colour = CHECK_SQ

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=colour, outline="")

                # Legal move dots
                if (r, c) in self.legal_targets:
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    p = int(self.board[r][c])
                    if p != EMPTY:
                        # Capture: ring around square
                        self.canvas.create_oval(
                            x1+4, y1+4, x2-4, y2-4,
                            outline=MOVE_DOT, width=3)
                    else:
                        # Quiet: small dot
                        rad = 8
                        self.canvas.create_oval(
                            cx-rad, cy-rad, cx+rad, cy+rad,
                            fill=MOVE_DOT, outline="")

                # Piece
                piece = int(self.board[r][c])
                if piece != EMPTY:
                    sym = PIECE_UNICODE.get(piece, "?")
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    self.canvas.create_text(
                        cx, cy, text=sym,
                        font=("Segoe UI Symbol", int(S * 0.55)),
                        fill="black" if is_white(piece) else "#1A1A1A")

        # File labels (A-F)
        for c in range(BOARD_SIZE):
            dc = (BOARD_SIZE - 1 - c) if self.view_flipped else c
            x = M + c * S + S / 2
            lbl = chr(ord('A') + dc)
            self.canvas.create_text(x, M + BOARD_SIZE * S + 15, text=lbl,
                                    font=("Segoe UI", 11, "bold"), fill=TEXT_COLOUR)
            self.canvas.create_text(x, M - 15, text=lbl,
                                    font=("Segoe UI", 11, "bold"), fill=TEXT_COLOUR)

        # Rank labels (1-6)
        for r in range(BOARD_SIZE):
            if self.view_flipped:
                rank_num = r + 1
                dy = r
            else:
                rank_num = BOARD_SIZE - r
                dy = r
            y = M + dy * S + S / 2
            self.canvas.create_text(M - 15, y, text=str(rank_num),
                                    font=("Segoe UI", 11, "bold"), fill=TEXT_COLOUR)
            self.canvas.create_text(M + BOARD_SIZE * S + 15, y, text=str(rank_num),
                                    font=("Segoe UI", 11, "bold"), fill=TEXT_COLOUR)

        # Hint arrow
        if getattr(self, 'hint_arrow', None):
            (sr, sc), (tr, tc) = self.hint_arrow
            sx1, sy1, sx2, sy2 = self._sq_coords(sr, sc)
            tx1, ty1, tx2, ty2 = self._sq_coords(tr, tc)
            cx1, cy1 = (sx1 + sx2) / 2, (sy1 + sy2) / 2
            cx2, cy2 = (tx1 + tx2) / 2, (ty1 + ty2) / 2
            # Draw a thick, translucent-looking arrow
            self.canvas.create_line(cx1, cy1, cx2, cy2, fill="#FF8C00", width=8, arrow=tk.LAST, arrowshape=(16, 20, 6))

        # Status bar at bottom
        status = "Thinking ..." if self.engine_thinking else ""
        if self.game_over:
            status = self.result_text
        self.canvas.create_text(
            M + (BOARD_SIZE * S) / 2, M + BOARD_SIZE * S + 30,
            text=status, font=("Segoe UI", 10), fill="#FFAA00")

    # ----- User interaction -----

    def _board_pos_from_pixel(self, px, py):
        """Convert pixel coords to (row, col) or None."""
        M, S = self.MARGIN, self.SQ_SIZE
        gc = int((px - M) / S)
        gr = int((py - M) / S)
        if gc < 0 or gc >= BOARD_SIZE or gr < 0 or gr >= BOARD_SIZE:
            return None
        if self.view_flipped:
            return (gr, BOARD_SIZE - 1 - gc)
        else:
            return (BOARD_SIZE - 1 - gr, gc)

    def _show_promotion_dialog(self, promo_types):
        """Show a modal dialog letting the human pick a promotion piece.

        promo_types: list of piece IDs the pawn can promote to (from captured pool).
        Returns the chosen piece ID, or None if the dialog is cancelled.
        """
        chosen = [None]  # mutable container so the nested function can write to it

        dlg = tk.Toplevel(self.root)
        dlg.title("Promote pawn")
        dlg.configure(bg=BG_COLOUR)
        dlg.resizable(False, False)
        dlg.wait_visibility()   # ensure window is mapped before grabbing
        dlg.grab_set()          # modal
        dlg.transient(self.root)

        tk.Label(
            dlg, text="Choose promotion piece:",
            font=("Segoe UI", 13, "bold"), bg=BG_COLOUR, fg=TEXT_COLOUR
        ).pack(padx=20, pady=(15, 8))

        btn_frame = tk.Frame(dlg, bg=BG_COLOUR)
        btn_frame.pack(padx=20, pady=(0, 15))

        for pid in promo_types:
            sym = PIECE_UNICODE.get(pid, "?")
            name = PIECE_NAME.get(pid, "?")

            def _pick(p=pid):
                chosen[0] = p
                dlg.destroy()

            tk.Button(
                btn_frame, text=f" {sym} {name} ",
                font=("Segoe UI Symbol", 18),
                bg="#555", fg="white", activebackground="#777",
                relief=tk.FLAT, cursor="hand2",
                command=_pick,
            ).pack(side=tk.LEFT, padx=6, pady=4)

        # Centre the dialog over the main window
        dlg.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width()  - dlg.winfo_width())  // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dlg.winfo_height()) // 2
        dlg.geometry(f"+{x}+{y}")

        # If the user closes the dialog without picking, treat as cancel
        dlg.protocol("WM_DELETE_WINDOW", dlg.destroy)
        self.root.wait_window(dlg)

        return chosen[0]

    def _on_click(self, event):
        if self.game_over or self.engine_thinking:
            return
        # In engine-vs-engine mode, space/auto handles moves
        if self.human_side is None:
            return

        pos = self._board_pos_from_pixel(event.x, event.y)
        if pos is None:
            self.selected = None
            self.legal_targets = []
            self._draw_board()
            return

        r, c = pos
        piece = int(self.board[r][c])

        # Is this the human's turn?
        human_is_white = (self.human_side == 'white')
        if human_is_white != self.white_to_move:
            return  # Not human's turn

        our_pieces = WHITE_PIECES if human_is_white else BLACK_PIECES

        # If clicking a legal target, make the move
        if self.selected and (r, c) in self.legal_targets:
            sr, sc = self.selected
            sp = int(self.board[sr][sc])

            # Check if this is a pawn promotion move
            is_pawn = (sp == WHITE_PAWN or sp == BLACK_PAWN)
            promo_rank = 5 if human_is_white else 0
            if is_pawn and r == promo_rank:
                promo_types = _get_promotion_types(self.board, human_is_white)
                if not promo_types:
                    # No pieces in the captured pool — shouldn't happen
                    # because legal move gen already filters this out,
                    # but guard defensively.
                    self.selected = None
                    self.legal_targets = []
                    self._draw_board()
                    return
                if len(promo_types) == 1:
                    # Only one choice — auto-promote
                    sp = promo_types[0]
                else:
                    chosen = self._show_promotion_dialog(promo_types)
                    if chosen is None:
                        # User cancelled — don't make the move
                        return
                    sp = chosen

            self._make_move(sp, sr, sc, r, c)
            self.selected = None
            self.legal_targets = []
            self._draw_board()
            return

        # If clicking own piece, select it
        if piece in our_pieces:
            self.selected = (r, c)
            # Get legal targets (filters out moves that leave own king in check)
            from Game import get_legal_targets
            self.legal_targets = get_legal_targets(
                self.board, r, c, piece, playing_white=human_is_white
            )
            self._draw_board()
        else:
            self.selected = None
            self.legal_targets = []
            self._draw_board()

    def _make_move(self, piece, sr, sc, tr, tc):
        """Apply a move, update state, record history."""
        source_piece = int(self.board[sr][sc])
        move_str = format_move(piece, sr, sc, tr, tc, src_piece=source_piece)
        self.move_history.append((move_str, self.board.copy()))

        # Update halfmove clock: reset on capture or pawn move (including promotion)
        is_capture = self.board[tr][tc] != 0
        is_pawn = (source_piece == WHITE_PAWN or source_piece == BLACK_PAWN)
        if is_capture or is_pawn:
            self.halfmove_clock = 0
            self.position_history.clear()
        else:
            self.halfmove_clock += 1

        self.board = apply_move(self.board, piece, sr, sc, tr, tc)
        self.last_move = ((sr, sc), (tr, tc))
        self.white_to_move = not self.white_to_move

        # Track position for 3-fold repetition
        board_key = self.board.tobytes()
        self.position_history[board_key] = self.position_history.get(board_key, 0) + 1
        
        self.hint_arrow = None
        self.hint_var.set("")

        # Update UI
        n = len(self.move_history)
        side = "W" if not self.white_to_move else "B"  # side that just moved
        self.move_listbox.insert(tk.END, f"{n:>3}. [{side}] {move_str}")
        self.move_listbox.see(tk.END)

        self.turn_var.set("White to move" if self.white_to_move else "Black to move")
        if self.last_engine_eval is not None:
            self.eval_var.set(f"Eval: {self.last_engine_eval:+d} cp")
        else:
            try:
                ev = engine_eval(self.board, self.white_to_move)
                self.eval_var.set(f"Eval: {ev:+d} cp")
            except Exception:
                ev = evaluate(self.board)
                self.eval_var.set(f"Eval: {ev:+.0f} cp (material)")

        self._draw_board()

        # Check for game end
        self._check_game_over()

        # Trigger engine response if it's the engine's turn
        if not self.game_over and self.human_side is not None:
            human_is_white = (self.human_side == 'white')
            if human_is_white != self.white_to_move:
                # Engine's turn
                self.root.after(200, self._trigger_engine_move)

    def _check_game_over(self):
        """Check if game is over (checkmate, stalemate, 3-fold repetition, 50-move rule)."""
        moves = get_all_moves(self.board, self.white_to_move)
        if not moves:
            self.game_over = True
            if is_in_check(self.board, self.white_to_move):
                self.result_text = "Black wins!" if self.white_to_move else "White wins!"
            else:
                self.result_text = "Stalemate — Draw!"
            self.result_var.set(self.result_text)
            self._draw_board()
            return True

        # 3-fold repetition
        board_key = self.board.tobytes()
        if self.position_history.get(board_key, 0) >= 3:
            self.game_over = True
            self.result_text = "Draw by 3-fold repetition"
            self.result_var.set(self.result_text)
            self._draw_board()
            return True

        # 50-move rule (100 halfmoves)
        if self.halfmove_clock >= 100:
            self.game_over = True
            self.result_text = "Draw by 50-move rule"
            self.result_var.set(self.result_text)
            self._draw_board()
            return True

        return False

    def _undo_move(self):
        if self.engine_thinking or self._closing or not self.move_history:
            return
        
        # Decrement position history for current board
        board_key = self.board.tobytes()
        if board_key in self.position_history:
            self.position_history[board_key] -= 1
            if self.position_history[board_key] <= 0:
                del self.position_history[board_key]

        # Pop last move and revert state
        _, prev_board = self.move_history.pop()
        self.board = prev_board
        self.white_to_move = not self.white_to_move
        self.game_over = False
        self.result_text = ""
        self.result_var.set("")

        # Rebuild halfmove_clock from history (simplest correct approach)
        self.halfmove_clock = 0
        for i in range(len(self.move_history) - 1, -1, -1):
            move_str, board_before = self.move_history[i]
            pid = int(move_str.split(":")[0])
            cells = move_str.split(":")[1].split("->")
            tr, tc = cell_to_idx(cells[1])
            is_capture = board_before[tr][tc] != 0
            is_pawn = (pid == WHITE_PAWN or pid == BLACK_PAWN)
            if is_capture or is_pawn:
                break
            self.halfmove_clock += 1
        
        # Remove from UI listbox
        self.move_listbox.delete(tk.END)
        
        # Clear highlights & hint
        self.selected = None
        self.legal_targets = []
        self.last_move = None
        self.hint_arrow = None
        self.hint_var.set("")
        
        # Update labels
        self.turn_var.set("White to move" if self.white_to_move else "Black to move")
        if self.last_engine_eval is not None:
            self.eval_var.set(f"Eval: {self.last_engine_eval:+d} cp")
        else:
            try:
                ev = engine_eval(self.board, self.white_to_move)
                self.eval_var.set(f"Eval: {ev:+d} cp")
            except Exception:
                ev = evaluate(self.board)
                self.eval_var.set(f"Eval: {ev:+.0f} cp (material)")
            
        self._draw_board()
        
        # In Human vs Engine, undoing once reverts the engine's move and it becomes human's turn. 
        # But if the human wants to undo *their* move, they'd have to undo twice (once for engine, once for human).
        # To simplify, we'll undo exactly one move. If the human undos when it's their turn, it undos the engine's move.
        # If the human undos when it's the engine's turn (engine was thinking), it will cancel, but we check `engine_thinking` above.

    def _apply_hint(self, move_str):
        parts = move_str.split(":")
        if len(parts) == 2:
            cells = parts[1].split("->")
            sr, sc = cell_to_idx(cells[0])
            tr, tc = cell_to_idx(cells[1])
            self.hint_arrow = ((sr, sc), (tr, tc))

            # Check if the engine's move is a promotion
            hint_text = parts[1]
            try:
                piece_id = int(parts[0])
                src_piece = int(self.board[sr][sc])
                if src_piece in (WHITE_PAWN, BLACK_PAWN) and piece_id != src_piece:
                    sym = PIECE_UNICODE.get(piece_id, "")
                    name = PIECE_NAME.get(piece_id, "")
                    hint_text += f" (={sym}{name})"
            except (ValueError, IndexError):
                pass

            self.hint_var.set(f"Hint: {hint_text}")
            self._draw_board()
        else:
            self.hint_var.set("Hint: No moves")

    def _show_hint(self):
        if self.game_over or self.engine_thinking or self._closing:
            return
            
        self.hint_var.set("Hint: Thinking...")
        self.root.update_idletasks()
        
        def hint_worker():
            try:
                board_copy = self.board.copy().astype(np.int32)
                
                if getattr(self, 'helper_type', None) == 'fairy' and getattr(self, 'helper_fairy_engine', None) is not None:
                    side_to_move = 0 if self.white_to_move else 1
                    move_str = self.helper_fairy_engine.get_move(board_copy, side_to_move, self.engine_time)
                else:
                    move_str = get_best_move(board_copy, playing_white=self.white_to_move)
                
                if self._closing:
                    return
                    
                if move_str:
                    self.root.after(0, lambda: self._apply_hint(move_str))
                else:
                    self.root.after(0, lambda: self.hint_var.set("Hint: No moves"))
            except Exception:
                pass
                
        threading.Thread(target=hint_worker, daemon=True).start()

    def _show_engine_hint(self):
        """Show hint from the local NNUE engine (always uses libchess6x6.so)."""
        if self.game_over or self.engine_thinking or self._closing:
            return

        self.hint_var.set("Engine Hint: Thinking...")
        self.root.update_idletasks()

        def engine_hint_worker():
            try:
                board_copy = self.board.copy().astype(np.int32)
                move_str = get_best_move(board_copy, playing_white=self.white_to_move)

                if self._closing:
                    return

                if move_str:
                    self.root.after(0, lambda: self._apply_hint(move_str))
                else:
                    self.root.after(0, lambda: self.hint_var.set("Engine Hint: No moves"))
            except Exception:
                if not self._closing:
                    self.root.after(0, lambda: self.hint_var.set("Engine Hint: Error"))

        threading.Thread(target=engine_hint_worker, daemon=True).start()

    def _trigger_engine_move(self):
        """Ask the engine to compute and play a move."""
        if self.game_over or self.engine_thinking or self._closing:
            return
        self.engine_thinking = True
        self._draw_board()
        # Run engine in background thread
        t = threading.Thread(target=self._engine_think, daemon=True)
        self._engine_thread = t
        t.start()

    def _engine_think(self):
        """Run in background thread."""
        try:
            board_copy = self.board.copy().astype(np.int32)
            
            if getattr(self, 'fairy_engine', None) is not None:
                side_to_move = 0 if self.white_to_move else 1
                move_str = self.fairy_engine.get_move(board_copy, side_to_move, self.engine_time)
                stats = (self.fairy_engine.last_nodes, self.fairy_engine.last_depth, self.fairy_engine.last_score)
            else:
                move_str = get_best_move(board_copy, playing_white=self.white_to_move)

                # Get stats
                try:
                    nodes, depth, score = _engine_stats()
                    stats = (nodes, depth, score)
                except Exception:
                    stats = None

            # If closing was requested while we were searching, don't touch the UI
            if self._closing:
                return

            # Schedule UI update on main thread
            self.root.after(0, lambda: self._engine_move_done(move_str, stats))
        except Exception as e:
            if self._closing:
                return
            self.root.after(0, lambda: self._engine_move_failed(str(e)))

    def _engine_move_done(self, move_str, stats):
        self.engine_thinking = False

        if move_str is None:
            self.game_over = True
            if is_in_check(self.board, self.white_to_move):
                self.result_text = "Black wins!" if self.white_to_move else "White wins!"
            else:
                self.result_text = "Stalemate — Draw!"
            self.result_var.set(self.result_text)
            self._draw_board()
            return

        # Update stats display
        if stats:
            nodes, depth, score = stats
            self.info_var.set(f"Depth: {depth}  Nodes: {nodes:,}")
            side_mult = 1 if self.white_to_move else -1
            white_score = score * side_mult
            self.last_engine_eval = white_score
            self.eval_var.set(f"Eval: {white_score:+d} cp")

        # Parse move string: "<id>:<source>-><target>[=<promoted_id>]"
        try:
            parts = move_str.split(":")
            piece_id = int(parts[0])
            target_part = parts[1].split("->")[1]
            # Strip promotion suffix if present (e.g. "A6=4" -> "A6")
            if "=" in target_part:
                target_part = target_part.split("=")[0]
            cells = parts[1].split("->")
            sr, sc = cell_to_idx(cells[0])
            tr, tc = cell_to_idx(target_part)
        except (ValueError, IndexError):
            self._engine_move_failed(f"Bad move string: {move_str}")
            return

        self._make_move(piece_id, sr, sc, tr, tc)

    def _engine_move_failed(self, msg):
        self.engine_thinking = False
        self._draw_board()
        print(f"[GUI] Engine error: {msg}")

    # ----- Auto-play (engine vs engine) -----

    def _auto_play_step(self):
        """One step of engine-vs-engine play."""
        if self.game_over or self._closing:
            return
        self._trigger_engine_move()
        # Schedule next check (wait for thinking to finish)
        self._wait_and_continue()

    def _wait_and_continue(self):
        """Poll until engine finishes, then schedule next move."""
        if self._closing:
            return
        if self.engine_thinking:
            self.root.after(100, self._wait_and_continue)
        elif not self.game_over:
            self.root.after(300, self._auto_play_step)

    # ----- Controls -----

    def _on_close(self):
        """Handle window close: stop engine search, wait for thread, then exit."""
        self._closing = True
        self.game_over = True  # prevent new moves from being scheduled

        if getattr(self, 'fairy_engine', None) is not None:
            self.fairy_engine.quit()
            
        if getattr(self, 'helper_fairy_engine', None) is not None:
            self.helper_fairy_engine.quit()

        # Signal the C++ engine to abort its search immediately
        try:
            if _engine_initialized and _engine_lib:
                _engine_lib.engine_stop_search()
        except Exception:
            pass

        # Wait for the engine thread to finish (it checks info.stopped every 4096 nodes)
        if self._engine_thread is not None and self._engine_thread.is_alive():
            self._engine_thread.join(timeout=2.0)

        self.root.destroy()

        # Hard exit to prevent C++ destructor ordering issues during
        # Python's normal shutdown when the shared library is unloaded.
        os._exit(0)

    def _toggle_flip(self):
        self.view_flipped = not self.view_flipped
        self._draw_board()

    def _reset_game(self):
        reset_flip_cache()   # re-detect board orientation for the new game
        if getattr(self, 'fairy_engine', None) is not None:
            self.fairy_engine._send("ucinewgame")
        if getattr(self, 'helper_fairy_engine', None) is not None:
            self.helper_fairy_engine._send("ucinewgame")
        self.board = generate_fischer_random_board()
        self.white_to_move = True
        self.move_history = []
        self.game_over = False
        self.result_text = ""
        self.halfmove_clock = 0
        self.position_history = {}
        self.position_history[self.board.tobytes()] = 1
        self.selected = None
        self.legal_targets = []
        self.last_move = None
        self.hint_arrow = None
        self.engine_thinking = False

        self.turn_var.set("White to move")
        self.eval_var.set("Eval: --")
        self.info_var.set("Depth: --  Nodes: --")
        self.hint_var.set("")
        self.result_var.set("")
        self.move_listbox.delete(0, tk.END)
        self._draw_board()

        if self.human_side is None:
            self.root.after(500, self._auto_play_step)
        elif self.human_side == 'black':
            self.root.after(500, self._trigger_engine_move)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RoboGambit 6x6 Chess GUI")
    parser.add_argument("--white", action="store_true", help="Play as White vs engine")
    parser.add_argument("--black", action="store_true", help="Play as Black vs engine")
    parser.add_argument("--time", type=int, default=None, help="Engine time per move (ms)")
    parser.add_argument("--flip", action="store_true", help="Start with flipped board view")
    parser.add_argument("--fairy", action="store_true", help="Play against Fairy Stockfish")
    parser.add_argument("--skill", type=int, default=20, help="Fairy Stockfish Skill Level (-20 to 20)")
    parser.add_argument("--helper", type=str, choices=["local", "fairy"], default="local", help="Engine to use for hints (local or fairy)")
    parser.add_argument("--helper-skill", type=int, default=20, help="Skill level if helper is fairy (-20 to 20)")
    args = parser.parse_args()

    human = None
    if args.white:
        human = "white"
    elif args.black:
        human = "black"

    # Update engine time in Game module (only if explicitly set)
    import Game
    engine_time = args.time if args.time is not None else Game.DEFAULT_TIME_MS
    Game.DEFAULT_TIME_MS = engine_time

    fairy_skill = args.skill if args.fairy else None

    root = tk.Tk()
    app = ChessGUI(root, human_side=human, engine_time=engine_time,
                   start_flipped=args.flip, fairy_skill=fairy_skill,
                   helper_engine=args.helper, helper_skill=args.helper_skill)
    root.mainloop()


if __name__ == "__main__":
    main()
