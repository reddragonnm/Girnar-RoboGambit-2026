#!/usr/bin/env python3
"""
RoboGambit 6x6 Chess — Matchmaker GUI
======================================
Visual engine-vs-engine tool. Load two engine snapshots (or "current" / "fairy")
and step through a game move-by-move with dedicated buttons for each engine.

Usage:
    python3 scripts/matchmaker_gui.py <engine_a> <engine_b> [options]

Examples:
    python3 scripts/matchmaker_gui.py current baseline
    python3 scripts/matchmaker_gui.py current fairy --skill-b 10
    python3 scripts/matchmaker_gui.py tuned_v2 baseline --time 1000
    python3 scripts/matchmaker_gui.py current fairy --time-a 500 --time-b 100 --skill-b 0

Requires: tkinter (standard library)
"""

import tkinter as tk
from tkinter import messagebox
import numpy as np
import argparse
import threading
import os
import sys
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SNAPSHOT_DIR = os.path.join(PROJECT_DIR, "snapshots")
LIB_NAME = "libchess6x6.so"

sys.path.insert(0, SCRIPT_DIR)
from validate_engine import (
    generate_fischer_random, generate_legal_moves, make_move_on_board,
    move_to_str, validate_move, in_check, piece_type, piece_color,
    sq_to_cell, cell_to_sq, parse_move_str,
)

# Reuse engine wrappers & FEN conversion from the CLI matchmaker
from matchmaker import (
    EngineInstance, UCIEngineInstance, board_to_fen,
    elo_diff, los, sprt_llr,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOARD_SIZE = 6
EMPTY = 0
WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_QUEEN, WHITE_KING = 1, 2, 3, 4, 5
BLACK_PAWN, BLACK_KNIGHT, BLACK_BISHOP, BLACK_QUEEN, BLACK_KING = 6, 7, 8, 9, 10
WHITE_PIECES = {1, 2, 3, 4, 5}
BLACK_PIECES = {6, 7, 8, 9, 10}

PIECE_UNICODE = {
    WHITE_KING: "\u2654", WHITE_QUEEN: "\u2655", WHITE_BISHOP: "\u2657",
    WHITE_KNIGHT: "\u2658", WHITE_PAWN: "\u2659",
    BLACK_KING: "\u265A", BLACK_QUEEN: "\u265B", BLACK_BISHOP: "\u265D",
    BLACK_KNIGHT: "\u265E", BLACK_PAWN: "\u265F",
}

PIECE_NAME = {
    WHITE_PAWN: "Pawn", WHITE_KNIGHT: "Knight", WHITE_BISHOP: "Bishop",
    WHITE_QUEEN: "Queen", WHITE_KING: "King",
    BLACK_PAWN: "Pawn", BLACK_KNIGHT: "Knight", BLACK_BISHOP: "Bishop",
    BLACK_QUEEN: "Queen", BLACK_KING: "King",
}

# Colour palette
LIGHT_SQ   = "#F0D9B5"
DARK_SQ    = "#B58863"
HIGHLIGHT  = "#FFFF66"
LAST_FROM  = "#AAD576"
LAST_TO    = "#D5E8A0"
CHECK_SQ   = "#FF6B6B"
BG_COLOUR  = "#2C2C2C"
TEXT_COLOUR = "#E0E0E0"
PANEL_BG   = "#3A3A3A"

ENGINE_A_COLOUR = "#4FC3F7"   # light blue accent
ENGINE_B_COLOUR = "#FF8A65"   # light orange accent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_white(pid):
    return 1 <= pid <= 5

def is_black(pid):
    return 6 <= pid <= 10

def find_king(board, white):
    kid = WHITE_KING if white else BLACK_KING
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == kid:
                return (r, c)
    return None


def is_in_check(board, white):
    kpos = find_king(board, white)
    if kpos is None:
        return False
    kr, kc = kpos
    # Knights
    for dr, dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
        nr, nc = kr+dr, kc+dc
        if 0 <= nr < 6 and 0 <= nc < 6:
            eknight = BLACK_KNIGHT if white else WHITE_KNIGHT
            if int(board[nr][nc]) == eknight:
                return True
    # Pawns
    pdir = 1 if white else -1
    epawn = BLACK_PAWN if white else WHITE_PAWN
    for dc in (-1, 1):
        nr, nc = kr + pdir, kc + dc
        if 0 <= nr < 6 and 0 <= nc < 6 and int(board[nr][nc]) == epawn:
            return True
    # Diagonals (bishop / queen)
    ebishop = BLACK_BISHOP if white else WHITE_BISHOP
    equeen  = BLACK_QUEEN  if white else WHITE_QUEEN
    for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        nr, nc = kr+dr, kc+dc
        while 0 <= nr < 6 and 0 <= nc < 6:
            p = int(board[nr][nc])
            if p != EMPTY:
                if p == ebishop or p == equeen:
                    return True
                break
            nr += dr; nc += dc
    # Straights (queen only)
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = kr+dr, kc+dc
        while 0 <= nr < 6 and 0 <= nc < 6:
            p = int(board[nr][nc])
            if p != EMPTY:
                if p == equeen:
                    return True
                break
            nr += dr; nc += dc
    # King adjacency
    eking = BLACK_KING if white else WHITE_KING
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = kr+dr, kc+dc
            if 0 <= nr < 6 and 0 <= nc < 6 and int(board[nr][nc]) == eking:
                return True
    return False


def resolve_engine_path(name):
    if name == "current":
        return os.path.join(PROJECT_DIR, LIB_NAME)
    elif name == "fairy":
        import shutil
        p = os.path.join(PROJECT_DIR, "fairy-stockfish")
        if not os.path.exists(p):
            p = shutil.which("fairy-stockfish") or p
        return p
    else:
        return os.path.join(SNAPSHOT_DIR, f"{name}.so")


def load_engine(name, skill=None):
    path = resolve_engine_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Engine '{name}' not found at {path}")
    if name == "fairy":
        return UCIEngineInstance(path, name=name, skill_level=skill)
    else:
        eng = EngineInstance(path, name=name)
        print("[ENGINE] PST ONLY")
        return eng


# ---------------------------------------------------------------------------
# GUI Application
# ---------------------------------------------------------------------------

class MatchmakerGUI:
    SQ_SIZE = 90
    MARGIN  = 40
    PANEL_W = 360

    def __init__(self, root, eng_a, eng_b, name_a, name_b,
                 time_a=200, time_b=200):
        self.root = root
        self.root.title(f"Matchmaker — {name_a} vs {name_b}")
        self.root.configure(bg=BG_COLOUR)
        self.root.resizable(True, True)
        self.root.minsize(900, 700)

        self.eng_a = eng_a
        self.eng_b = eng_b
        self.name_a = name_a
        self.name_b = name_b
        self.time_a = time_a
        self.time_b = time_b
        self.view_flipped = False

        # Match stats
        self.wins_a = 0
        self.wins_b = 0
        self.draws  = 0

        self._closing = False
        self._engine_thread = None
        self.engine_thinking = False

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.geometry("1020x760")

        # --- Layout ---
        board_w = self.MARGIN + BOARD_SIZE * self.SQ_SIZE + self.MARGIN
        board_h = self.MARGIN + BOARD_SIZE * self.SQ_SIZE + self.MARGIN + 35
        total_w = board_w + self.PANEL_W

        self.canvas = tk.Canvas(root, width=board_w, height=board_h,
                                bg=BG_COLOUR, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT)

        self.panel = tk.Frame(root, width=self.PANEL_W, bg=PANEL_BG)
        self.panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.panel.pack_propagate(False)

        self._build_panel()
        self._engines_fresh = True
        self._new_game()

        # Key bindings
        self.root.bind("<f>", lambda e: self._toggle_flip())
        self.root.bind("<a>", lambda e: self._step_engine_a())
        self.root.bind("<b>", lambda e: self._step_engine_b())
        self.root.bind("<space>", lambda e: self._step_current())
        self.root.bind("<r>", lambda e: self._new_game())
        self.root.bind("<p>", lambda e: self._toggle_autoplay())

    # -----------------------------------------------------------------
    # Panel
    # -----------------------------------------------------------------

    def _build_panel(self):
        pad  = dict(padx=10, pady=3, anchor="w")
        font_h  = ("Segoe UI", 14, "bold")
        font_n  = ("Segoe UI", 11)
        font_sm = ("Segoe UI", 9)

        tk.Label(self.panel, text="Matchmaker", font=("Segoe UI", 16, "bold"),
                 bg=PANEL_BG, fg="#F5C542").pack(pady=(12, 2))

        sep = tk.Frame(self.panel, height=2, bg="#555"); sep.pack(fill=tk.X, padx=10, pady=5)

        # Engine labels + per-engine eval
        self.eng_a_label = tk.Label(
            self.panel,
            text=f"Engine A (White): {self.name_a}",
            font=font_n, bg=PANEL_BG, fg=ENGINE_A_COLOUR,
        )
        self.eng_a_label.pack(**pad)

        self.eval_a_var = tk.StringVar(value="Eval: --")
        tk.Label(self.panel, textvariable=self.eval_a_var, font=font_sm,
                 bg=PANEL_BG, fg=ENGINE_A_COLOUR).pack(**pad)

        self.eng_b_label = tk.Label(
            self.panel,
            text=f"Engine B (Black): {self.name_b}",
            font=font_n, bg=PANEL_BG, fg=ENGINE_B_COLOUR,
        )
        self.eng_b_label.pack(**pad)

        self.eval_b_var = tk.StringVar(value="Eval: --")
        tk.Label(self.panel, textvariable=self.eval_b_var, font=font_sm,
                 bg=PANEL_BG, fg=ENGINE_B_COLOUR).pack(**pad)

        # Turn / status
        self.turn_var = tk.StringVar(value="White to move")
        tk.Label(self.panel, textvariable=self.turn_var, font=font_h,
                 bg=PANEL_BG, fg=TEXT_COLOUR).pack(**pad)

        # 50-move counter
        self.fifty_var = tk.StringVar(value="50-move: 0 / 100")
        self.fifty_label = tk.Label(self.panel, textvariable=self.fifty_var, font=font_sm,
                 bg=PANEL_BG, fg="#AAAAAA")
        self.fifty_label.pack(**pad)

        # Last-move search stats (depth + nodes)
        self.info_var = tk.StringVar(value="Depth: --  Nodes: --")
        tk.Label(self.panel, textvariable=self.info_var, font=font_sm,
                 bg=PANEL_BG, fg="#AAAAAA").pack(**pad)

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
            yscrollcommand=scrollbar.set,
        )
        self.move_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.move_listbox.yview)

        # Result
        self.result_var = tk.StringVar(value="")
        tk.Label(self.panel, textvariable=self.result_var, font=("Segoe UI", 12, "bold"),
                 bg=PANEL_BG, fg="#FF6B6B").pack(pady=3)

        # Match score
        self.score_var = tk.StringVar(value="Score: 0 - 0 - 0")
        tk.Label(self.panel, textvariable=self.score_var, font=font_n,
                 bg=PANEL_BG, fg="#F5C542").pack(pady=2)

        sep3 = tk.Frame(self.panel, height=2, bg="#555"); sep3.pack(fill=tk.X, padx=10, pady=5)

        # Buttons
        btn_frame = tk.Frame(self.panel, bg=PANEL_BG)
        btn_frame.pack(pady=6)
        btn_style = dict(font=("Segoe UI", 10), width=16, relief=tk.FLAT, cursor="hand2")

        self.btn_a = tk.Button(
            btn_frame, text=f"▶ {self.name_a} move (A)",
            bg="#1565C0", fg="white", activebackground="#1976D2",
            command=self._step_engine_a, **btn_style,
        )
        self.btn_a.grid(row=0, column=0, padx=4, pady=3)

        self.btn_b = tk.Button(
            btn_frame, text=f"▶ {self.name_b} move (B)",
            bg="#D84315", fg="white", activebackground="#E64A19",
            command=self._step_engine_b, **btn_style,
        )
        self.btn_b.grid(row=0, column=1, padx=4, pady=3)

        self.btn_step = tk.Button(
            btn_frame, text="▶ Next move (Space)",
            bg="#2E7D32", fg="white", activebackground="#388E3C",
            command=self._step_current, **btn_style,
        )
        self.btn_step.grid(row=1, column=0, padx=4, pady=3)

        self.btn_auto = tk.Button(
            btn_frame, text="⏩ Auto-play (P)",
            bg="#6A1B9A", fg="white", activebackground="#7B1FA2",
            command=self._toggle_autoplay, **btn_style,
        )
        self.btn_auto.grid(row=1, column=1, padx=4, pady=3)

        btn_frame2 = tk.Frame(self.panel, bg=PANEL_BG)
        btn_frame2.pack(pady=3)
        btn_style2 = dict(font=("Segoe UI", 9), width=12, relief=tk.FLAT, cursor="hand2")

        tk.Button(btn_frame2, text="Flip (F)", command=self._toggle_flip,
                  bg="#555", fg="white", **btn_style2).grid(row=0, column=0, padx=3, pady=2)
        tk.Button(btn_frame2, text="New Game (R)", command=self._new_game,
                  bg="#555", fg="white", **btn_style2).grid(row=0, column=1, padx=3, pady=2)

        # Mode / info
        self.auto_playing = False
        self.autoplay_var = tk.StringVar(value="")
        tk.Label(self.panel, textvariable=self.autoplay_var, font=font_sm,
                 bg=PANEL_BG, fg="#CE93D8").pack(side=tk.BOTTOM, pady=2)

        mode_text = (f"A={self.name_a} ({self.time_a}ms) | "
                     f"B={self.name_b} ({self.time_b}ms)")
        tk.Label(self.panel, text=mode_text, font=font_sm,
                 bg=PANEL_BG, fg="#888").pack(side=tk.BOTTOM, pady=4)

    # -----------------------------------------------------------------
    # Game state
    # -----------------------------------------------------------------

    def _new_game(self):
        if self.engine_thinking:
            return

        self.auto_playing = False
        self.autoplay_var.set("")

        self.board = generate_fischer_random()
        self.white_to_move = True
        self.move_history = []
        self.game_over = False
        self.result_text = ""
        self.halfmove_clock = 0
        self.position_history = {}
        self.position_history[self.board.tobytes()] = 1
        self.last_move = None
        self.game_history_hashes = []

        # Engine A is always white, Engine B is always black for this game
        self.current_color = "W"

        # Reset engine state (skip on first game — engines are already freshly loaded)
        if self._engines_fresh:
            self._engines_fresh = False
            for eng in (self.eng_a, self.eng_b):
                eng.clear_history()
        else:
            for eng in (self.eng_a, self.eng_b):
                if isinstance(eng, UCIEngineInstance):
                    eng.clear_history()
                else:
                    eng.cleanup()
                    eng.lib.engine_init()
                    eng.clear_history()

        # Update colour assignment labels
        self.eng_a_label.config(text=f"Engine A (White): {self.name_a}")
        self.eng_b_label.config(text=f"Engine B (Black): {self.name_b}")

        self.turn_var.set("White to move")
        self.eval_a_var.set("Eval: --")
        self.eval_b_var.set("Eval: --")
        self.fifty_var.set("50-move: 0 / 100")
        self.info_var.set("Depth: --  Nodes: --")
        self.result_var.set("")
        self.move_listbox.delete(0, tk.END)

        self._update_buttons()
        self._draw_board()

    # -----------------------------------------------------------------
    # Drawing
    # -----------------------------------------------------------------

    def _sq_coords(self, row, col):
        if self.view_flipped:
            dr = row
            dc = col
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

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                x1, y1, x2, y2 = self._sq_coords(r, c)
                light = (r + c) % 2 == 0
                colour = LIGHT_SQ if light else DARK_SQ

                # Last move highlight
                if self.last_move:
                    if (r, c) == self.last_move[0]:
                        colour = LAST_FROM
                    elif (r, c) == self.last_move[1]:
                        colour = LAST_TO

                # King in check
                if not self.game_over:
                    kw = find_king(self.board, self.white_to_move)
                    if kw and (r, c) == kw and is_in_check(self.board, self.white_to_move):
                        colour = CHECK_SQ

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=colour, outline="")

                # Piece
                piece = int(self.board[r][c])
                if piece != EMPTY:
                    sym = PIECE_UNICODE.get(piece, "?")
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    self.canvas.create_text(
                        cx, cy, text=sym,
                        font=("Segoe UI Symbol", int(S * 0.55)),
                        fill="black" if is_white(piece) else "#1A1A1A")

        # File labels
        for c in range(BOARD_SIZE):
            x = M + c * S + S / 2
            lbl = chr(ord('A') + c)
            self.canvas.create_text(x, M + BOARD_SIZE * S + 15, text=lbl,
                                    font=("Segoe UI", 11, "bold"), fill=TEXT_COLOUR)
            self.canvas.create_text(x, M - 15, text=lbl,
                                    font=("Segoe UI", 11, "bold"), fill=TEXT_COLOUR)

        # Rank labels
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

        # Status bar
        status = "Thinking ..." if self.engine_thinking else ""
        if self.game_over:
            status = self.result_text
        self.canvas.create_text(
            M + (BOARD_SIZE * S) / 2, M + BOARD_SIZE * S + 30,
            text=status, font=("Segoe UI", 10), fill="#FFAA00")

    # -----------------------------------------------------------------
    # Button state management
    # -----------------------------------------------------------------

    def _update_buttons(self):
        if self.game_over or self.engine_thinking:
            self.btn_a.config(state=tk.DISABLED)
            self.btn_b.config(state=tk.DISABLED)
            self.btn_step.config(state=tk.DISABLED)
        else:
            # Enable the button whose turn it is, disable the other
            if self.current_color == "W":
                self.btn_a.config(state=tk.NORMAL)
                self.btn_b.config(state=tk.DISABLED)
            else:
                self.btn_a.config(state=tk.DISABLED)
                self.btn_b.config(state=tk.NORMAL)
            self.btn_step.config(state=tk.NORMAL)

    # -----------------------------------------------------------------
    # Engine stepping
    # -----------------------------------------------------------------

    def _step_engine_a(self):
        if self.game_over or self.engine_thinking or self.current_color != "W":
            return
        self._do_engine_move(self.eng_a, self.time_a)

    def _step_engine_b(self):
        if self.game_over or self.engine_thinking or self.current_color != "B":
            return
        self._do_engine_move(self.eng_b, self.time_b)

    def _step_current(self):
        """Move whichever engine's turn it is."""
        if self.game_over or self.engine_thinking:
            return
        if self.current_color == "W":
            self._step_engine_a()
        else:
            self._step_engine_b()

    def _do_engine_move(self, engine, time_ms):
        self.engine_thinking = True
        self._update_buttons()
        self._draw_board()
        t = threading.Thread(
            target=self._engine_think, args=(engine, time_ms), daemon=True
        )
        self._engine_thread = t
        t.start()

    def _engine_think(self, engine, time_ms):
        try:
            board_copy = self.board.copy()
            side = 0 if self.current_color == "W" else 1

            # Provide game history for repetition-aware search
            engine.set_game_history(self.game_history_hashes)

            move_str = engine.get_move(board_copy, side, time_ms)
            board_hash = engine.get_hash()

            stats = (engine.nodes, engine.depth, engine.score)

            if self._closing:
                return

            self.root.after(
                0, lambda: self._engine_move_done(engine, move_str, board_hash, stats)
            )
        except Exception as e:
            if self._closing:
                return
            self.root.after(0, lambda: self._engine_move_failed(engine, str(e)))

    def _engine_move_done(self, engine, move_str, board_hash, stats):
        self.engine_thinking = False

        # Check for legal moves first
        legal_moves = generate_legal_moves(self.board, self.current_color)

        if not legal_moves:
            self.game_over = True
            if in_check(self.board, self.current_color):
                winner = "B" if self.current_color == "W" else "W"
                self._record_result(winner)
            else:
                self._record_result("D")
            self._update_buttons()
            self._draw_board()
            return

        if move_str is None:
            # Engine returned no move — treat as a loss
            self.game_over = True
            loser = self.current_color
            winner = "B" if loser == "W" else "W"
            self.result_text = f"{engine.name} returned no move — {self._winner_name(winner)} wins!"
            self.result_var.set(self.result_text)
            self._tally_result(winner)
            self._update_buttons()
            self._draw_board()
            return

        # Validate the move
        is_valid, matched_move, err = validate_move(
            self.board, self.current_color, move_str, legal_moves
        )

        if not is_valid:
            self.game_over = True
            loser = self.current_color
            winner = "B" if loser == "W" else "W"
            self.result_text = f"{engine.name} illegal: {move_str} — {self._winner_name(winner)} wins!"
            self.result_var.set(self.result_text)
            self._tally_result(winner)
            self._update_buttons()
            self._draw_board()
            return

        # Update stats display — each engine keeps its own eval label
        if stats:
            nodes, depth, score = stats
            self.info_var.set(f"Depth: {depth}  Nodes: {nodes:,}")
            side_mult = 1 if self.current_color == "W" else -1
            eval_cp = score * side_mult
            eval_str = f"Eval: {eval_cp/100:+.2f}"
            if engine is self.eng_a:
                self.eval_a_var.set(eval_str)
            else:
                self.eval_b_var.set(eval_str)

        # Apply the move
        is_capture = self.board[matched_move[2][0]][matched_move[2][1]] != EMPTY
        is_pawn = piece_type(matched_move[0]) == 1
        self.board = make_move_on_board(self.board, matched_move)

        # Track move source/dest for highlighting
        (sr, sc), (tr, tc) = matched_move[1], matched_move[2]
        self.last_move = ((sr, sc), (tr, tc))

        # Game history for hash-based repetition in engines
        self.game_history_hashes.append(board_hash)

        # Halfmove clock
        if is_capture or is_pawn:
            self.halfmove_clock = 0
            self.game_history_hashes = []
            self.position_history.clear()
        else:
            self.halfmove_clock += 1

        # Update 50-move counter display (warn in orange when ≥ 80)
        self.fifty_var.set(f"50-move: {self.halfmove_clock} / 100")
        self.fifty_label.config(fg="#FF8A65" if self.halfmove_clock >= 80 else "#AAAAAA")

        # Record move in listbox
        move_num = len(self.move_history) + 1
        eng_label = "A" if engine is self.eng_a else "B"
        side_label = self.current_color
        display = f"{move_num:>3}. [{side_label}/{eng_label}] {move_str}"
        self.move_history.append((move_str, self.current_color, engine.name))
        self.move_listbox.insert(tk.END, display)
        self.move_listbox.see(tk.END)

        # Swap side
        self.current_color = "B" if self.current_color == "W" else "W"
        self.white_to_move = self.current_color == "W"
        self.turn_var.set("White to move" if self.white_to_move else "Black to move")

        # Position tracking for 3-fold repetition
        board_key = self.board.tobytes()
        self.position_history[board_key] = self.position_history.get(board_key, 0) + 1

        # Check terminal conditions
        if self._check_game_over():
            self._update_buttons()
            self._draw_board()
            return

        self._update_buttons()
        self._draw_board()

        # Continue auto-play if active
        if self.auto_playing and not self.game_over:
            self.root.after(150, self._step_current)

    def _engine_move_failed(self, engine, msg):
        self.engine_thinking = False
        self.game_over = True
        loser_color = self.current_color
        winner = "B" if loser_color == "W" else "W"
        self.result_text = f"{engine.name} crashed: {msg}"
        self.result_var.set(self.result_text)
        self._tally_result(winner)
        self._update_buttons()
        self._draw_board()

    # -----------------------------------------------------------------
    # Game-over detection
    # -----------------------------------------------------------------

    def _check_game_over(self):
        legal = generate_legal_moves(self.board, self.current_color)
        if not legal:
            self.game_over = True
            if in_check(self.board, self.current_color):
                winner = "B" if self.current_color == "W" else "W"
                self._record_result(winner)
            else:
                self._record_result("D")
            return True

        # 3-fold repetition
        board_key = self.board.tobytes()
        if self.position_history.get(board_key, 0) >= 3:
            self.game_over = True
            self._record_result("D", reason="3-fold repetition")
            return True

        # 50-move rule
        if self.halfmove_clock >= 100:
            self.game_over = True
            self._record_result("D", reason="50-move rule")
            return True

        return False

    def _record_result(self, result, reason=None):
        if result == "D":
            reason_text = f" ({reason})" if reason else ""
            self.result_text = f"Draw{reason_text}"
        elif result == "W":
            self.result_text = f"{self.name_a} (White) wins by checkmate!"
        else:
            self.result_text = f"{self.name_b} (Black) wins by checkmate!"
        self.result_var.set(self.result_text)
        self._tally_result(result)

    def _tally_result(self, result):
        if result == "W":
            self.wins_a += 1
        elif result == "B":
            self.wins_b += 1
        else:
            self.draws += 1
        total = self.wins_a + self.wins_b + self.draws
        self.score_var.set(
            f"Score: {self.name_a} {self.wins_a} - {self.draws} - {self.wins_b} {self.name_b}  "
            f"({total} games)"
        )

    def _winner_name(self, color):
        return self.name_a if color == "W" else self.name_b

    # -----------------------------------------------------------------
    # Auto-play
    # -----------------------------------------------------------------

    def _toggle_autoplay(self):
        if self.game_over:
            return
        self.auto_playing = not self.auto_playing
        if self.auto_playing:
            self.autoplay_var.set("⏩ Auto-play ON")
            self.btn_auto.config(text="⏸ Stop (P)", bg="#AD1457")
            if not self.engine_thinking:
                self.root.after(100, self._step_current)
        else:
            self.autoplay_var.set("")
            self.btn_auto.config(text="⏩ Auto-play (P)", bg="#6A1B9A")

    # -----------------------------------------------------------------
    # Controls
    # -----------------------------------------------------------------

    def _toggle_flip(self):
        self.view_flipped = not self.view_flipped
        self._draw_board()

    def _on_close(self):
        self._closing = True
        self.game_over = True
        self.auto_playing = False

        if self._engine_thread and self._engine_thread.is_alive():
            self._engine_thread.join(timeout=2.0)

        # Cleanup engines
        try:
            self.eng_a.cleanup()
        except Exception:
            pass
        try:
            self.eng_b.cleanup()
        except Exception:
            pass

        self.root.destroy()
        os._exit(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Matchmaker GUI: visual engine-vs-engine for 6x6 chess",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/matchmaker_gui.py current baseline
  python3 scripts/matchmaker_gui.py current fairy --skill-b 10
  python3 scripts/matchmaker_gui.py tuned_v2 baseline --time 1000
  python3 scripts/matchmaker_gui.py current fairy --time-a 500 --time-b 100 --skill-b 0
        """,
    )
    parser.add_argument("engine_a", help='Engine A (snapshot name, "current", or "fairy")')
    parser.add_argument("engine_b", help='Engine B (snapshot name, "current", or "fairy")')
    parser.add_argument("--time", type=int, default=200, help="Time per move in ms (default: 200)")
    parser.add_argument("--time-a", type=int, help="Override time for engine A")
    parser.add_argument("--time-b", type=int, help="Override time for engine B")
    parser.add_argument("--skill-a", type=int, help="Skill level for engine A (if fairy)")
    parser.add_argument("--skill-b", type=int, help="Skill level for engine B (if fairy)")
    parser.add_argument("--flip", action="store_true", help="Start with flipped board")
    args = parser.parse_args()

    time_a = args.time_a if args.time_a is not None else args.time
    time_b = args.time_b if args.time_b is not None else args.time

    print("Starting GUI ...\n")
    root = tk.Tk()

    print(f"Loading engine A: {args.engine_a} ...")
    eng_a = load_engine(args.engine_a, skill=args.skill_a)
    print(f"Loading engine B: {args.engine_b} ...")
    eng_b = load_engine(args.engine_b, skill=args.skill_b)
    print("Engines loaded.\n")

    app = MatchmakerGUI(
        root, eng_a, eng_b, args.engine_a, args.engine_b,
        time_a=time_a, time_b=time_b,
    )
    if args.flip:
        app.view_flipped = True
        app._draw_board()
    root.mainloop()


if __name__ == "__main__":
    main()
