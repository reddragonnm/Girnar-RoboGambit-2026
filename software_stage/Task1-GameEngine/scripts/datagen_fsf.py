#!/usr/bin/env python3
import sys
import os
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from matchmaker import EngineInstance, UCIEngineInstance, LIB_NAME
from validate_engine import generate_fischer_random, generate_legal_moves, make_move_on_board, validate_move, in_check, piece_type

def play_game_and_save(engine_w, engine_b, board, time_ms, verbose=False):
    engines = {'W': engine_w, 'B': engine_b}
    current_color = 'W'
    move_count = 0
    halfmove_clock = 0
    position_history = {}
    game_history_hashes = []
    
    positions_this_game = []

    while move_count < 200:
        legal_moves = generate_legal_moves(board, current_color)

        if len(legal_moves) == 0:
            if in_check(board, current_color):
                winner = 'B' if current_color == 'W' else 'W'
                return winner, positions_this_game
            else:
                return 'D', positions_this_game

        if halfmove_clock >= 100:
            return 'D', positions_this_game

        board_key = board.tobytes()
        position_history[board_key] = position_history.get(board_key, 0) + 1
        if position_history[board_key] >= 3:
            return 'D', positions_this_game

        # We only save positions after the opening (move 8) and avoid deep endgames
        # where EGTB would just dominate anyway.
        if 8 < move_count < 100:
            # Flatten board to match C++ (row 0 = rank 1)
            flat_board = list(board.flatten())
            side = 0 if current_color == 'W' else 1
            positions_this_game.append((flat_board, side))

        side = 0 if current_color == 'W' else 1
        eng = engines[current_color]

        eng.set_game_history(game_history_hashes)

        try:
            move_str = eng.get_move(board, side, time_ms)
            board_hash = eng.get_hash()
        except:
            return ('B' if current_color == 'W' else 'W'), positions_this_game

        if move_str is None:
            return ('B' if current_color == 'W' else 'W'), positions_this_game

        is_valid, matched_move, err = validate_move(board, current_color, move_str, legal_moves)

        if not is_valid:
            return ('B' if current_color == 'W' else 'W'), positions_this_game

        game_history_hashes.append(board_hash)

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
        move_count += 1

    return 'D', positions_this_game

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=50)
    parser.add_argument('--time', type=int, default=30)
    parser.add_argument('--out', type=str, default='tuning_data_fsf.txt')
    args = parser.parse_args()

    lib_path = os.path.join(PROJECT_DIR, LIB_NAME)
    if not os.path.exists(lib_path):
        print(f"Error: {LIB_NAME} not found. Run 'make lib'.")
        return

    eng_current = EngineInstance(lib_path, name='current')
    
    fsf_path = os.path.join(PROJECT_DIR, "fairy-stockfish")
    if not os.path.exists(fsf_path):
        import shutil
        fsf_path = shutil.which("fairy-stockfish")
        if not fsf_path:
            print("Error: fairy-stockfish not found.")
            return

    eng_fsf = UCIEngineInstance(fsf_path, name='fairy')

    total_positions = 0
    with open(args.out, 'w') as f:
        for i in range(args.games):
            base_board = generate_fischer_random()
            
            # Alternate sides
            if i % 2 == 0:
                engine_w, engine_b = eng_current, eng_fsf
            else:
                engine_w, engine_b = eng_fsf, eng_current

            # Reset internal engine states
            eng_current.cleanup()
            eng_current.lib.engine_init()
            eng_current.clear_history()
            
            if eng_fsf.process is None:
                eng_fsf.__init__(eng_fsf.binary_path, name=eng_fsf.name, variant=eng_fsf.variant)
            else:
                eng_fsf.clear_history()

            result, positions = play_game_and_save(engine_w, engine_b, base_board.copy(), args.time)

            if result == 'D': res_val = 0.5
            elif result == 'W': res_val = 1.0
            else: res_val = 0.0

            for board_arr, side in positions:
                line = " ".join(str(x) for x in board_arr)
                f.write(f"{line} {side} {res_val}\n")
                total_positions += 1

            if (i+1) % 5 == 0:
                print(f"Games done: {i+1}/{args.games} | Positions extracted: {total_positions}")

    eng_current.cleanup()
    eng_fsf.cleanup()
    print(f"\nFinished! Saved {total_positions} positions to {args.out}")

if __name__ == '__main__':
    main()
