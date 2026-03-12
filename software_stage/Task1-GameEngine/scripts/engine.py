"""
Python wrapper for the RoboGambit 6x6 Chess Engine.

Usage:
    import numpy as np
    from engine import ChessEngine

    engine = ChessEngine()
    
    # Board state as 6x6 numpy array
    # Row 0 = Rank 1 (bottom), Row 5 = Rank 6 (top)
    # Values: 0=empty, 1=WPawn, 2=WKnight, 3=WBishop, 4=WQueen, 5=WKing,
    #         6=BPawn, 7=BKnight, 8=BBishop, 9=BQueen, 10=BKing
    board = np.array([
        [3, 2, 4, 5, 2, 3],  # Rank 1: B N Q K N B
        [1, 1, 1, 1, 1, 1],  # Rank 2: pawns
        [0, 0, 0, 0, 0, 0],  # Rank 3
        [0, 0, 0, 0, 0, 0],  # Rank 4
        [6, 6, 6, 6, 6, 6],  # Rank 5: pawns
        [8, 7, 9, 10, 7, 8], # Rank 6: b n q k n b
    ], dtype=np.int32)
    
    move = engine.get_move(board, side_to_move=0, time_limit_ms=2000)
    print(move)  # e.g., "1:E2->E3"
"""

import ctypes
import numpy as np
import os
import subprocess
import sys


class ChessEngine:
    def __init__(self, lib_path=None):
        if lib_path is None:
            # Try to find the shared library in the same directory
            dir_path = os.path.dirname(os.path.abspath(__file__))
            lib_path = os.path.join(dir_path, "libchess6x6.so")
            
            # Build if not found
            if not os.path.exists(lib_path):
                print("Building engine shared library...")
                subprocess.run(["make", "lib"], cwd=dir_path, check=True)
        
        self.lib = ctypes.CDLL(lib_path)
        
        # Set up function signatures
        self.lib.engine_init.restype = None
        self.lib.engine_init.argtypes = []
        
        self.lib.engine_get_move.restype = ctypes.c_int
        self.lib.engine_get_move.argtypes = [
            ctypes.POINTER(ctypes.c_int),   # board_arr (36 ints)
            ctypes.c_int,                    # side_to_move
            ctypes.c_int,                    # time_limit_ms
            ctypes.POINTER(ctypes.c_int),    # captured_white (6 ints, nullable)
            ctypes.POINTER(ctypes.c_int),    # captured_black (6 ints, nullable)
            ctypes.c_char_p,                 # result_buf
            ctypes.c_int                     # buf_size
        ]
        
        self.lib.engine_get_nodes.restype = ctypes.c_int
        self.lib.engine_get_depth.restype = ctypes.c_int
        self.lib.engine_get_score.restype = ctypes.c_int
        
        self.lib.engine_cleanup.restype = None
        
        # Initialize
        self.lib.engine_init()
    
    def get_move(self, board_state: np.ndarray, side_to_move: int = 0,
                 time_limit_ms: int = 2000,
                 captured_white: np.ndarray = None,
                 captured_black: np.ndarray = None) -> str:
        """
        Get the best move for the given board state.
        
        Args:
            board_state: 6x6 numpy array with piece IDs (0-10)
                Row 0 = Rank 1 (bottom), Row 5 = Rank 6 (top)
            side_to_move: 0 for white, 1 for black
            time_limit_ms: search time limit in milliseconds
            captured_white: array of 6 ints [NONE, PAWN, KNIGHT, BISHOP, QUEEN, KING]
                           counts of captured white pieces (for promotion logic)
            captured_black: same for black pieces
        
        Returns:
            Move string in format "<piece_id>:<source>-><target>"
            e.g., "1:E2->E3"
        """
        # Flatten board to 1D array (row-major, rank 1 first)
        flat = board_state.flatten().astype(np.int32)
        if len(flat) != 36:
            raise ValueError(f"Board must be 6x6 (36 elements), got {len(flat)}")
        
        board_ptr = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        
        # Handle captured piece counts
        cap_w_ptr = None
        cap_b_ptr = None
        if captured_white is not None:
            cap_w = captured_white.astype(np.int32)
            cap_w_ptr = cap_w.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        if captured_black is not None:
            cap_b = captured_black.astype(np.int32)
            cap_b_ptr = cap_b.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        
        # Result buffer
        buf = ctypes.create_string_buffer(64)
        
        result = self.lib.engine_get_move(
            board_ptr, side_to_move, time_limit_ms,
            cap_w_ptr, cap_b_ptr,
            buf, 64
        )
        
        if result < 0:
            raise RuntimeError("Engine failed to find a move (no legal moves?)")
        
        return buf.value.decode('ascii')
    
    @property
    def nodes(self) -> int:
        """Number of nodes searched in last search."""
        return self.lib.engine_get_nodes()
    
    @property
    def depth(self) -> int:
        """Maximum depth reached in last search."""
        return self.lib.engine_get_depth()
    
    @property
    def score(self) -> int:
        """Evaluation score (centipawns) from last search."""
        return self.lib.engine_get_score()
    
    def __del__(self):
        if hasattr(self, 'lib'):
            self.lib.engine_cleanup()


# Convenience function matching the competition interface
def get_best_move(board_state: np.ndarray, side_to_move: int = 0,
                  time_limit_ms: int = 2000) -> str:
    """
    Competition interface: takes a NumPy array, returns a move string.
    
    Args:
        board_state: 6x6 numpy array with piece IDs
        side_to_move: 0 for white, 1 for black
        time_limit_ms: search time limit
    
    Returns:
        Move string in format "<piece_id>:<source>-><target>"
    """
    engine = ChessEngine()
    return engine.get_move(board_state, side_to_move, time_limit_ms)


if __name__ == "__main__":
    # Test
    board = np.array([
        [3, 2, 4, 5, 2, 3],
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [6, 6, 6, 6, 6, 6],
        [8, 7, 9, 10, 7, 8],
    ], dtype=np.int32)
    
    engine = ChessEngine()
    move = engine.get_move(board, side_to_move=0, time_limit_ms=3000)
    print(f"Best move: {move}")
    print(f"Nodes: {engine.nodes}")
    print(f"Depth: {engine.depth}")
    print(f"Score: {engine.score} cp")
