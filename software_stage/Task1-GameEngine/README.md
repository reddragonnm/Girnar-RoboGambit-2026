# RoboGambit 2026 - Team Cunning Wazirs

**Submission from Girnar Hostel, IIT Delhi**

A high-performance chess engine built for RoboGambit 2026 (ARIES x Robotics Club, IIT Delhi), a 6x6 Fischer Random chess variant competition.

## Team

**Cunning Wazirs** (Girnar Hostel)

## Competition Rules (6x6 Variant)

- 6x6 board (A1 to F6), no rooks
- Pieces: Pawn (1/6), Knight (2/7), Bishop (3/8), Queen (4/9), King (5/10) for White/Black
- Pawns move one square forward only (no double push, no en passant)
- No castling
- Promotion is compulsory when a pawn reaches the last rank
- Promotion only to pieces that have been captured (e.g. cannot promote to queen if your queen is still alive)
- Fischer Random starting positions: back rank is shuffled each match (bishops on opposite colors)
- Input: 6x6 NumPy array where `board[0][0]` = A1 (bottom-left), 0 = empty, 1-10 = pieces
- Output: move string like `1:B2->B3` or `4:A5->A6=4` for promotion

## Move Format

```
<piece_id>:<source_cell>-><target_cell>
```

For pawn promotions, the promoted piece ID is appended:

```
<piece_id>:<source_cell>-><target_cell>=<piece_id>
```

Examples:
- `1:B2->B3` (white pawn from B2 to B3)
- `4:A5->A6=4` (white pawn promotes to queen on A6)
- `9:A2->A1=9` (black pawn promotes to queen on A1)

## Project Structure

```
chess6x6/
├── Game.py                 # Competition entry point (Python API)
├── gui.py                  # Interactive Tkinter GUI
├── libchess6x6.so          # Compiled shared library (Python ctypes binding)
├── chess6x6                # Standalone engine binary
├── Makefile                # Build system
├── engine/                 # C++ engine source
│   ├── types.h             # Core types, piece IDs, move encoding
│   ├── attacks.h           # PEXT-based attack tables for 6x6
│   ├── zobrist.h           # Zobrist hashing
│   ├── pst.h               # Piece-square tables (Texel-tuned)
│   ├── board.h             # Board representation, movegen, make/unmake, SEE
│   ├── eval.h              # Tapered evaluation (material, PST, mobility, pawns, king safety)
│   ├── search.h            # PVS search with 15+ pruning/reduction techniques
│   ├── engine_api.cpp      # C API exposed to Python via ctypes
│   ├── main.cpp            # Standalone engine with benchmark suite
│   ├── datagen.cpp         # Multi-threaded self-play data generator
│   └── tuner.cpp           # Texel tuner (analytical gradients, Adam optimizer)
├── scripts/
│   ├── matchmaker_gui.py   # Engine vs engine GUI tournament tool
│   └── matchmaker.py       # Engine wrapper for matchmaker
├── snapshots/              # Saved engine versions (.so files) for comparison
└── variants.ini            # Fairy Stockfish variant definition
```

## Building

Requires: `g++` with C++17 support

```bash
# Build the shared library (for Python)
make lib

# Build the standalone engine binary
make

# Build the training data generator
make datagen

# Debug build (with address sanitizer)
make debug

# Clean all
make clean
```

## Running

### Play against the engine (GUI)

```bash
# Play as black against the engine (engine plays white)
python3 gui.py --black --time 1000

# Play as white
python3 gui.py --white --time 1000

# Watch engine vs engine
python3 gui.py --time 1000

# Play against Fairy Stockfish
python3 gui.py --black --fairy --skill 16 --time 1000

# Flip the board view
python3 gui.py --black --time 1000 --flip
```

### Engine vs Engine matchmaker

```bash
# Current engine vs a saved snapshot
python3 scripts/matchmaker_gui.py current baseline --time 1000

# Current engine vs Fairy Stockfish (skill 16)
python3 scripts/matchmaker_gui.py current fairy --skill-b 16 --time 1000
```

### Using Game.py directly (competition API)

```python
import numpy as np
from Game import get_best_move

board = np.array([
    [3, 2, 4, 5, 2, 3],   # rank 1 (white pieces)
    [1, 1, 1, 1, 1, 1],   # rank 2 (white pawns)
    [0, 0, 0, 0, 0, 0],   # rank 3
    [0, 0, 0, 0, 0, 0],   # rank 4
    [6, 6, 6, 6, 6, 6],   # rank 5 (black pawns)
    [8, 7, 9, 10, 7, 8],  # rank 6 (black pieces)
])

move = get_best_move(board, playing_white=True)
print(move)  # e.g. "1:B2->B3"
```

### Generating training data

```bash
# Generate 10000 games at depth 8
./datagen 10000 8 tuning_data.txt
```

## Engine Architecture

### Board Representation

Hybrid bitboard + mailbox system optimized for a 6x6 board (36 squares, fits in a single 64-bit integer).

- Bitboards for each piece type and color (fast attack/occupancy queries)
- Mailbox array for O(1) piece-at-square lookups
- Incremental Zobrist hashing
- Incremental PST score maintenance (no recomputation on make/unmake)

### Attack Generation

PEXT-based sliding attack tables for bishops and queens (4 or 8 ray directions per square). Knights, kings, and pawns use precomputed lookup tables. All attack generation is O(1) per piece.

### Evaluation

Tapered evaluation blending midgame and endgame scores based on game phase (0 to 10, computed from remaining material).

Features:
- Material values (Texel-tuned): Pawn 122, Knight 308, Bishop 287, Queen 890
- Piece-square tables for all piece types (separate midgame/endgame tables for king)
- Bishop pair bonus (74 cp)
- Mobility (knight, bishop, queen move counts)
- Pawn structure: passed pawns (distance-to-promotion bonus), doubled pawns penalty, isolated pawns penalty
- King safety: pawn shield bonus, center penalty in midgame
- Insufficient material detection: KvK, K+minor vs K, K+minor vs K+minor all return 0
- Pawn hash table (8192 entries) for caching pawn structure evaluation

### Search

Principal Variation Search (PVS) with iterative deepening. The following techniques are implemented:

**Core:**
- Alpha-beta with negamax framework
- Iterative deepening with aspiration windows
- Transposition table (64 MB, 4M entries, two slots per bucket, aging)
- Quiescence search with delta pruning and SEE pruning
- Check extension (+1 depth when in check)

**Move Ordering:**
- TT move first (score: 1,000,000)
- Good captures ordered by MVV-LVA and SEE (score: 100,000+)
- Promotions (score: 90,000+)
- Killer moves (2 per ply, scores: 80,000 / 79,000)
- Countermove heuristic (score: 70,000)
- History heuristic with gravity damping

**Pruning and Reductions:**
- Null move pruning (R = 3 + depth/4, requires non-pawn material)
- Late Move Reductions (LMR) with log-based reduction table
- Reverse futility pruning (depth <= 7)
- Futility pruning (depth <= 6)
- Razoring (depth <= 2)
- Late Move Pruning / movecount pruning (depth <= 5)
- SEE pruning for bad captures (depth <= 5)
- Singular extension (extend TT move if significantly better than alternatives)
- Internal Iterative Deepening (IID) for PV nodes without a hash move

**Draw Detection:**
- Three-fold repetition (with contempt factor of 15 cp)
- 50-move rule (halfmove clock >= 100)
- Insufficient material (KvK, K+minor vs K, K+minor vs K+minor)

**Time Management:**
- Configurable time limit per move (default: 1000 ms)
- Time check every 4096 nodes for minimal overhead

### Data Generation

Multi-threaded self-play data generator using Fischer Random starting positions. Each game:
- Starts from a random legal Fischer Random position
- Plays 4 to 10 random opening moves for diversity
- Records quiet positions only (not in check, no immediate captures)
- Outputs 39-column format: board[36], side to move, search score, game result

## GUI Controls

| Key/Action | Function |
|-----------|----------|
| Click piece, click target | Make a move |
| Space | Trigger engine move |
| F | Flip board |
| H | Show hint (Fairy Stockfish) |
| N | Show engine hint (local engine) |
| U | Undo last move |
| R | New game (random Fischer Random position) |
| Arrow keys | Navigate move history |

## Dependencies

- Python 3.8+
- NumPy
- tkinter (for GUI, usually included with Python)
- g++ with C++17 support (for building)
- Optional: Fairy Stockfish binary (for playing against or generating data)
