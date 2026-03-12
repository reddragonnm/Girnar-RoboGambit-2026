// C API for Python ctypes interface
#include "types.h"
#include "attacks.h"
#include "zobrist.h"
#include "board.h"
#include "eval.h"
#include "search.h"
#include <cstdlib>
#include <cstring>

static bool initialized = false;
static Search* engine = nullptr;

// Game history for cross-call repetition detection
static uint64_t gameHistory[MAX_GAME_PLY];
static int gameHistoryCount = 0;

extern "C" {

// Initialize the engine (call once)
void engine_init() {
    if (!initialized) {
        Attacks::init();
        Zobrist::init();
        initialized = true;
    }
    if (engine) delete engine;
    engine = new Search();
}

// Set game position history for cross-call repetition detection.
// hashes: array of Zobrist hashes from previous positions in the game
// count: number of hashes
// Call this before engine_get_move() each turn.
// The hashes should be in chronological order (oldest first).
void engine_set_game_history(const uint64_t* hashes, int count) {
    if (count > MAX_GAME_PLY) count = MAX_GAME_PLY;
    if (count > 0 && hashes) {
        memcpy(gameHistory, hashes, count * sizeof(uint64_t));
    }
    gameHistoryCount = count;
}

// Clear game history (call when starting a new game)
void engine_clear_game_history() {
    gameHistoryCount = 0;
}

// Get the best move for a given board state
// board_arr: flat array of 36 ints (row-major, rank 1 first)
//   board_arr[rank * 6 + file] where rank 0 = rank 1, file 0 = file A
// side_to_move: 0 = white, 1 = black
// time_limit_ms: time limit in milliseconds
// captured_white: flat array of 6 ints [NONE, PAWN, KNIGHT, BISHOP, QUEEN, KING] captured counts for white
// captured_black: flat array of 6 ints captured counts for black
// result_buf: output buffer for move string (at least 32 bytes)
// Returns: length of result string, or -1 on error
int engine_get_move(const int* board_arr, int side_to_move, int time_limit_ms,
                    const int* captured_white, const int* captured_black,
                    char* result_buf, int buf_size) {
    if (!initialized || !engine) {
        engine_init();
    }
    
    // Set up board
    engine->board.fromFlatArray(board_arr);
    engine->board.side = (side_to_move == 0) ? WHITE : BLACK;
    // Always use known variant starting counts (6P,2N,2B,1Q,1K) for promotion rules.
    // fromFlatArray() sets initialCount from the CURRENT board, which is wrong for mid-game input.
    engine->board.setRoboGambitInitialCounts();
    
    // Set captured counts: use provided arrays if given, otherwise derive
    // from the known RoboGambit starting counts (6P, 2N, 2B, 1Q, 1K per side).
    // This is critical: fromFlatArray calls computeInitialCounts() which
    // incorrectly assumes the current board IS the starting position.
    // For mid-game boards, we must use the known starting counts.
    if (captured_white && captured_black) {
        for (int i = 0; i < PIECE_TYPE_COUNT; i++) {
            engine->board.capturedCount[WHITE][i] = captured_white[i];
            engine->board.capturedCount[BLACK][i] = captured_black[i];
        }
    } else {
        // Derive from known RoboGambit starting piece counts
        engine->board.computeCapturedFromInitial();
    }
    
    // Recompute hash for correct side
    // Hash is computed incrementally during fromFlatArray, but side key needs setting
    if (engine->board.side == BLACK) {
        engine->board.hash ^= Zobrist::sideKey;
    }
    
    // Inject game history into the undo stack for cross-call repetition detection.
    // The search's repetition check (search.h) scans undoStack[0..undoCount-1]
    // looking for matching hashes. By placing prior game position hashes here,
    // the search will detect positions that occurred in earlier turns.
    //
    // We store each prior hash in undoStack[i].hash. We also set halfmoveClock
    // high enough that the repetition scan window covers the entire history.
    // The search checks: limit = max(0, stackSize - board.halfmoveClock)
    // So we set halfmoveClock = gameHistoryCount to ensure all entries are scanned.
    if (gameHistoryCount > 0) {
        for (int i = 0; i < gameHistoryCount && i < MAX_GAME_PLY; i++) {
            engine->board.undoStack[i].hash = gameHistory[i];
            engine->board.undoStack[i].move = MOVE_NONE;
            engine->board.undoStack[i].captured = 0;
            engine->board.undoStack[i].capturedType = NONE;
            engine->board.undoStack[i].capturedColor = WHITE;
            engine->board.undoStack[i].halfmoveClock = 0;
        }
        engine->board.undoCount = gameHistoryCount;
        // Set halfmoveClock large enough so the repetition scan window
        // covers all injected history entries. This is safe because
        // irreversible moves (captures, pawn moves) would have reset the
        // real half-move clock in-game — the Python side should clear
        // the history at those points or just let the engine see all of it.
        engine->board.halfmoveClock = gameHistoryCount;
    }
    
    // Search
    Move best = engine->search(0, time_limit_ms);
    
    if (best == MOVE_NONE) {
        result_buf[0] = '\0';
        return -1;
    }
    
    // Format result
    std::string moveStr = engine->bestMoveStr();
    
    int len = (int)moveStr.size();
    if (len > buf_size - 1) len = buf_size - 1;
    memcpy(result_buf, moveStr.c_str(), len);
    result_buf[len] = '\0';
    
    return len;
}

// Get the Zobrist hash of the current board position (after engine_get_move setup)
// Returns the hash as two 32-bit halves (lo, hi) to avoid uint64 ABI issues with ctypes.
// hash_out: pointer to 2 uint32_t values [lo, hi]
void engine_get_hash(uint32_t* hash_out) {
    if (!engine) {
        hash_out[0] = hash_out[1] = 0;
        return;
    }
    hash_out[0] = (uint32_t)(engine->board.hash & 0xFFFFFFFF);
    hash_out[1] = (uint32_t)(engine->board.hash >> 32);
}

// Get search statistics after last search
int engine_get_nodes() {
    return engine ? engine->info.nodes : 0;
}

int engine_get_depth() {
    return engine ? engine->info.maxDepth : 0;
}

int engine_get_score() {
    return engine ? engine->info.bestScore : 0;
}

// Static evaluation of a position (no search, just eval function)
// Returns score in centipawns from white's perspective
int engine_static_eval(const int* board_arr, int side_to_move) {
    if (!initialized || !engine) {
        engine_init();
    }
    
    Board tempBoard;
    tempBoard.fromFlatArray(board_arr);
    tempBoard.side = (side_to_move == 0) ? WHITE : BLACK;
    tempBoard.setRoboGambitInitialCounts();
    tempBoard.computeCapturedFromInitial();
    
    int score = Eval::evaluate(tempBoard);
    // evaluate() returns from side-to-move perspective; convert to white perspective
    if (tempBoard.side == BLACK) score = -score;
    return score;
}

// Signal the engine to stop searching immediately.
// Safe to call from any thread while engine_get_move() is running.
void engine_stop_search() {
    if (engine) {
        engine->info.stopped = true;
    }
}

// Clean up
void engine_cleanup() {
    delete engine;
    engine = nullptr;
}

} // extern "C"
