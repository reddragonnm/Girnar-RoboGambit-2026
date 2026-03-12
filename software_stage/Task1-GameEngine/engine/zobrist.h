#pragma once
#include "types.h"
#include <random>
#include <array>

namespace Zobrist {

// Hash keys: [color][pieceType][square]
uint64_t pieceKeys[COLOR_COUNT][PIECE_TYPE_COUNT][NUM_SQUARES];
uint64_t sideKey; // XOR when it's black's turn

inline void init() {
    std::mt19937_64 rng(0xDEADBEEF42ULL); // Fixed seed for reproducibility
    
    for (int c = 0; c < COLOR_COUNT; c++)
        for (int pt = 0; pt < PIECE_TYPE_COUNT; pt++)
            for (int sq = 0; sq < NUM_SQUARES; sq++)
                pieceKeys[c][pt][sq] = rng();
    
    sideKey = rng();
}

} // namespace Zobrist
