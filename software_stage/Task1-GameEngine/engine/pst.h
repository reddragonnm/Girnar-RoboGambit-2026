#pragma once
#include "types.h"

// Material values (centipawns)
constexpr float MATERIAL_VALUE[PIECE_TYPE_COUNT] = {
    0.0f,       // NONE
    122.1f,     // PAWN
    307.8f,     // KNIGHT
    287.3f,     // BISHOP
    889.8f,    // QUEEN
    20000.0f    // KING
};

// Phase calculation for tapered eval
constexpr int PHASE_WEIGHT[PIECE_TYPE_COUNT] = {0, 0, 1, 1, 3, 0};
constexpr int TOTAL_PHASE = 10; 


// Piece-Square Tables (from white's perspective, A1=index 0)
constexpr float PAWN_PST[NUM_SQUARES] = {
        0.00f,    0.00f,    0.00f,    0.00f,    0.00f,    0.00f,
        2.45f,   -2.25f,   -3.20f,   -3.20f,   -2.25f,    2.45f,
       -0.95f,   -0.01f,    1.45f,    1.45f,   -0.01f,   -0.95f,
       -0.40f,    1.00f,    1.74f,    1.74f,    1.00f,   -0.40f,
        0.21f,    0.19f,    0.15f,    0.15f,    0.19f,    0.21f,
        0.00f,    0.00f,    0.00f,    0.00f,    0.00f,    0.00f
};

constexpr float KNIGHT_PST[NUM_SQUARES] = {
       -1.57f,   -0.49f,   -0.15f,   -0.15f,   -0.49f,   -1.57f,
       -0.02f,    0.37f,    1.01f,    1.01f,    0.37f,   -0.02f,
       -0.09f,   -0.00f,   -0.14f,   -0.14f,   -0.00f,   -0.09f,
        0.19f,    0.25f,    0.17f,    0.17f,    0.25f,    0.19f,
        0.08f,    0.25f,    0.04f,    0.04f,    0.25f,    0.08f,
        0.06f,    0.03f,   -0.05f,   -0.05f,    0.03f,    0.06f
};

constexpr float BISHOP_PST[NUM_SQUARES] = {
       -0.85f,   -0.09f,   -0.42f,   -0.42f,   -0.09f,   -0.85f,
       -0.14f,    1.07f,    0.57f,    0.57f,    1.07f,   -0.14f,
        0.08f,    0.18f,   -0.29f,   -0.29f,    0.18f,    0.08f,
       -0.03f,   -0.01f,   -0.13f,   -0.13f,   -0.01f,   -0.03f,
       -0.02f,    0.10f,   -0.16f,   -0.16f,    0.10f,   -0.02f,
        0.05f,   -0.03f,   -0.05f,   -0.05f,   -0.03f,    0.05f
};

constexpr float QUEEN_PST[NUM_SQUARES] = {
       -0.42f,   -0.53f,   -0.25f,   -0.25f,   -0.53f,   -0.42f,
       -0.33f,   -0.08f,    0.40f,    0.40f,   -0.08f,   -0.33f,
        0.17f,    0.09f,   -0.08f,   -0.08f,    0.09f,    0.17f,
        0.11f,    0.11f,    0.19f,    0.19f,    0.11f,    0.11f,
        0.11f,    0.11f,    0.06f,    0.06f,    0.11f,    0.11f,
        0.10f,    0.03f,    0.09f,    0.09f,    0.03f,    0.10f
};

constexpr float KING_PST_MIDGAME[NUM_SQUARES] = {
        0.27f,   -0.06f,   -0.16f,   -0.16f,   -0.06f,    0.27f,
       -0.08f,    0.12f,   -0.13f,   -0.13f,    0.12f,   -0.08f,
        0.05f,    0.02f,    0.03f,    0.03f,    0.02f,    0.05f,
       -0.02f,   -0.00f,   -0.04f,   -0.04f,   -0.00f,   -0.02f,
        0.00f,    0.01f,   -0.02f,   -0.02f,    0.01f,    0.00f,
       -0.00f,   -0.01f,   -0.00f,   -0.00f,   -0.01f,   -0.00f
};

constexpr float KING_PST_ENDGAME[NUM_SQUARES] = {
        0.15f,   -0.18f,   -0.26f,   -0.26f,   -0.18f,    0.15f,
       -0.01f,    0.13f,    0.01f,    0.01f,    0.13f,   -0.01f,
        0.13f,    0.07f,    0.02f,    0.02f,    0.07f,    0.13f,
        0.06f,    0.07f,   -0.31f,   -0.31f,    0.07f,    0.06f,
        0.06f,    0.07f,   -0.05f,   -0.05f,    0.07f,    0.06f,
        0.01f,    0.02f,    0.02f,    0.02f,    0.02f,    0.01f
};

// PST lookup table (indexed by piece type)
// Note: King uses KING_PST_MIDGAME for mg and KING_PST_ENDGAME for eg
inline const float* PST_TABLE[PIECE_TYPE_COUNT] = {
    nullptr,
    PAWN_PST,
    KNIGHT_PST,
    BISHOP_PST,
    QUEEN_PST,
    KING_PST_MIDGAME
};

// Get PST square index for a color (flip rank for black)
constexpr int pstSquare(int sq, Color c) {
    return (c == WHITE) ? sq : (5 - rankOf(sq)) * 6 + fileOf(sq);
}

// Get midgame PST + material value for a piece on a square
inline float mgPstValue(PieceType pt, int sq, Color c) {
    int pstSq = pstSquare(sq, c);
    float val = MATERIAL_VALUE[pt];
    if (pt == KING) {
        val += KING_PST_MIDGAME[pstSq];
    } else {
        val += PST_TABLE[pt][pstSq];
    }
    return val;
}

// Get endgame PST + material value for a piece on a square
inline float egPstValue(PieceType pt, int sq, Color c) {
    int pstSq = pstSquare(sq, c);
    float val = MATERIAL_VALUE[pt];
    if (pt == KING) {
        val += KING_PST_ENDGAME[pstSq];
    } else if (PST_TABLE[pt]) {
        val += PST_TABLE[pt][pstSq];
    }
    return val;
}
