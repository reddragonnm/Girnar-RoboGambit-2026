#pragma once
#include "board.h"
#include <string>

namespace Eval {

// Bishop pair bonus
constexpr float BISHOP_PAIR_BONUS = 74.1f;

// Mobility weights per piece type
constexpr float MOBILITY_WEIGHT[PIECE_TYPE_COUNT] = {
    0.0f,   // NONE
    0.0f,   // PAWN
    25.0f,  // KNIGHT
    25.0f,  // BISHOP
    22.1f,  // QUEEN
    0.0f    // KING
};

// King safety
constexpr float PAWN_SHIELD_BONUS = 10.0f;
constexpr float KING_CENTER_PENALTY = 30.0f;

// ============================================================
// Pawn hash table
// ============================================================
struct PawnEntry {
    uint64_t key;
    float mgScore;
    float egScore;
};

constexpr size_t PAWN_HASH_SIZE = 8192; // small, fits in L1 cache (must be power of 2)
constexpr size_t PAWN_HASH_MASK = PAWN_HASH_SIZE - 1;
PawnEntry pawnHash[PAWN_HASH_SIZE];

inline void clearPawnHash() {
    memset(pawnHash, 0, sizeof(pawnHash));
}

// Compute a hash key for pawn structure only
inline uint64_t pawnHashKey(const Board& board) {
    // Use just pawn bitboards as a simple hash
    return board.pieces[WHITE][PAWN] * 0x9E3779B97F4A7C15ULL ^ 
           board.pieces[BLACK][PAWN] * 0x517CC1B727220A95ULL;
}

// Evaluate pawn structure using bitboard operations (no loops over enemy pawns)
inline void evaluatePawns(const Board& board, float& mgScore, float& egScore) {
    uint64_t key = pawnHashKey(board);
    size_t idx = key & PAWN_HASH_MASK;
    PawnEntry& entry = pawnHash[idx];
    
    if (entry.key == key) {
        mgScore += entry.mgScore;
        egScore += entry.egScore;
        return;
    }
    
    float mg = 0.0f, eg = 0.0f;
    
    for (int c = 0; c < COLOR_COUNT; c++) {
        float sign = (c == WHITE) ? 1.0f : -1.0f;
        Bitboard pawns = board.pieces[c][PAWN];
        Bitboard enemyPawns = board.pieces[1 - c][PAWN];
        
        Bitboard tmp = pawns;
        while (tmp) {
            int sq = popLsb(tmp);
            int f = fileOf(sq);
            int r = rankOf(sq);
            
            // Passed pawn: no enemy pawns on same or adjacent files ahead
            if (!(Attacks::passedPawnMask[c][sq] & enemyPawns)) {
                int distToPromo = (c == WHITE) ? (5 - r) : r;
                float bonus = (float)((5 - distToPromo)) * 1.5f;
                mg += sign * bonus * 12.53f;
                eg += sign * bonus * 36.53f;
            }
            
            // Doubled pawn: another friendly pawn on same file
            Bitboard sameFileFriends = pawns & Attacks::fileMask[f] & ~sqBit(sq);
            if (sameFileFriends) {
                float doubled = (float)popcount(sameFileFriends);
                mg -= sign * 14.85f * doubled;
                eg -= sign * 17.17f * doubled;
            }
            
            // Isolated pawn: no friendly pawns on adjacent files
            if (!(pawns & Attacks::adjacentFileMask[f])) {
                mg -= sign * 6.79f;
                eg -= sign * 4.43f;
            }
        }
    }
    
    entry.key = key;
    entry.mgScore = mg;
    entry.egScore = eg;
    
    mgScore += mg;
    egScore += eg;
}

// Evaluate the board position from the perspective of the side to move
inline int evaluateClassical(const Board& board) {
    // Start with incrementally-maintained material + PST scores
    float mgScore = board.mgPst[WHITE] - board.mgPst[BLACK];
    float egScore = board.egPst[WHITE] - board.egPst[BLACK];
    int phase = board.phase;
    
    // Bishop pair bonus (unrolled, no loop)
    if (popcount(board.pieces[WHITE][BISHOP]) >= 2) {
        mgScore += BISHOP_PAIR_BONUS;
        egScore += BISHOP_PAIR_BONUS;
    }
    if (popcount(board.pieces[BLACK][BISHOP]) >= 2) {
        mgScore -= BISHOP_PAIR_BONUS;
        egScore -= BISHOP_PAIR_BONUS;
    }
    
    // Mobility evaluation - compute white minus black mobility in one pass
    float mobMg = 0.0f;
    
    // White pieces
    {
        Bitboard notOwn = ~board.occupied[WHITE];
        Bitboard bb;
        
        bb = board.pieces[WHITE][KNIGHT];
        while (bb) { int sq = popLsb(bb); int mob = popcount(Attacks::knightAttacks[sq] & notOwn); mobMg += mob * MOBILITY_WEIGHT[KNIGHT]; }
        
        bb = board.pieces[WHITE][BISHOP];
        while (bb) { int sq = popLsb(bb); int mob = popcount(Attacks::bishopAttacks(sq, board.allOccupied) & notOwn); mobMg += mob * MOBILITY_WEIGHT[BISHOP]; }
        
        bb = board.pieces[WHITE][QUEEN];
        while (bb) { int sq = popLsb(bb); int mob = popcount(Attacks::queenAttacks(sq, board.allOccupied) & notOwn); mobMg += mob * MOBILITY_WEIGHT[QUEEN]; }
    }
    
    // Black pieces
    {
        Bitboard notOwn = ~board.occupied[BLACK];
        Bitboard bb;
        
        bb = board.pieces[BLACK][KNIGHT];
        while (bb) { int sq = popLsb(bb); int mob = popcount(Attacks::knightAttacks[sq] & notOwn); mobMg -= mob * MOBILITY_WEIGHT[KNIGHT]; }
        
        bb = board.pieces[BLACK][BISHOP];
        while (bb) { int sq = popLsb(bb); int mob = popcount(Attacks::bishopAttacks(sq, board.allOccupied) & notOwn); mobMg -= mob * MOBILITY_WEIGHT[BISHOP]; }
        
        bb = board.pieces[BLACK][QUEEN];
        while (bb) { int sq = popLsb(bb); int mob = popcount(Attacks::queenAttacks(sq, board.allOccupied) & notOwn); mobMg -= mob * MOBILITY_WEIGHT[QUEEN]; }
    }
    
    // Mobility is same for mg and eg (same weights)
    mgScore += mobMg;
    egScore += mobMg;
    
    // Pawn structure (with hash table)
    evaluatePawns(board, mgScore, egScore);
    
    // King safety (midgame) - unrolled
    {
        int ksq = board.kingSq[WHITE];
        mgScore += popcount(Attacks::kingAttacks[ksq] & board.pieces[WHITE][PAWN]) * PAWN_SHIELD_BONUS;
        int kf = fileOf(ksq), kr = rankOf(ksq);
        if (kf >= 2 && kf <= 3 && kr >= 2 && kr <= 3) mgScore -= KING_CENTER_PENALTY;
    }
    {
        int ksq = board.kingSq[BLACK];
        mgScore -= popcount(Attacks::kingAttacks[ksq] & board.pieces[BLACK][PAWN]) * PAWN_SHIELD_BONUS;
        int kf = fileOf(ksq), kr = rankOf(ksq);
        if (kf >= 2 && kf <= 3 && kr >= 2 && kr <= 3) mgScore += KING_CENTER_PENALTY;
    }
    
    // Tapered evaluation
    if (phase > TOTAL_PHASE) phase = TOTAL_PHASE;
    float score = (mgScore * phase + egScore * (TOTAL_PHASE - phase)) / (float)TOTAL_PHASE;
    
    // Convert to int for search compatibility
    int intScore = (int)score;
    return (board.side == WHITE) ? intScore : -intScore;
}

inline int evaluate(const Board& board) {
    if (board.isInsufficientMaterial()) return 0;
    return evaluateClassical(board);
}

} // namespace Eval
