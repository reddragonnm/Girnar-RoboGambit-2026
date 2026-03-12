#pragma once
#include "types.h"
#include <array>
#include <cstring>
// Portable PEXT (Parallel Bit Extract) implementation.
// Uses hardware BMI2 intrinsic on x86 when available, otherwise falls back
// to a software implementation that works on any architecture (e.g. ARM/Apple
// Silicon).
#ifdef __BMI2__
#include <immintrin.h>
inline uint64_t pext_u64(uint64_t src, uint64_t mask) {
  return _pext_u64(src, mask);
}
#else
inline uint64_t pext_u64(uint64_t src, uint64_t mask) {
  uint64_t result = 0;
  for (uint64_t bit = 1; mask; bit <<= 1) {
    if (src & mask & -mask)
      result |= bit;
    mask &= mask - 1; // clear lowest set bit
  }
  return result;
}
#endif

// Precomputed attack tables for the 6x6 board
namespace Attacks {

// Knight attacks from each square
Bitboard knightAttacks[NUM_SQUARES];

// King attacks from each square
Bitboard kingAttacks[NUM_SQUARES];

// Pawn attacks from each square for each color
Bitboard pawnAttacks[COLOR_COUNT][NUM_SQUARES];

// Direction offsets
constexpr int KNIGHT_MOVES[][2] = {{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
                                   {1, -2},  {1, 2},  {2, -1},  {2, 1}};

constexpr int KING_MOVES[][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
                                 {0, 1},   {1, -1}, {1, 0},  {1, 1}};

inline bool validSquare(int file, int rank) {
  return file >= 0 && file < 6 && rank >= 0 && rank < 6;
}

// ============================================================
// PEXT-based sliding attack tables for 6x6 board
// ============================================================
// For each square and each ray direction, we store:
//   - A mask of all squares on the ray
//   - A lookup table indexed by PEXT(occupancy, mask)
// PEXT extracts the bits corresponding to the mask into a contiguous index.
// Max ray length is 5, so max 2^5=32 entries per ray.

struct RayData {
  Bitboard mask;        // mask of squares on this ray
  Bitboard attacks[32]; // attacks indexed by PEXT(occ, mask)
};

// 36 squares * 8 directions
RayData rays[NUM_SQUARES][8];

// Direction vectors: 0=(-1,-1), 1=(-1,0), 2=(-1,1), 3=(0,-1), 4=(0,1),
// 5=(1,-1), 6=(1,0), 7=(1,1)
constexpr int dirVecs[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
                               {0, 1},   {1, -1}, {1, 0},  {1, 1}};

// Bishop uses directions 0,2,5,7 (diagonals)
constexpr int BISHOP_DIR_INDICES[] = {0, 2, 5, 7};
// Queen uses all 8 directions

inline Bitboard bishopAttacks(int sq, Bitboard occupied) {
  Bitboard attacks = 0;
  for (int i = 0; i < 4; i++) {
    int dir = BISHOP_DIR_INDICES[i];
    const RayData &rd = rays[sq][dir];
    if (rd.mask) {
      int idx = (int)pext_u64(occupied, rd.mask);
      attacks |= rd.attacks[idx];
    }
  }
  return attacks;
}

inline Bitboard queenAttacks(int sq, Bitboard occupied) {
  Bitboard attacks = 0;
  for (int dir = 0; dir < 8; dir++) {
    const RayData &rd = rays[sq][dir];
    if (rd.mask) {
      int idx = (int)pext_u64(occupied, rd.mask);
      attacks |= rd.attacks[idx];
    }
  }
  return attacks;
}

// File masks
Bitboard fileMask[6];
// Rank masks
Bitboard rankMask[6];
// Adjacent file masks (for isolated pawn detection)
Bitboard adjacentFileMask[6];
// Passed pawn masks: passedPawnMask[color][square]
Bitboard passedPawnMask[COLOR_COUNT][NUM_SQUARES];

inline void initRayTables() {
  for (int sq = 0; sq < NUM_SQUARES; sq++) {
    int f = fileOf(sq), r = rankOf(sq);

    for (int dir = 0; dir < 8; dir++) {
      RayData &rd = rays[sq][dir];
      rd.mask = 0;

      int df = dirVecs[dir][0], dr = dirVecs[dir][1];

      // Collect ray squares in walking order (from piece outward)
      int raySq[5];
      int rayLen = 0;
      int nf = f + df, nr = r + dr;
      while (validSquare(nf, nr) && rayLen < 5) {
        raySq[rayLen] = makeSquare(nf, nr);
        rd.mask |= sqBit(raySq[rayLen]);
        rayLen++;
        nf += df;
        nr += dr;
      }

      if (rayLen == 0) {
        memset(rd.attacks, 0, sizeof(rd.attacks));
        continue;
      }

      // Enumerate all subsets of the mask using Carry-Rippler trick
      // For each subset, compute the actual sliding attacks by walking the ray
      // and checking which squares are occupied.
      // The index into rd.attacks is PEXT(subset, mask).
      memset(rd.attacks, 0, sizeof(rd.attacks));

      Bitboard mask = rd.mask;
      Bitboard subset = 0;
      do {
        // Compute attacks for this occupancy pattern
        Bitboard att = 0;
        for (int i = 0; i < rayLen; i++) {
          att |= sqBit(raySq[i]);
          if (subset & sqBit(raySq[i]))
            break; // blocked
        }

        int idx = (int)pext_u64(subset, mask);
        rd.attacks[idx] = att;

        subset = (subset - mask) & mask; // Carry-Rippler
      } while (subset);
    }
  }
}

inline void initPawnMasks() {
  for (int f = 0; f < 6; f++) {
    fileMask[f] = 0;
    for (int r = 0; r < 6; r++)
      fileMask[f] |= sqBit(makeSquare(f, r));
  }

  for (int r = 0; r < 6; r++) {
    rankMask[r] = 0;
    for (int f = 0; f < 6; f++)
      rankMask[r] |= sqBit(makeSquare(f, r));
  }

  for (int f = 0; f < 6; f++) {
    adjacentFileMask[f] = 0;
    if (f > 0)
      adjacentFileMask[f] |= fileMask[f - 1];
    if (f < 5)
      adjacentFileMask[f] |= fileMask[f + 1];
  }

  for (int c = 0; c < COLOR_COUNT; c++) {
    for (int sq = 0; sq < NUM_SQUARES; sq++) {
      int f = fileOf(sq);
      int r = rankOf(sq);
      passedPawnMask[c][sq] = 0;

      for (int ff = std::max(0, f - 1); ff <= std::min(5, f + 1); ff++) {
        if (c == WHITE) {
          for (int rr = r + 1; rr < 6; rr++)
            passedPawnMask[c][sq] |= sqBit(makeSquare(ff, rr));
        } else {
          for (int rr = r - 1; rr >= 0; rr--)
            passedPawnMask[c][sq] |= sqBit(makeSquare(ff, rr));
        }
      }
    }
  }
}

inline void init() {
  for (int sq = 0; sq < NUM_SQUARES; sq++) {
    int f = fileOf(sq), r = rankOf(sq);

    // Knight
    knightAttacks[sq] = 0;
    for (auto &km : KNIGHT_MOVES) {
      int nf = f + km[0], nr = r + km[1];
      if (validSquare(nf, nr))
        knightAttacks[sq] |= sqBit(makeSquare(nf, nr));
    }

    // King
    kingAttacks[sq] = 0;
    for (auto &km : KING_MOVES) {
      int nf = f + km[0], nr = r + km[1];
      if (validSquare(nf, nr))
        kingAttacks[sq] |= sqBit(makeSquare(nf, nr));
    }

    // Pawn attacks
    pawnAttacks[WHITE][sq] = 0;
    if (r < 5) {
      if (f > 0)
        pawnAttacks[WHITE][sq] |= sqBit(makeSquare(f - 1, r + 1));
      if (f < 5)
        pawnAttacks[WHITE][sq] |= sqBit(makeSquare(f + 1, r + 1));
    }

    pawnAttacks[BLACK][sq] = 0;
    if (r > 0) {
      if (f > 0)
        pawnAttacks[BLACK][sq] |= sqBit(makeSquare(f - 1, r - 1));
      if (f < 5)
        pawnAttacks[BLACK][sq] |= sqBit(makeSquare(f + 1, r - 1));
    }
  }

  initRayTables();
  initPawnMasks();
}

} // namespace Attacks
