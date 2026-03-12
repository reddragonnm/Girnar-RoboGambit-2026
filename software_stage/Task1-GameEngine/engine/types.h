#pragma once
#include <cstdint>
#include <string>

// Board is 6x6, mapped into bits 0..35 of a uint64_t
// Bit index = row * 6 + col, where row 0 = rank 1, col 0 = file A
// So A1=0, B1=1, ..., F1=5, A2=6, ..., F6=35

using Bitboard = uint64_t;

constexpr int BOARD_SIZE = 6;
constexpr int NUM_SQUARES = 36;
constexpr Bitboard BOARD_MASK = (1ULL << 36) - 1;

// Piece types (match competition IDs for white; black = type + 5)
enum PieceType : int {
    NONE = 0,
    PAWN = 1,
    KNIGHT = 2,
    BISHOP = 3,
    QUEEN = 4,
    KING = 5,
    PIECE_TYPE_COUNT = 6
};

enum Color : int {
    WHITE = 0,
    BLACK = 1,
    COLOR_COUNT = 2
};

constexpr Color operator~(Color c) { return Color(c ^ 1); }

// Competition piece IDs: White 1-5, Black 6-10
// pieceId = (color == WHITE) ? type : type + 5
inline int pieceId(Color c, PieceType pt) {
    return (c == WHITE) ? pt : pt + 5;
}

inline Color pieceColor(int id) {
    return (id >= 1 && id <= 5) ? WHITE : BLACK;
}

inline PieceType pieceTypeFromId(int id) {
    if (id >= 1 && id <= 5) return PieceType(id);
    if (id >= 6 && id <= 10) return PieceType(id - 5);
    return NONE;
}

// Square utilities
constexpr int makeSquare(int file, int rank) { return rank * 6 + file; }
constexpr int fileOf(int sq) { return sq % 6; }
constexpr int rankOf(int sq) { return sq / 6; }

inline std::string squareToStr(int sq) {
    return std::string(1, 'A' + fileOf(sq)) + std::string(1, '1' + rankOf(sq));
}

inline int strToSquare(const std::string& s) {
    int file = s[0] - 'A';
    int rank = s[1] - '1';
    return makeSquare(file, rank);
}

// Bitboard utilities
constexpr Bitboard sqBit(int sq) { return 1ULL << sq; }

inline int popcount(Bitboard b) { return __builtin_popcountll(b); }

inline int lsb(Bitboard b) { return __builtin_ctzll(b); }

inline int popLsb(Bitboard& b) {
    int sq = lsb(b);
    b &= b - 1;
    return sq;
}

// Move encoding: 16 bits
// bits 0-5: from square (0-35)
// bits 6-11: to square (0-35)
// bits 12-14: promotion piece type (0 = none, 1-5 = piece type)
// bit 15: reserved
using Move = uint16_t;

constexpr Move MOVE_NONE = 0;

constexpr Move encodeMove(int from, int to, PieceType promo = NONE) {
    return Move(from | (to << 6) | (int(promo) << 12));
}

constexpr int moveFrom(Move m) { return m & 0x3F; }
constexpr int moveTo(Move m) { return (m >> 6) & 0x3F; }
constexpr PieceType movePromo(Move m) { return PieceType((m >> 12) & 0x7); }

inline std::string moveToStr(Move m, Color side, PieceType pt) {
    // Format: <piece_id>:<source>-><target>[=<promoted_id>]
    int id = pieceId(side, pt);
    std::string s = std::to_string(id) + ":" + squareToStr(moveFrom(m)) + "->" + squareToStr(moveTo(m));
    PieceType promo = movePromo(m);
    if (promo != NONE) {
        s += "=" + std::to_string(pieceId(side, promo));
    }
    return s;
}

// Score constants
// Keep all scores within int16_t range [-32768, 32767] so TT stores them without truncation.
// PST evaluations stay well below ±10000, leaving 10000-28000 as mate range.
constexpr int INF_SCORE = 32000;
constexpr int MATE_SCORE = 30000;
constexpr int MATE_THRESHOLD = 28000;
constexpr int DRAW_SCORE = 0;

// Max moves in a position (generous upper bound)
constexpr int MAX_MOVES = 128;
