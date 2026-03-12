// ============================================================
// Endgame Tablebase Generator for 6x6 Chess
// 
// Usage: ./egtb_gen [max_pieces] [output_dir]
//   max_pieces: 3, 4, or 5 (default: 3)
//   output_dir: directory for output files (default: egtb)
//
// Generates tables for all material configurations up to
// max_pieces total pieces (including both kings).
// Saves to output directory, one file per configuration.
//
// Tables are generated bottom-up (3-piece first, then 4, etc.)
// and kept in memory for cross-table lookups when captures or
// promotions change the material configuration.
// ============================================================

#include "types.h"
#include "attacks.h"
#include "zobrist.h"
#include "board.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cassert>
#include <chrono>
#include <map>
#include <sys/stat.h>
#include <algorithm>
#include <functional>

// ============================================================
// EGTB value encoding (1 byte per position)
// ============================================================
// Values from WHITE's perspective:
//   0          = ILLEGAL / uninitialized
//   1          = DRAW
//   2 .. 127   = White wins, DTM = value - 2 (so 2 = mate in 0 = checkmate for black)
//   128        = DRAW (alternate, unused)
//   129 .. 255 = Black wins, DTM = value - 129

constexpr uint8_t TB_ILLEGAL   = 0;
constexpr uint8_t TB_DRAW      = 1;
constexpr uint8_t TB_UNKNOWN   = 255; // temporary during generation

inline uint8_t tbWWin(int dtm) { return (uint8_t)(2 + std::min(dtm, 125)); }
inline uint8_t tbBWin(int dtm) { return (uint8_t)(129 + std::min(dtm, 125)); }

inline bool isTbWWin(uint8_t v) { return v >= 2 && v <= 127; }
inline bool isTbBWin(uint8_t v) { return v >= 129 && v <= 254; }
inline bool isTbDraw(uint8_t v) { return v == TB_DRAW; }
inline bool isTbResolved(uint8_t v) { return v != TB_UNKNOWN && v != TB_ILLEGAL; }

inline int tbDTM(uint8_t v) {
    if (isTbWWin(v)) return v - 2;
    if (isTbBWin(v)) return v - 129;
    return 0;
}

inline bool isTbWinForSTM(uint8_t v, Color stm) {
    if (stm == WHITE) return isTbWWin(v);
    return isTbBWin(v);
}
inline bool isTbLossForSTM(uint8_t v, Color stm) {
    if (stm == WHITE) return isTbBWin(v);
    return isTbWWin(v);
}

// ============================================================
// Material configuration descriptor
// ============================================================
struct MaterialConfig {
    int pieces[2][PIECE_TYPE_COUNT]; // pieces[color][pieceType] count (PAWN..QUEEN)
    std::string name;
    int totalPieces; // including both kings
    
    MaterialConfig() {
        memset(pieces, 0, sizeof(pieces));
        totalPieces = 2;
    }
    
    void computeName() {
        const char* ptChar = "?PNBQ";
        name = "K";
        for (int pt = QUEEN; pt >= PAWN; pt--)
            for (int i = 0; i < pieces[WHITE][pt]; i++)
                name += ptChar[pt];
        name += "vK";
        for (int pt = QUEEN; pt >= PAWN; pt--)
            for (int i = 0; i < pieces[BLACK][pt]; i++)
                name += ptChar[pt];
    }
    
    void computeTotal() {
        totalPieces = 2;
        for (int c = 0; c < 2; c++)
            for (int pt = PAWN; pt <= QUEEN; pt++)
                totalPieces += pieces[c][pt];
    }
    
    int whiteExtras() const {
        int n = 0;
        for (int pt = PAWN; pt <= QUEEN; pt++) n += pieces[WHITE][pt];
        return n;
    }
    int blackExtras() const {
        int n = 0;
        for (int pt = PAWN; pt <= QUEEN; pt++) n += pieces[BLACK][pt];
        return n;
    }
    
    bool isCanonical() const {
        int wExtra = whiteExtras();
        int bExtra = blackExtras();
        if (wExtra > bExtra) return true;
        if (wExtra < bExtra) return false;
        for (int pt = QUEEN; pt >= PAWN; pt--) {
            if (pieces[WHITE][pt] > pieces[BLACK][pt]) return true;
            if (pieces[WHITE][pt] < pieces[BLACK][pt]) return false;
        }
        return true;
    }
    
    bool operator==(const MaterialConfig& o) const {
        return memcmp(pieces, o.pieces, sizeof(pieces)) == 0;
    }
};

// ============================================================
// Enumerate all material configurations for a given piece count
// ============================================================
void enumerateConfigs(int maxPieces, std::vector<MaterialConfig>& configs) {
    for (int total = 3; total <= maxPieces; total++) {
        int extras = total - 2;
        // Slots: WP, WN, WB, WQ, BP, BN, BB, BQ
        int maxPerSlot[8] = {6, 2, 2, 1, 6, 2, 2, 1};
        
        std::function<void(int, int, int[8])> enumerate2 = [&](int remaining, int minSlot, int counts[8]) {
            if (remaining == 0) {
                MaterialConfig cfg;
                cfg.pieces[WHITE][PAWN]   = counts[0];
                cfg.pieces[WHITE][KNIGHT] = counts[1];
                cfg.pieces[WHITE][BISHOP] = counts[2];
                cfg.pieces[WHITE][QUEEN]  = counts[3];
                cfg.pieces[BLACK][PAWN]   = counts[4];
                cfg.pieces[BLACK][KNIGHT] = counts[5];
                cfg.pieces[BLACK][BISHOP] = counts[6];
                cfg.pieces[BLACK][QUEEN]  = counts[7];
                cfg.computeTotal();
                cfg.computeName();
                if (cfg.isCanonical()) {
                    configs.push_back(cfg);
                }
                return;
            }
            
            for (int slot = minSlot; slot < 8; slot++) {
                if (counts[slot] < maxPerSlot[slot]) {
                    counts[slot]++;
                    enumerate2(remaining - 1, slot, counts);
                    counts[slot]--;
                }
            }
        };
        
        int counts[8] = {0};
        enumerate2(extras, 0, counts);
    }
}

// ============================================================
// Position indexing for a given material configuration
// ============================================================

struct TableIndex {
    struct PieceDesc {
        Color color;
        PieceType type;
        int numSquares; // 36 for non-pawns, 24 for pawns (ranks 2-5)
    };
    
    std::vector<PieceDesc> extraPieces;
    size_t totalPositions;
    
    void init(const MaterialConfig& cfg) {
        extraPieces.clear();
        for (int c = 0; c < 2; c++) {
            for (int pt = PAWN; pt <= QUEEN; pt++) {
                for (int k = 0; k < cfg.pieces[c][pt]; k++) {
                    PieceDesc pd;
                    pd.color = Color(c);
                    pd.type = PieceType(pt);
                    pd.numSquares = (pt == PAWN) ? 24 : 36;
                    extraPieces.push_back(pd);
                }
            }
        }
        totalPositions = 36ULL * 36;
        for (auto& pd : extraPieces)
            totalPositions *= pd.numSquares;
        totalPositions *= 2;
    }
    
    static int pawnIdxToSquare(int idx) { return idx + 6; }
    static int squareToPawnIdx(int sq)  { return sq - 6; }
    static bool isValidPawnSquare(int sq) {
        int r = rankOf(sq);
        return r >= 1 && r <= 4;
    }
    
    size_t encode(int wkSq, int bkSq, const int extraSq[], Color stm) const {
        size_t idx = wkSq;
        idx = idx * 36 + bkSq;
        for (int i = 0; i < (int)extraPieces.size(); i++) {
            int sq = extraSq[i];
            int sqIdx = (extraPieces[i].type == PAWN) ? squareToPawnIdx(sq) : sq;
            idx = idx * extraPieces[i].numSquares + sqIdx;
        }
        idx = idx * 2 + (stm == BLACK ? 1 : 0);
        return idx;
    }
    
    void decode(size_t idx, int& wkSq, int& bkSq, int extraSq[], Color& stm) const {
        stm = (idx % 2 == 1) ? BLACK : WHITE;
        idx /= 2;
        for (int i = (int)extraPieces.size() - 1; i >= 0; i--) {
            int ns = extraPieces[i].numSquares;
            int sqIdx = idx % ns;
            idx /= ns;
            extraSq[i] = (extraPieces[i].type == PAWN) ? pawnIdxToSquare(sqIdx) : sqIdx;
        }
        bkSq = idx % 36;
        idx /= 36;
        wkSq = idx;
    }
    
    bool isValidPosition(int wkSq, int bkSq, const int extraSq[]) const {
        if (wkSq == bkSq) return false;
        if (Attacks::kingAttacks[wkSq] & sqBit(bkSq)) return false;
        Bitboard occ = sqBit(wkSq) | sqBit(bkSq);
        for (int i = 0; i < (int)extraPieces.size(); i++) {
            if (occ & sqBit(extraSq[i])) return false;
            occ |= sqBit(extraSq[i]);
        }
        return true;
    }
    
    void setupBoard(Board& board, int wkSq, int bkSq, const int extraSq[], Color stm,
                    const MaterialConfig& cfg) const {
        board.clear();
        board.putPiece(WHITE, KING, wkSq);
        board.putPiece(BLACK, KING, bkSq);
        for (int i = 0; i < (int)extraPieces.size(); i++) {
            board.putPiece(extraPieces[i].color, extraPieces[i].type, extraSq[i]);
        }
        board.side = stm;
        if (stm == BLACK) board.hash ^= Zobrist::sideKey;
        
        // Use RoboGambit starting counts so the promotion rule works correctly.
        // The Board will know which pieces have been "captured" relative to the
        // full starting army (6P, 2N, 2B, 1Q, 1K per side) and can generate
        // the right promotion moves.
        board.setRoboGambitInitialCounts();
        board.computeCapturedFromInitial();
    }
    
    // Extract extra piece squares from a board state (after a move).
    // The board must have exactly the pieces described by extraPieces.
    // Returns false if the pieces don't match (e.g. capture changed material).
    bool extractExtraSq(const Board& board, int outSq[]) const {
        for (int pi = 0; pi < (int)extraPieces.size(); pi++) {
            Color pc = extraPieces[pi].color;
            PieceType pt = extraPieces[pi].type;
            int countBefore = 0;
            for (int pj = 0; pj < pi; pj++) {
                if (extraPieces[pj].color == pc && extraPieces[pj].type == pt)
                    countBefore++;
            }
            Bitboard bb = board.pieces[pc][pt];
            for (int skip = 0; skip < countBefore; skip++) {
                if (bb == 0) return false;
                bb &= bb - 1;
            }
            if (bb == 0) return false;
            outSq[pi] = lsb(bb);
        }
        return true;
    }
};

// ============================================================
// Table cache: holds generated tables in memory for cross-table
// lookups during generation of larger tables.
// ============================================================
struct CachedTable {
    MaterialConfig cfg;
    TableIndex tidx;
    std::vector<uint8_t> data;
};

struct TableCache {
    std::map<std::string, CachedTable> tables;
    
    // Probe a cached table for a given board state.
    // Determines the material config from the board, looks up the table,
    // encodes the position, and returns the value.
    // Returns TB_UNKNOWN if the table is not in cache.
    uint8_t probe(const Board& board) const {
        // Build material config from board
        MaterialConfig cfg;
        for (int c = 0; c < 2; c++)
            for (int pt = PAWN; pt <= QUEEN; pt++)
                cfg.pieces[c][pt] = popcount(board.pieces[c][pt]);
        
        // Make canonical: if black has more pieces (or equal but "better"),
        // we need to flip colors to look up the canonical table.
        // Our tables store the canonical form where white >= black.
        cfg.computeTotal();
        cfg.computeName();
        
        bool needFlip = !cfg.isCanonical();
        
        if (needFlip) {
            // Swap white and black in the config
            MaterialConfig flipped;
            for (int pt = PAWN; pt <= QUEEN; pt++) {
                flipped.pieces[WHITE][pt] = cfg.pieces[BLACK][pt];
                flipped.pieces[BLACK][pt] = cfg.pieces[WHITE][pt];
            }
            flipped.computeTotal();
            flipped.computeName();
            cfg = flipped;
        }
        
        auto it = tables.find(cfg.name);
        if (it == tables.end()) return TB_UNKNOWN;
        
        const CachedTable& ct = it->second;
        
        int wkSq, bkSq;
        Color stm;
        
        if (needFlip) {
            wkSq = board.kingSq[BLACK];
            bkSq = board.kingSq[WHITE];
            stm = ~board.side;
        } else {
            wkSq = board.kingSq[WHITE];
            bkSq = board.kingSq[BLACK];
            stm = board.side;
        }
        
        // Extract extra piece squares in the order expected by the table
        int extraSq[8];
        for (int pi = 0; pi < (int)ct.tidx.extraPieces.size(); pi++) {
            Color pc = ct.tidx.extraPieces[pi].color;
            PieceType pt = ct.tidx.extraPieces[pi].type;
            
            // When color-flipped, WHITE pieces in the table correspond to
            // BLACK pieces on the board (and vice versa)
            Color boardColor = needFlip ? ~pc : pc;
            
            int countBefore = 0;
            for (int pj = 0; pj < pi; pj++) {
                if (ct.tidx.extraPieces[pj].color == pc && ct.tidx.extraPieces[pj].type == pt)
                    countBefore++;
            }
            Bitboard bb = board.pieces[boardColor][pt];
            for (int skip = 0; skip < countBefore; skip++) {
                if (bb == 0) return TB_UNKNOWN;
                bb &= bb - 1;
            }
            if (bb == 0) return TB_UNKNOWN;
            extraSq[pi] = lsb(bb);
        }
        
        size_t idx = ct.tidx.encode(wkSq, bkSq, extraSq, stm);
        if (idx >= ct.data.size()) return TB_UNKNOWN;
        
        uint8_t val = ct.data[idx];
        
        // If we flipped colors, flip the result (white wins <-> black wins)
        if (needFlip) {
            if (isTbWWin(val)) {
                val = tbBWin(tbDTM(val));
            } else if (isTbBWin(val)) {
                val = tbWWin(tbDTM(val));
            }
        }
        
        return val;
    }
    
    // Load a table from disk into cache
    bool loadFromDisk(const std::string& dir, const MaterialConfig& cfg) {
        std::string filename = dir + "/" + cfg.name + ".egtb";
        std::ifstream f(filename, std::ios::binary);
        if (!f) return false;
        
        uint32_t magic, version;
        f.read(reinterpret_cast<char*>(&magic), 4);
        f.read(reinterpret_cast<char*>(&version), 4);
        if (magic != 0x45475442 || version != 1) return false;
        
        // Skip piece config (48 bytes)
        f.seekg(8 + 48, std::ios::beg);
        uint64_t tp;
        f.read(reinterpret_cast<char*>(&tp), 8);
        
        CachedTable ct;
        ct.cfg = cfg;
        ct.tidx.init(cfg);
        ct.data.resize(tp);
        f.read(reinterpret_cast<char*>(ct.data.data()), tp);
        
        if (!f.good()) return false;
        
        tables[cfg.name] = std::move(ct);
        return true;
    }
    
    // Store a freshly-generated table into cache
    void store(const MaterialConfig& cfg, const TableIndex& tidx, const std::vector<uint8_t>& data) {
        CachedTable ct;
        ct.cfg = cfg;
        ct.tidx = tidx;
        ct.data = data;
        tables[cfg.name] = std::move(ct);
    }
    
    size_t memoryUsage() const {
        size_t total = 0;
        for (auto& [name, ct] : tables)
            total += ct.data.size();
        return total;
    }
};

static TableCache g_cache;

// ============================================================
// Generate one endgame table via retrograde analysis
// ============================================================
void generateTable(const MaterialConfig& cfg, const std::string& outputDir) {
    TableIndex tidx;
    tidx.init(cfg);
    
    std::cout << "  Generating " << cfg.name 
              << " (" << tidx.totalPositions << " positions, "
              << tidx.totalPositions / 1024 << " KB)..." << std::flush;
    
    auto startTime = std::chrono::steady_clock::now();
    
    // Allocate table
    std::vector<uint8_t> table(tidx.totalPositions, TB_UNKNOWN);
    
    int numExtras = (int)tidx.extraPieces.size();
    int extraSq[8];
    
    // ============================================================
    // Pass 1: Classify all positions
    // ============================================================
    
    Board board;
    size_t numLegal = 0, numCheckmate = 0, numStalemate = 0, numIllegal = 0;
    
    for (size_t pos = 0; pos < tidx.totalPositions; pos++) {
        int wkSq, bkSq;
        Color stm;
        tidx.decode(pos, wkSq, bkSq, extraSq, stm);
        
        if (!tidx.isValidPosition(wkSq, bkSq, extraSq)) {
            table[pos] = TB_ILLEGAL;
            numIllegal++;
            continue;
        }
        
        tidx.setupBoard(board, wkSq, bkSq, extraSq, stm, cfg);
        
        Color opponent = ~stm;
        if (board.isAttackedBy(board.kingSq[opponent], stm)) {
            table[pos] = TB_ILLEGAL;
            numIllegal++;
            continue;
        }
        
        Move moves[MAX_MOVES];
        int numMoves = board.generateMoves(moves);
        
        if (numMoves == 0) {
            if (board.inCheck()) {
                if (stm == WHITE) table[pos] = tbBWin(0);
                else              table[pos] = tbWWin(0);
                numCheckmate++;
            } else {
                table[pos] = TB_DRAW;
                numStalemate++;
            }
        } else {
            numLegal++;
        }
    }
    
    // ============================================================
    // Pass 2: Retrograde iteration
    // ============================================================
    
    int iteration = 0;
    bool changed = true;
    
    while (changed) {
        changed = false;
        iteration++;
        
        for (size_t pos = 0; pos < tidx.totalPositions; pos++) {
            if (table[pos] != TB_UNKNOWN) continue;
            
            int wkSq, bkSq;
            Color stm;
            tidx.decode(pos, wkSq, bkSq, extraSq, stm);
            
            tidx.setupBoard(board, wkSq, bkSq, extraSq, stm, cfg);
            
            Move moves[MAX_MOVES];
            int numMoves = board.generateMoves(moves);
            
            bool hasUnknown = false;
            bool allResolvedLoss = true;
            int bestWinDTM = 999;
            int worstLossDTM = 0;
            bool hasWin = false;
            
            for (int i = 0; i < numMoves; i++) {
                board.makeMove(moves[i]);
                
                int capturedPiece = board.undoStack[board.undoCount - 1].captured;
                PieceType promoType = movePromo(moves[i]);
                
                uint8_t childVal;
                
                bool materialChanged = (capturedPiece != 0) || (promoType != NONE);
                
                if (materialChanged) {
                    // Material changed: look up the resulting position in a
                    // different (smaller or same-size) table via the cache.
                    //
                    // After the move, the board has the new material config.
                    // If it's KvK (2 pieces), that's always a draw.
                    // Otherwise, probe the cache for the resulting config.
                    
                    int totalPiecesAfter = popcount(board.allOccupied);
                    
                    if (totalPiecesAfter <= 2) {
                        // KvK = always draw
                        childVal = TB_DRAW;
                    } else {
                        childVal = g_cache.probe(board);
                        // If TB_UNKNOWN here, the child table wasn't generated yet.
                        // With the pawn-last sort order this should not happen for
                        // legal promotion targets (e.g. KPvK->KQvK, KQPvK->KQNvK).
                        // Leave as TB_UNKNOWN so the parent position stays UNKNOWN
                        // until all children are resolved.
                    }
                } else {
                    // Normal quiet move: look up in same table
                    int newWkSq = board.kingSq[WHITE];
                    int newBkSq = board.kingSq[BLACK];
                    int newExtraSq[8];
                    
                    if (!tidx.extractExtraSq(board, newExtraSq)) {
                        // Shouldn't happen for non-material-changing moves
                        board.unmakeMove();
                        continue;
                    }
                    
                    size_t childIdx = tidx.encode(newWkSq, newBkSq, newExtraSq, board.side);
                    childVal = table[childIdx];
                }
                
                board.unmakeMove();
                
                if (childVal == TB_UNKNOWN) {
                    hasUnknown = true;
                    allResolvedLoss = false;
                    continue;
                }
                
                if (childVal == TB_ILLEGAL) {
                    continue;
                }
                
                // childVal is from white's perspective, opponent is STM in child
                if (isTbLossForSTM(childVal, ~stm)) {
                    hasWin = true;
                    int childDTM = tbDTM(childVal);
                    if (childDTM + 1 < bestWinDTM) {
                        bestWinDTM = childDTM + 1;
                    }
                } else if (isTbWinForSTM(childVal, ~stm)) {
                    int childDTM = tbDTM(childVal);
                    if (childDTM + 1 > worstLossDTM) {
                        worstLossDTM = childDTM + 1;
                    }
                } else {
                    // Draw
                    allResolvedLoss = false;
                }
            }
            
            if (hasWin) {
                if (stm == WHITE) table[pos] = tbWWin(bestWinDTM);
                else              table[pos] = tbBWin(bestWinDTM);
                changed = true;
            } else if (!hasUnknown && allResolvedLoss && numMoves > 0) {
                if (stm == WHITE) table[pos] = tbBWin(worstLossDTM);
                else              table[pos] = tbWWin(worstLossDTM);
                changed = true;
            } else if (!hasUnknown) {
                table[pos] = TB_DRAW;
                changed = true;
            }
        }
    }
    
    // Remaining UNKNOWN positions are draws
    size_t numDrawByDefault = 0;
    for (size_t pos = 0; pos < tidx.totalPositions; pos++) {
        if (table[pos] == TB_UNKNOWN) {
            table[pos] = TB_DRAW;
            numDrawByDefault++;
        }
    }
    
    // Count results
    size_t wWins = 0, bWins = 0, draws = 0, illegal = 0;
    int maxWDTM = 0, maxBDTM = 0;
    for (size_t pos = 0; pos < tidx.totalPositions; pos++) {
        if (table[pos] == TB_ILLEGAL) { illegal++; continue; }
        if (isTbWWin(table[pos])) { wWins++; maxWDTM = std::max(maxWDTM, tbDTM(table[pos])); }
        else if (isTbBWin(table[pos])) { bWins++; maxBDTM = std::max(maxBDTM, tbDTM(table[pos])); }
        else draws++;
    }
    
    auto endTime = std::chrono::steady_clock::now();
    int ms = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    
    std::cout << " done (" << ms << "ms, " << iteration << " iters)\n";
    std::cout << "    W wins: " << wWins << " (max DTM=" << maxWDTM << ")"
              << "  B wins: " << bWins << " (max DTM=" << maxBDTM << ")"
              << "  Draws: " << draws
              << "  Illegal: " << illegal << "\n";
    
    // Store in cache for use by larger tables
    g_cache.store(cfg, tidx, table);
    
    // Save to file
    std::string filename = outputDir + "/" + cfg.name + ".egtb";
    std::ofstream f(filename, std::ios::binary);
    if (!f) {
        std::cerr << "    ERROR: Cannot write to " << filename << "\n";
        return;
    }
    
    uint32_t magic = 0x45475442; // "EGTB"
    uint32_t version = 1;
    f.write(reinterpret_cast<const char*>(&magic), 4);
    f.write(reinterpret_cast<const char*>(&version), 4);
    f.write(reinterpret_cast<const char*>(&cfg.pieces), sizeof(cfg.pieces));
    uint64_t tp = tidx.totalPositions;
    f.write(reinterpret_cast<const char*>(&tp), 8);
    f.write(reinterpret_cast<const char*>(table.data()), table.size());
    
    if (f.good()) {
        std::cout << "    Saved: " << filename << " (" << table.size() / 1024 << " KB)\n";
    } else {
        std::cerr << "    ERROR: Write failed for " << filename << "\n";
    }
}

// ============================================================
// Main
// ============================================================
void Board::print() const {
    const char* pieceChars = ".PNBQKpnbqk";
    std::cout << "\n  A B C D E F\n";
    for (int r = 5; r >= 0; r--) {
        std::cout << (r + 1) << " ";
        for (int f = 0; f < 6; f++) {
            int sq = makeSquare(f, r);
            int piece = mailbox[sq];
            if (piece == 0) { std::cout << ". "; }
            else {
                Color c = pieceColor(piece);
                PieceType pt = pieceTypeFromId(piece);
                char ch = pieceChars[pt];
                if (c == BLACK) ch = pieceChars[pt + 5];
                std::cout << ch << " ";
            }
        }
        std::cout << (r + 1) << "\n";
    }
    std::cout << "  A B C D E F\n";
}

int main(int argc, char* argv[]) {
    int maxPieces = 3;
    std::string outputDir = "egtb";
    bool skipExisting = true;
    
    if (argc > 1) {
        maxPieces = std::atoi(argv[1]);
        if (maxPieces < 3) maxPieces = 3;
        if (maxPieces > 6) maxPieces = 6;
    }
    if (argc > 2) {
        outputDir = argv[2];
    }
    
    Attacks::init();
    Zobrist::init();
    
    std::cout << "=== EGTB Generator for 6x6 Chess ===\n";
    std::cout << "Max pieces: " << maxPieces << "\n";
    std::cout << "Output dir: " << outputDir << "\n\n";
    
    mkdir(outputDir.c_str(), 0755);
    
    // Enumerate ALL configurations up to maxPieces
    std::vector<MaterialConfig> configs;
    enumerateConfigs(maxPieces, configs);

    // Critical: sort so non-pawn tables come before pawn tables within each
    // piece count.  Pawn tables depend on promoted-piece tables (e.g. KPvK
    // needs KQvK in cache for pawn promotion probes).  Within the same total
    // piece count, fewer pawns first ensures dependencies are already cached.
    std::sort(configs.begin(), configs.end(), [](const MaterialConfig& a, const MaterialConfig& b) {
        if (a.totalPieces != b.totalPieces) return a.totalPieces < b.totalPieces;
        int aP = a.pieces[WHITE][PAWN] + a.pieces[BLACK][PAWN];
        int bP = b.pieces[WHITE][PAWN] + b.pieces[BLACK][PAWN];
        return aP < bP;
    });

    std::cout << "Total configurations: " << configs.size() << "\n\n";
    
    // Separate into existing (load into cache) and to-generate
    std::vector<MaterialConfig> toGenerate;
    size_t totalSize = 0;
    int skippedCount = 0;
    int loadedCount = 0;
    
    for (auto& cfg : configs) {
        TableIndex tidx;
        tidx.init(cfg);
        
        std::string filename = outputDir + "/" + cfg.name + ".egtb";
        struct stat st;
        if (skipExisting && stat(filename.c_str(), &st) == 0 && st.st_size > 0) {
            // Load existing table into cache (needed for cross-table lookups)
            if (g_cache.loadFromDisk(outputDir, cfg)) {
                std::cout << "  [loaded] " << cfg.name << " (" << st.st_size / 1024 << " KB)\n";
                loadedCount++;
            } else {
                std::cout << "  [skip/err] " << cfg.name << " (failed to load, will regenerate)\n";
                totalSize += tidx.totalPositions;
                toGenerate.push_back(cfg);
            }
            skippedCount++;
        } else {
            totalSize += tidx.totalPositions;
            std::cout << "  [new] " << cfg.name << ": " << tidx.totalPositions << " positions ("
                      << tidx.totalPositions / 1024 << " KB)\n";
            toGenerate.push_back(cfg);
        }
    }
    
    std::cout << "\nLoaded from disk: " << loadedCount << " tables (" 
              << g_cache.memoryUsage() / (1024*1024) << " MB in cache)\n";
    std::cout << "To generate: " << toGenerate.size() << " tables, "
              << totalSize / 1024 << " KB (" << totalSize / (1024*1024) << " MB)\n\n";
    
    if (toGenerate.empty()) {
        std::cout << "Nothing to generate!\n";
        return 0;
    }
    
    auto startTime = std::chrono::steady_clock::now();
    
    int completed = 0;
    for (auto& cfg : toGenerate) {
        completed++;
        std::cout << "[" << completed << "/" << toGenerate.size() << "] ";
        generateTable(cfg, outputDir);
        std::cout << "    Cache: " << g_cache.memoryUsage() / (1024*1024) << " MB\n";
    }
    
    auto endTime = std::chrono::steady_clock::now();
    int totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    
    std::cout << "\n=== Generation complete in " << totalMs / 1000 << "s ===\n";
    
    return 0;
}
