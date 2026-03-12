/**
 * EGTB Validity Checker for chess6x6
 *
 * Compile:  make validate_egtb
 * Run:      ./validate_egtb [egtb_dir]   (default: egtb)
 *
 * Checks performed for every table:
 *   1. File magic, version, header integrity, file-size match
 *   2. Value distribution (illegal / draw / wwin / bwin / unknown)
 *   3. Kings-adjacent / same-square sanity on non-illegal positions
 *   4. DTM consistency via full move generation:
 *        WIN(D)  : ∃ child LOSS(D-1)
 *        LOSS(D) : ∀ children are WIN;  max(child DTM) = D-1
 *        DRAW    : ∃ child DRAW  OR  stalemate
 *   5. Self-consistency: re-encoding the decoded position returns the same index
 */

#include "board.h"
#include "egtb.h"
#include "attacks.h"
#include "zobrist.h"
#include "types.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <cstring>

// =========================================================================
// Helpers
// =========================================================================

// Decode a flat table index back to (wkSq, bkSq, extraSq[], stm).
// Mirrors the encode() in EGTBTable / egtb_gen.cpp.
static void decodeIdx(const EGTBTable& tbl,
                      size_t            idx,
                      int&              wkSq,
                      int&              bkSq,
                      int               extraSq[8],
                      Color&            stm)
{
    stm = (idx & 1) ? BLACK : WHITE;
    idx >>= 1;
    for (int i = (int)tbl.extraPieces.size() - 1; i >= 0; --i) {
        int ns    = tbl.extraPieces[i].numSquares;
        int sqIdx = (int)(idx % ns);
        idx /= ns;
        extraSq[i] = (tbl.extraPieces[i].type == PAWN)
                     ? EGTBTable::pawnIdxToSquare(sqIdx)
                     : sqIdx;
    }
    bkSq  = (int)(idx % 36);
    idx  /= 36;
    wkSq  = (int)(idx % 36);
}

// Build a playable Board from decoded piece positions.
// We assign generous initialCount so the engine's promotion guard never
// blocks a legal promoting move.
static Board buildBoard(int wkSq, int bkSq,
                        const EGTBTable& tbl, const int extraSq[8],
                        Color stm)
{
    Board b;
    b.clear();

    b.putPiece(WHITE, KING, wkSq);
    b.putPiece(BLACK, KING, bkSq);

    for (int i = 0; i < (int)tbl.extraPieces.size(); ++i) {
        Color     c  = tbl.extraPieces[i].color;
        PieceType pt = tbl.extraPieces[i].type;
        b.putPiece(c, pt, extraSq[i]);
    }

    // Allow promotion to any piece type (set initial counts high enough)
    for (int c = 0; c < 2; ++c)
        for (int pt = PAWN; pt <= QUEEN; ++pt)
            b.initialCount[c][pt] = 4;

    b.side = stm;
    if (stm == BLACK) b.hash ^= Zobrist::sideKey;
    return b;
}

// Are two squares "adjacent" (king-adjacent, including diagonals)?
static bool kingsAdjacent(int a, int b) {
    int dr = std::abs(rankOf(a) - rankOf(b));
    int df = std::abs(fileOf(a) - fileOf(b));
    return dr <= 1 && df <= 1;
}

// =========================================================================
// Per-table validation
// =========================================================================

struct TableReport {
    std::string name;
    size_t      totalPos = 0;
    size_t      nIllegal = 0, nDraw = 0, nWWin = 0, nBWin = 0, nUnknown = 0;
    size_t      maxDTM   = 0;

    // Errors
    size_t errWinNoLossChild  = 0; // WIN(D) but no child is LOSS(D-1)
    size_t errLossNonWinChild = 0; // LOSS(D) but some child is not WIN
    size_t errLossMaxDTM      = 0; // LOSS(D) but max(child WIN DTM) ≠ D-1
    size_t errDrawBadChild    = 0; // DRAW but no draw-escape and not stalemate
    size_t errKingsAdj        = 0; // non-illegal pos but kings are adjacent
    size_t errSameSquare      = 0; // kings on same square
    size_t errReEncode        = 0; // re-encode ≠ original index

    size_t posChecked = 0;
};

static TableReport validateTable(const std::string& name,
                                  const EGTBTable&   tbl,
                                  EGTB&              allTables,
                                  size_t             sampleLimit)
{
    TableReport r;
    r.name     = name;
    r.totalPos = tbl.totalPositions;

    // Determine step for sampling (check at most sampleLimit positions)
    size_t step = 1;
    if (sampleLimit > 0 && tbl.totalPositions > sampleLimit)
        step = tbl.totalPositions / sampleLimit;

    for (size_t idx = 0; idx < tbl.totalPositions; idx += step) {
        uint8_t val = tbl.data[idx];

        // Distribution counts (use raw-value approach, always count)
        if (val == TB_ILLEGAL)        ++r.nIllegal;
        else if (val == TB_DRAW)      ++r.nDraw;
        else if (isTbWWin(val))       { ++r.nWWin;  r.maxDTM = std::max(r.maxDTM, (size_t)tbDTM(val)); }
        else if (isTbBWin(val))       { ++r.nBWin;  r.maxDTM = std::max(r.maxDTM, (size_t)tbDTM(val)); }
        else if (val == TB_UNKNOWN)   ++r.nUnknown;

        // Only run DTM/board checks on sampled positions
        if (step == 1 || (idx % step) == 0) {
            ++r.posChecked;

            if (val == TB_ILLEGAL || val == TB_UNKNOWN)
                continue; // nothing to validate

            // --- Decode position ---
            int   extraSq[8] = {};
            int   wkSq, bkSq;
            Color stm;
            decodeIdx(tbl, idx, wkSq, bkSq, extraSq, stm);

            // --- Self-consistency: re-encode ---
            size_t reIdx = tbl.encode(wkSq, bkSq, extraSq, stm);
            if (reIdx != idx) {
                ++r.errReEncode;
                continue;
            }

            // --- Kings checks ---
            if (wkSq == bkSq)                  { ++r.errSameSquare; continue; }
            if (kingsAdjacent(wkSq, bkSq))     { ++r.errKingsAdj;   continue; }

            // --- Build board and generate moves ---
            Board b = buildBoard(wkSq, bkSq, tbl, extraSq, stm);
            Move  moves[MAX_MOVES];
            int   nMoves = b.generateMoves(moves);

            // --- DTM consistency checks ---
            if (isTbWWin(val) || isTbBWin(val)) {
                // Determine if STM wins or loses
                bool stmWins;
                if (isTbWWin(val)) stmWins = (stm == WHITE);
                else               stmWins = (stm == BLACK);

                int dtm = tbDTM(val);

                if (stmWins) {
                    // WIN(D): need at least one child that is LOSS(D-1)
                    bool foundLossChild = false;
                    for (int i = 0; i < nMoves && !foundLossChild; ++i) {
                        b.makeMove(moves[i]);
                        TBProbeResult child = allTables.probe(b);
                        b.unmakeMove();

                        if (child.result == TB_RESULT_LOSS && child.dtm == dtm - 1)
                            foundLossChild = true;
                    }
                    if (!foundLossChild) ++r.errWinNoLossChild;

                } else {
                    // LOSS(D): all children must be WIN; max(child WIN DTM) = D-1
                    int  maxChildDTM    = -1;
                    bool allChildrenWin = true;

                    for (int i = 0; i < nMoves; ++i) {
                        b.makeMove(moves[i]);
                        TBProbeResult child = allTables.probe(b);
                        b.unmakeMove();

                        if (child.result == TB_RESULT_WIN) {
                            maxChildDTM = std::max(maxChildDTM, child.dtm);
                        } else if (child.result != TB_RESULT_NONE) {
                            // DRAW or LOSS child — illegal for a pure LOSS position
                            allChildrenWin = false;
                        }
                        // TB_RESULT_NONE: position left our EGTB coverage (different material); skip
                    }

                    if (!allChildrenWin)            ++r.errLossNonWinChild;
                    else if (nMoves > 0 && maxChildDTM != dtm - 1)
                                                    ++r.errLossMaxDTM;
                }

            } else if (val == TB_DRAW) {
                // DRAW: either stalemate OR at least one child is also DRAW
                if (nMoves == 0) {
                    // Stalemate — board correctly returns 0 legal moves, not in check
                    // (if king is in check with 0 moves that's checkmate = LOSS, but table says DRAW → error)
                    if (b.inCheck()) ++r.errDrawBadChild;
                } else {
                    bool foundDrawEscape = false;
                    for (int i = 0; i < nMoves && !foundDrawEscape; ++i) {
                        b.makeMove(moves[i]);
                        TBProbeResult child = allTables.probe(b);
                        b.unmakeMove();
                        if (child.result == TB_RESULT_DRAW) foundDrawEscape = true;
                    }
                    if (!foundDrawEscape) ++r.errDrawBadChild;
                }
            }
        }
    }

    return r;
}

// =========================================================================
// File-level format checks (separate from board logic)
// =========================================================================

struct FileReport {
    std::string path;
    bool        exists       = false;
    bool        magicOk      = false;
    bool        versionOk    = false;
    bool        sizeOk       = false;
    uint64_t    declaredPos  = 0;
    uint64_t    actualBytes  = 0;
    std::string error;
};

static FileReport checkFile(const std::string& path) {
    FileReport r;
    r.path = path;

    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { r.error = "cannot open"; return r; }
    r.exists     = true;
    r.actualBytes = (uint64_t)f.tellg();
    f.seekg(0);

    uint32_t magic = 0, version = 0;
    f.read(reinterpret_cast<char*>(&magic),   4);
    f.read(reinterpret_cast<char*>(&version), 4);

    r.magicOk   = (magic   == 0x45475442u); // "EGTB"
    r.versionOk = (version == 1u);

    // Header: 4+4+48+8 = 64 bytes, then totalPos bytes of data
    int filePieces[2][PIECE_TYPE_COUNT];
    f.read(reinterpret_cast<char*>(filePieces), sizeof(filePieces));
    uint64_t totalPos = 0;
    f.read(reinterpret_cast<char*>(&totalPos), 8);

    r.declaredPos = totalPos;
    r.sizeOk      = (r.actualBytes == 64 + totalPos);

    if (!r.magicOk)   r.error = "bad magic";
    else if (!r.versionOk) r.error = "bad version";
    else if (!r.sizeOk)    r.error = "size mismatch (expected " +
                                      std::to_string(64 + totalPos) +
                                      " got " + std::to_string(r.actualBytes) + ")";
    return r;
}

// =========================================================================
// main
// =========================================================================

int main(int argc, char* argv[]) {
    const std::string dir = (argc > 1) ? argv[1] : "egtb";

    // 1. Initialise engine subsystems
    Attacks::init();
    Zobrist::init();

    std::cout << "=== EGTB Validity Checker ===\n";
    std::cout << "Directory: " << dir << "\n\n";

    // 2. Collect all .egtb files
    std::vector<std::string> files;
    if (!std::filesystem::exists(dir)) {
        std::cerr << "ERROR: directory does not exist: " << dir << "\n";
        return 1;
    }
    for (auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() == ".egtb")
            files.push_back(entry.path().string());
    }
    std::sort(files.begin(), files.end());
    std::cout << "Found " << files.size() << " .egtb files\n\n";

    // 3. File-format checks (independent of engine loading)
    std::cout << "--- File Format Checks ---\n";
    int fmtErrors = 0;
    for (auto& p : files) {
        FileReport fr = checkFile(p);
        std::string base = std::filesystem::path(p).filename().string();
        if (!fr.magicOk || !fr.versionOk || !fr.sizeOk) {
            ++fmtErrors;
            std::cout << "  FAIL  " << base << "  " << fr.error << "\n";
        } else {
            std::cout << "  OK    " << base
                      << "  (" << fr.declaredPos << " positions)\n";
        }
    }
    if (fmtErrors == 0) std::cout << "  All files pass format check.\n";
    std::cout << "\n";

    // 4. Load all tables via the engine's EGTB loader
    EGTB egtb;
    bool loaded = egtb.loadAll(dir);
    std::cout << "--- Load Summary ---\n";
    std::cout << "  Tables loaded : " << egtb.tables.size() << "\n";
    std::cout << "  TableMap keys : " << egtb.tableMap.size()
              << "  (includes flipped keys)\n";
    std::cout << "  Total positions: " << egtb.totalPositions() << "\n\n";

    if (!loaded) {
        std::cerr << "ERROR: loadAll() returned false – no tables loaded.\n";
        return 1;
    }

    // 5. Per-table DTM consistency checks
    // Limit: check every position in ≤4-piece tables; sample 500K for 5-piece.
    constexpr size_t SAMPLE_5PIECE = 500000;
    std::cout << "--- DTM Consistency Checks ---\n";
    std::cout << "  (5-piece tables sampled to " << SAMPLE_5PIECE << " positions)\n\n";

    // We need to map filename → table index.  Build a name→tableRef map.
    // The tables vector is in load order; tableMap keys include flipped copies.
    // Build a set of table indices we've already reported.
    std::vector<bool> reported(egtb.tables.size(), false);

    size_t totalErrors = 0;

    // We enumerate via the same generateMaterialConfigs that EGTB::loadAll uses.
    // Simplest: iterate over egtb.tableMap, skip flipped entries.
    for (auto& [key, ref] : egtb.tableMap) {
        if (ref.flipped) continue; // each table appears twice in map; skip alias

        const EGTBTable& tbl  = egtb.tables[ref.tableIdx];
        // Derive name from key
        const char* ptN = "?PNBQ";
        std::string name = "K";
        for (int pt = 4; pt >= 1; --pt)
            for (int i = 0; i < key.pieces[0][pt]; ++i) name += ptN[pt];
        name += "vK";
        for (int pt = 4; pt >= 1; --pt)
            for (int i = 0; i < key.pieces[1][pt]; ++i) name += ptN[pt];

        size_t limit = (tbl.totalPositions > 500000) ? SAMPLE_5PIECE : 0;
        TableReport tr = validateTable(name, tbl, egtb, limit);

        // Print header
        std::cout << std::left << std::setw(18) << tr.name;
        std::cout << " pos=" << std::setw(10) << tr.totalPos
                  << " chk=" << std::setw(8)  << tr.posChecked
                  << " maxDTM=" << std::setw(4) << tr.maxDTM
                  << " dist(I/Dr/WW/BW)="
                  << tr.nIllegal << "/" << tr.nDraw  << "/"
                  << tr.nWWin   << "/" << tr.nBWin;

        size_t errs = tr.errWinNoLossChild + tr.errLossNonWinChild +
                      tr.errLossMaxDTM     + tr.errDrawBadChild    +
                      tr.errKingsAdj       + tr.errSameSquare       +
                      tr.errReEncode;
        totalErrors += errs;

        if (errs == 0) {
            std::cout << "  ✓\n";
        } else {
            std::cout << "\n    ERRORS:\n";
            if (tr.errWinNoLossChild)
                std::cout << "      WIN with no LOSS(D-1) child    : " << tr.errWinNoLossChild << "\n";
            if (tr.errLossNonWinChild)
                std::cout << "      LOSS with non-WIN child        : " << tr.errLossNonWinChild << "\n";
            if (tr.errLossMaxDTM)
                std::cout << "      LOSS DTM mismatch              : " << tr.errLossMaxDTM << "\n";
            if (tr.errDrawBadChild)
                std::cout << "      DRAW with no draw-escape       : " << tr.errDrawBadChild << "\n";
            if (tr.errKingsAdj)
                std::cout << "      Kings adjacent in non-illegal  : " << tr.errKingsAdj << "\n";
            if (tr.errSameSquare)
                std::cout << "      Kings on same square           : " << tr.errSameSquare << "\n";
            if (tr.errReEncode)
                std::cout << "      Re-encode mismatch             : " << tr.errReEncode << "\n";
        }
    }

    std::cout << "\n--- Summary ---\n";
    std::cout << "  Total DTM errors: " << totalErrors << "\n";
    if (totalErrors == 0)
        std::cout << "  All checked positions are consistent.\n";

    // 6. Critical suggestions
    std::cout << "\n=== Critical Suggestions ===\n";
    std::cout << R"(
 1. EGTB loading is SILENT – loadAll() returns true even if only 1 of 73
    tables loads.  Add engine_get_egtb_count() to the C API and log it at
    startup so you know exactly how many tables the engine sees.

 2. EGTB only fires at !isRoot in pvs().  Root positions (ply=0) are never
    directly probed → the engine MUST find the EGTB-optimal move by searching
    children.  With MOVE_NONE stored for EGTB nodes and no move-ordering hint,
    the best move may be searched last and get LMR'd.  The root EGTB probe
    added recently (search.h) fixes this for winning positions; verify it is
    reached before the aspiration loop.

 3. TT score overflow (FIXED in last session) – MATE_SCORE=90000 exceeded
    int16_t range; all EGTB WIN/LOSS scores stored in TT were corrupted.
    Confirm MATE_SCORE=30000 and re-generate any evaluation checkpoints trained
    with the old score scale to avoid stale cached evaluations in your models.

 4. EGTB does not encode 50-move rule.  Tables assume infinite-horizon DTM.
    If DTM > 50 moves the engine may hit the 50-move draw before delivering
    mate – this is legal but may be why "winning" EGTB positions appear to
    draw.  Add a check: if best EGTB DTM > 50 and halfmoveClock > 0, fall
    through to alpha-beta (which now enforces the 50-move rule) instead of
    blindly returning the EGTB score.

 5. EGTB tables only cover ≤5 pieces.  Positions with 6+ pieces have no
    EGTB guidance.  If the opening of a KQvKBP (6-piece) position is reached,
    the engine relies purely on PST eval until a capture reduces to ≤5.  Consider
    probing sub-tables speculatively from 6-piece positions to guide move
    selection.

 6. The "flipped" key registration in loadAll() uses the SAME table data but
    interprets colours in reverse.  The probe() logic for flipped tables
    negates stm and swaps kingSq – verify with a known position
    (e.g. KvKQ – black queen vs lone white king) that flipped probe gives
    LOSS for white and WIN for black.

 7. DRAW positions in EGTB likely only occur through stalemate in this
    variant (no KNvK or KBvK material-only draws are expected in 6x6 chess
    where a queen is available).  If the distribution check above shows a
    large DRAW count beyond stalemate positions, the retrograde algorithm may
    be mis-labelling unresolved (TB_UNKNOWN) positions as draws.
)";

    return (totalErrors > 0) ? 2 : 0;
}
