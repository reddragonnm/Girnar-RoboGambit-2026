#pragma once
#include "board.h"
#include "eval.h"
#include "types.h"
#include <algorithm>
#include <cstring>
#include <chrono>
#include <cmath>

// Transposition table entry - packed to 16 bytes
struct TTEntry {
    uint32_t key32;     // Upper 32 bits of Zobrist hash
    int16_t score;      // Evaluation score
    int8_t depth;       // Depth of search
    uint8_t flag;       // EXACT, ALPHA (upper bound), BETA (lower bound)
    uint16_t bestMove;  // Best move found
    uint8_t age;        // Generation counter for replacement
    uint8_t padding;    // Pad to 12 bytes
};

// Two entries per TT bucket (24 bytes, fits in cache nicely)
struct TTBucket {
    TTEntry entries[2];
    uint8_t pad[8]; // Pad to 32 bytes for cache line alignment
};

enum TTFlag : uint8_t {
    TT_NONE = 0,
    TT_EXACT = 1,
    TT_ALPHA = 2,  // Upper bound (failed low)
    TT_BETA = 3    // Lower bound (failed high)
};

// Default TT size: 64MB, must be power of 2
constexpr size_t DEFAULT_TT_SIZE = 64 * 1024 * 1024;
constexpr size_t TT_ENTRIES = DEFAULT_TT_SIZE / sizeof(TTEntry);
// Mask for power-of-2 indexing (TT_ENTRIES is already power of 2 since sizeof(TTEntry)=16 and 64MB/16=4M)
constexpr size_t TT_MASK = TT_ENTRIES - 1;

// Killer moves: 2 per ply
constexpr int MAX_PLY = 128;

struct SearchInfo {
    int nodes;
    int ttHits;
    int maxDepth;
    Move bestMove;
    int bestScore;
    bool stopped;
    
    // Time management
    std::chrono::steady_clock::time_point startTime;
    int timeLimit; // milliseconds, 0 = no limit
    int depthLimit; // 0 = no limit
    
    void reset() {
        nodes = 0;
        ttHits = 0;
        maxDepth = 0;
        bestMove = MOVE_NONE;
        bestScore = -INF_SCORE;
        stopped = false;
    }
};

class Search {
public:
    Board board;
    TTEntry* tt;
    size_t ttMask;
    uint8_t ttAge; // generation counter
    
    // Killer moves [ply][slot]
    Move killers[MAX_PLY][2];
    
    // Static eval per ply (for improving heuristic)
    int staticEvalHist[MAX_PLY];
    
    // History heuristic [color][from][to]
    int history[COLOR_COUNT][NUM_SQUARES][NUM_SQUARES];
    
    // Countermove heuristic [piece_moved_to][to_square]
    Move countermoves[PIECE_TYPE_COUNT + 1][NUM_SQUARES];
    
    // LMR reduction table [depth][moveNumber]
    int lmrTable[MAX_PLY][MAX_MOVES];
    
    SearchInfo info;

    // The side the engine is playing as at the root of the current search.
    // Used to apply contempt: draws are slightly bad when we are ahead.
    Color rootSide = WHITE;

    // Contempt in centipawns: a draw is worth this many cp less than 0 for
    // the side that is ahead.  Keeps the engine fighting for wins instead of
    // agreeing to draws when it has a material/positional advantage.
    static constexpr int CONTEMPT_CP = 15;

    // Return the draw score from the perspective of `stm` at `ply`.
    // When stm == rootSide we are the engine-to-move; apply contempt so the
    // engine slightly prefers fighting on over agreeing to an immediate draw.
    inline int drawScore(Color stm) const {
        return (stm == rootSide) ? -CONTEMPT_CP : CONTEMPT_CP;
    }

    Search() : tt(nullptr), ttMask(0), ttAge(0) {
        allocTT(TT_ENTRIES);
        clearHistory();
        initLMR();
    }
    
    ~Search() {
        delete[] tt;
    }
    
    void allocTT(size_t entries) {
        delete[] tt;
        // Round down to power of 2
        size_t n = 1;
        while (n * 2 <= entries) n *= 2;
        ttMask = n - 1;
        tt = new TTEntry[n];
        clearTT();
    }
    
    void clearTT() {
        memset(tt, 0, (ttMask + 1) * sizeof(TTEntry));
    }
    
    void clearHistory() {
        memset(killers, 0, sizeof(killers));
        memset(history, 0, sizeof(history));
        memset(countermoves, 0, sizeof(countermoves));
    }

        void initLMR() {
        for (int d = 0; d < MAX_PLY; d++) {
            for (int m = 0; m < MAX_MOVES; m++) {
                if (d == 0 || m == 0) {
                    lmrTable[d][m] = 0;
                } else {
                    lmrTable[d][m] = int(0.75 + log(d) * log(m) / 2.25);
                }
            }
        }
    }
    
    // Prefetch TT entry for upcoming probe
    void prefetchTT(uint64_t key) const {
        __builtin_prefetch(&tt[key & ttMask]);
    }
    
    // Probe transposition table
    bool probeTT(uint64_t key, int depth, int alpha, int beta, int& score, Move& hashMove) {
        TTEntry& entry = tt[key & ttMask];
        uint32_t key32 = (uint32_t)(key >> 32);
        
        if (entry.key32 == key32) {
            hashMove = entry.bestMove;
            info.ttHits++;
            
            if (entry.depth >= depth) {
                score = entry.score;
                
                if (entry.flag == TT_EXACT) return true;
                if (entry.flag == TT_ALPHA && score <= alpha) { score = alpha; return true; }
                if (entry.flag == TT_BETA && score >= beta) { score = beta; return true; }
            }
        } else {
            hashMove = MOVE_NONE;
        }
        
        return false;
    }
    
    // Store in transposition table.
    // Mate scores are stored in a ply-independent form: we add ply to winning
    // scores (and subtract ply from losing scores) before storing, then reverse
    // on retrieval.  This ensures the same position reached at different plies
    // always sees the correct distance-to-mate rather than the one from the
    // first time it was stored.
    void storeTT(uint64_t key, int depth, int score, Move bestMove, TTFlag flag, int ply = 0) {
        TTEntry& entry = tt[key & ttMask];
        uint32_t key32 = (uint32_t)(key >> 32);
        
        // Normalize mate scores before storing
        if (score > MATE_THRESHOLD)       score += ply;
        else if (score < -MATE_THRESHOLD) score -= ply;
        
        // Replace if: empty, same position, different age, or shallower depth
        if (entry.key32 == 0 || entry.key32 == key32 || 
            entry.age != ttAge || entry.depth <= depth) {
            entry.key32 = key32;
            entry.score = (int16_t)score;
            entry.depth = (int8_t)depth;
            entry.bestMove = bestMove;
            entry.flag = flag;
            entry.age = ttAge;
        }
    }
    
    // Check if time is up
    bool shouldStop() {
        if (info.stopped) return true;
        if (info.timeLimit > 0 && (info.nodes & 4095) == 0) {
            auto now = std::chrono::steady_clock::now();
            int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - info.startTime).count();
            if (elapsed >= info.timeLimit) {
                info.stopped = true;
                return true;
            }
        }
        return false;
    }
    
    // Move ordering score (also caches SEE value for captures)
    int scoreMove(Move m, Move hashMove, int ply, Move counterMove, int& seeCache) {
        if (m == hashMove) { seeCache = 0; return 1000000; } // TT move first
        
        int from = moveFrom(m);
        int to = moveTo(m);
        int captured = board.mailbox[to];
        
        // Captures: MVV-LVA + SEE sign classification
        if (captured) {
            PieceType victim = pieceTypeFromId(captured);
            PieceType attacker = pieceTypeFromId(board.mailbox[from]);
            seeCache = board.see(m);
            if (seeCache >= 0)
                return 100000 + MATERIAL_VALUE[victim] * 10 - MATERIAL_VALUE[attacker];
            else
                return -100000 + MATERIAL_VALUE[victim] * 10 - MATERIAL_VALUE[attacker]; // bad captures last
        }
        
        seeCache = 0;
        
        // Promotions
        if (movePromo(m) != NONE) {
            return 90000 + MATERIAL_VALUE[movePromo(m)];
        }
        
        // Killer moves
        if (ply < MAX_PLY) {
            if (m == killers[ply][0]) return 80000;
            if (m == killers[ply][1]) return 79000;
        }
        
        // Countermove bonus
        if (m == counterMove) return 70000;
        
        // History heuristic
        return history[board.side][from][to];
    }
    
    // Pick the best move - simple version for quiescence (no SEE cache)
    inline void pickBest(Move moves[], int scores[], int start, int count) {
        int bestIdx = start;
        int bestScore = scores[start];
        for (int i = start + 1; i < count; i++) {
            if (scores[i] > bestScore) {
                bestScore = scores[i];
                bestIdx = i;
            }
        }
        if (bestIdx != start) {
            std::swap(moves[start], moves[bestIdx]);
            std::swap(scores[start], scores[bestIdx]);
        }
    }
    
    // Pick the best move from position 'start' onwards, swap it to position 'start'
    // This is much more efficient than full sort when we get beta cutoffs early
    inline void pickBest(Move moves[], int scores[], int seeVals[], int start, int count) {
        int bestIdx = start;
        int bestScore = scores[start];
        for (int i = start + 1; i < count; i++) {
            if (scores[i] > bestScore) {
                bestScore = scores[i];
                bestIdx = i;
            }
        }
        if (bestIdx != start) {
            std::swap(moves[start], moves[bestIdx]);
            std::swap(scores[start], scores[bestIdx]);
            std::swap(seeVals[start], seeVals[bestIdx]);
        }
    }
    
    // Quiescence search - search captures until position is quiet
    int quiescence(int alpha, int beta, int ply) {
        info.nodes++;
        
        if (shouldStop()) return 0;

        // If in check we cannot use stand-pat (the side to move MUST evade).
        // qsearch only generates captures so it can never find quiet evasions,
        // and SEE pruning may skip a mating capture entirely.
        // Detect checkmate/stalemate here before proceeding.
        if (board.inCheck()) {
            Move evasions[MAX_MOVES];
            if (board.generateMoves(evasions) == 0)
                return -MATE_SCORE + ply;
        }
        
        // Stand-pat score
        int standPat = Eval::evaluate(board);
        
        if (standPat >= beta) return beta;
        
        // Delta pruning
        constexpr int DELTA = 1050; // Queen value + margin
        if (standPat + DELTA < alpha) return alpha;
        
        if (alpha < standPat) alpha = standPat;
        
        // Generate only captures and promotions (much faster than generating all moves)
        Move captures[MAX_MOVES];
        int numCaptures = board.generateCaptures(captures);
        
        // Score captures for ordering (MVV-LVA)
        int capScores[MAX_MOVES];
        for (int i = 0; i < numCaptures; i++) {
            int to = moveTo(captures[i]);
            if (board.mailbox[to]) {
                PieceType victim = pieceTypeFromId(board.mailbox[to]);
                PieceType attacker = pieceTypeFromId(board.mailbox[moveFrom(captures[i])]);
                capScores[i] = MATERIAL_VALUE[victim] * 10 - MATERIAL_VALUE[attacker];
            } else {
                capScores[i] = MATERIAL_VALUE[movePromo(captures[i])];
            }
        }
        
        for (int i = 0; i < numCaptures; i++) {
            pickBest(captures, capScores, i, numCaptures);
            
            // SEE pruning: skip clearly losing captures
            if (board.see(captures[i]) < 0) continue;
            
            // Delta pruning per move
            int to = moveTo(captures[i]);
            if (board.mailbox[to]) {
                PieceType victim = pieceTypeFromId(board.mailbox[to]);
                if (standPat + MATERIAL_VALUE[victim] + 200 < alpha) continue;
            }
            
            board.makeMove(captures[i]);
            int score = -quiescence(-beta, -alpha, ply + 1);
            board.unmakeMove();
            
            if (info.stopped) return 0;
            
            if (score >= beta) return beta;
            if (score > alpha) alpha = score;
        }
        
        return alpha;
    }
    
    // Principal Variation Search (Negamax + Alpha-Beta)
    int pvs(int depth, int alpha, int beta, int ply, bool doNull = true) {
        if (info.stopped) return 0;
        
        bool isRoot = (ply == 0);
        bool isPV = (beta - alpha > 1);
        
        // Check for draw by repetition (threefold: need 2 prior occurrences)
        if (ply > 0) {
            int stackSize = board.undoCount;
            int limit = std::max(0, stackSize - board.halfmoveClock);
            int reps = 0;
            for (int i = stackSize - 2; i >= limit; i -= 2) {
                if (board.undoStack[i].hash == board.hash) {
                    reps++;
                    // 2nd prior occurrence = 3 total = threefold repetition.
                    // Also accept a draw on the 1st prior occurrence when inside
                    // the search tree (ply > 0) to avoid chasing repetitions —
                    // but score it with contempt so the engine won't prefer it
                    // when a winning alternative exists.
                    if (reps >= 2) return drawScore(board.side);
                }
            }
            if (reps >= 1) return drawScore(board.side);
        }

        // Check for draw by 50-move rule
        if (board.halfmoveClock >= 100)
            return drawScore(board.side);

        // Check for draw by insufficient material
        if (board.isInsufficientMaterial())
            return drawScore(board.side);
        
        // Probe TT
        Move hashMove = MOVE_NONE;
        int ttScore = 0;
        int8_t ttDepth = -1;
        uint8_t ttBound = TT_NONE;
        bool ttHit = false;
        {
            TTEntry& entry = tt[board.hash & ttMask];
            uint32_t key32 = (uint32_t)(board.hash >> 32);
            if (entry.key32 == key32) {
                hashMove = entry.bestMove;
                ttHit = true;
                ttDepth = entry.depth;
                ttBound = entry.flag;
                // Un-normalize mate scores: reverse the ply adjustment done in storeTT
                ttScore = entry.score;
                if (ttScore > MATE_THRESHOLD)       ttScore -= ply;
                else if (ttScore < -MATE_THRESHOLD) ttScore += ply;
                info.ttHits++;
                
                if (!isRoot && entry.depth >= depth) {
                    // Don't trust a TT_EXACT draw score (0 ± contempt) blindly:
                    // the stored entry may be from a different path where a
                    // repetition or 50-move rule triggered, but on the current
                    // path those conditions don't hold.  Only use the TT cutoff
                    // for non-draw scores, or for draws when halfmoveClock > 0
                    // (meaning we're genuinely in territory where a draw is likely).
                    bool isTTDraw = (ttScore >= -CONTEMPT_CP - 1 && ttScore <= CONTEMPT_CP + 1);
                    if (entry.flag == TT_EXACT && !(isTTDraw && board.halfmoveClock == 0)) return ttScore;
                    if (entry.flag == TT_ALPHA && ttScore <= alpha) return alpha;
                    if (entry.flag == TT_BETA  && ttScore >= beta)  return beta;
                }
            }
        }
        
        info.nodes++;
        if (ply > info.maxDepth) info.maxDepth = ply;

        bool inCheck = board.inCheck();

        // Check extension — must be applied BEFORE the depth<=0 branch so a mated
        // position arriving at depth=0 is extended to depth=1 and correctly returns
        // MATE_SCORE rather than being forwarded to qsearch.
        if (inCheck) depth++;

        // Leaf node: go to quiescence (in-check positions are already extended above)
        if (depth <= 0) {
            return quiescence(alpha, beta, ply);
        }
        
        // Static eval (computed once, reused for pruning decisions)
        int staticEval = Eval::evaluate(board);
        if (ply < MAX_PLY) staticEvalHist[ply] = staticEval;
        
        // Determine if position is improving (static eval is better than 2 plies ago)
        bool improving = (ply >= 2 && !inCheck && staticEval > staticEvalHist[ply - 2]);
        
        // Razoring: at low depth, if static eval is far below alpha, verify with qsearch
        if (!isPV && !inCheck && depth <= 2) {
            int razorMargin = 300 + depth * 60;
            if (staticEval + razorMargin <= alpha) {
                int razorScore = quiescence(alpha, beta, ply);
                if (razorScore <= alpha) return razorScore;
            }
        }
        
        // Reverse futility pruning (static eval pruning)
        if (!isPV && !inCheck && depth <= 7) {
            int margin = depth * 80 - (improving ? 60 : 0);
            if (staticEval - margin >= beta) {
                return staticEval - margin;
            }
        }
        
        // Null move pruning
        if (doNull && !isPV && !inCheck && depth >= 3) {
            int nonPawnMaterial = 0;
            for (int pt = KNIGHT; pt <= QUEEN; pt++)
                nonPawnMaterial += popcount(board.pieces[board.side][pt]) * (int)MATERIAL_VALUE[pt];
            
            if (nonPawnMaterial > 0) {
                int R = 3 + depth / 4;
                
                board.side = ~board.side;
                board.hash ^= Zobrist::sideKey;
                
                int nullScore = -pvs(depth - 1 - R, -beta, -beta + 1, ply + 1, false);
                
                board.side = ~board.side;
                board.hash ^= Zobrist::sideKey;
                
                if (info.stopped) return 0;
                
                if (nullScore >= beta) {
                    return beta;
                }
            }
        }
        
        // Internal Iterative Deepening (IID)
        // If no hash move available at sufficient depth, do a shallow search first
        if (hashMove == MOVE_NONE && depth >= 4 && isPV) {
            pvs(depth - 2, alpha, beta, ply, false);
            TTEntry& iidEntry = tt[board.hash & ttMask];
            if (iidEntry.key32 == (uint32_t)(board.hash >> 32)) {
                hashMove = iidEntry.bestMove;
            }
        }
        
        // Generate and order moves
        Move moves[MAX_MOVES];
        int numMoves = board.generateMoves(moves);
        
        if (numMoves == 0) {
            if (inCheck) return -MATE_SCORE + ply;
            return drawScore(board.side); // stalemate
        }
        
        // Determine countermove from the previous move
        Move counterMove = MOVE_NONE;
        if (board.undoCount > 0) {
            Move prevMove = board.undoStack[board.undoCount - 1].move;            int prevTo = moveTo(prevMove);
            PieceType prevPt = pieceTypeFromId(board.mailbox[prevTo]);
            if (prevPt != NONE) {
                counterMove = countermoves[prevPt][prevTo];
            }
        }
        
        // Score moves for ordering (with SEE cache for captures)
        int scores[MAX_MOVES];
        int seeValues[MAX_MOVES];
        for (int i = 0; i < numMoves; i++) {
            scores[i] = scoreMove(moves[i], hashMove, ply, counterMove, seeValues[i]);
        }
        
        Move bestMove = MOVE_NONE;
        int bestScore = -INF_SCORE;
        TTFlag ttFlag = TT_ALPHA;
        int movesSearched = 0;
        
        for (int i = 0; i < numMoves; i++) {
            pickBest(moves, scores, seeValues, i, numMoves);
            Move m = moves[i];
            bool isCapture = (board.mailbox[moveTo(m)] != 0);
            bool isPromo = (movePromo(m) != NONE);
            bool isKiller = (ply < MAX_PLY && (m == killers[ply][0] || m == killers[ply][1]));
            bool isQuiet = !isCapture && !isPromo;
            
            // Late Move Pruning (LMP): skip quiet moves at low depth after enough moves
            if (!isPV && !inCheck && isQuiet && depth <= 5 && movesSearched >= 3 + depth * depth) {
                continue;
            }
            
            // Futility pruning for quiet moves at low depth
            if (!isPV && !inCheck && isQuiet && depth <= 6 && movesSearched >= 1) {
                int margin = depth * 100 + (improving ? 50 : 0);
                if (staticEval + margin <= alpha) {
                    continue;
                }
            }
            
            // SEE pruning for bad captures in non-PV nodes (uses cached SEE)
            if (!isPV && !inCheck && isCapture && depth <= 5 && movesSearched >= 1) {
                if (seeValues[i] < -30 * depth) {
                    continue;
                }
            }
            
            int score;
            int extension = 0;
            
            // Singular extension: if the TT move is much better than alternatives, extend it
            if (!isRoot && m == hashMove && depth >= 8 && !inCheck
                && ttHit && ttDepth >= depth - 3
                && (ttBound == TT_BETA || ttBound == TT_EXACT)
                && std::abs(ttScore) < MATE_THRESHOLD)
            {
                int sBeta = ttScore - 2 * depth;
                int sDepth = (depth - 1) / 2;
                
                // Search with excluded move: set hash move score to minimum temporarily
                scores[i] = -INF_SCORE;
                
                // Use pvs at reduced depth/window to check if other moves can reach sBeta
                bool singular = true;
                for (int j = i + 1; j < numMoves; j++) {
                    pickBest(moves, scores, seeValues, j, numMoves);
                    
                    board.makeMove(moves[j]);
                    int seScore = -pvs(sDepth - 1, -sBeta, -sBeta + 1, ply + 1, false);
                    board.unmakeMove();
                    
                    if (info.stopped) return 0;
                    if (seScore >= sBeta) { singular = false; break; }
                }
                
                scores[i] = 1000000; // Restore hash move priority
                
                if (singular) {
                    extension = 1;
                }
            }
            
            board.makeMove(m);
            movesSearched++;
            
            // Prefetch the TT entry for the child position
            prefetchTT(board.hash);
            
            if (i == 0) {
                score = -pvs(depth - 1 + extension, -beta, -alpha, ply + 1);
            } else {
                // Late Move Reductions
                int reduction = 0;
                if (depth >= 2 && movesSearched >= 2 && !inCheck) {
                    if (isQuiet) {
                        // Use precomputed log-based table
                        reduction = lmrTable[std::min(depth, MAX_PLY - 1)][std::min(movesSearched, MAX_MOVES - 1)];
                        if (!isPV) reduction++;
                        if (!improving) reduction++;
                        // Reduce less for killers
                        if (isKiller) reduction--;
                        // History-based adjustment
                        int histVal = history[board.side ^ 1][moveFrom(m)][moveTo(m)];
                        if (histVal < -1000) reduction++;
                        if (histVal > 3000) reduction--;
                    } else if (isCapture && seeValues[i] < 0) {
                        // Reduce bad captures
                        reduction = 1 + (depth >= 4 ? 1 : 0);
                    }
                    
                    if (reduction < 0) reduction = 0;
                    if (reduction >= depth - 1) reduction = depth - 2;
                }
                
                // Zero-window search with reduction
                score = -pvs(depth - 1 - reduction, -alpha - 1, -alpha, ply + 1);
                
                // Re-search if reduced move raises alpha
                if (reduction > 0 && score > alpha) {
                    score = -pvs(depth - 1, -alpha - 1, -alpha, ply + 1);
                }
                
                // Re-search with full window if score is in (alpha, beta)
                if (score > alpha && score < beta) {
                    score = -pvs(depth - 1, -beta, -alpha, ply + 1);
                }
            }
            
            board.unmakeMove();
            
            if (info.stopped) return 0;
            
            if (score > bestScore) {
                bestScore = score;
                bestMove = m;
                
                if (score > alpha) {
                    alpha = score;
                    ttFlag = TT_EXACT;
                    
                    if (isRoot) {
                        info.bestMove = m;
                        info.bestScore = score;
                    }
                    
                    if (score >= beta) {
                        ttFlag = TT_BETA;
                        
                        // Update killer moves (quiet moves only)
                        if (isQuiet && ply < MAX_PLY) {
                            if (killers[ply][0] != m) {
                                killers[ply][1] = killers[ply][0];
                                killers[ply][0] = m;
                            }
                        }
                        
                        // Update countermove heuristic
                        if (isQuiet && board.undoCount > 0) {
                            Move prevMove = board.undoStack[board.undoCount - 1].move;
                            int prevTo = moveTo(prevMove);
                            PieceType prevPt = pieceTypeFromId(board.mailbox[prevTo]);
                            if (prevPt != NONE) {
                                countermoves[prevPt][prevTo] = m;
                            }
                        }
                        
                        // Update history with gravity (dampen old values)
                        if (isQuiet) {
                            int bonus = depth * depth;
                            int& hist = history[board.side][moveFrom(m)][moveTo(m)];
                            hist += bonus - hist * bonus / 10000;
                            
                            // Penalize other quiet moves that didn't cause a cutoff
                            for (int j = 0; j < i; j++) {
                                Move prev = moves[j];
                                if (board.mailbox[moveTo(prev)] == 0 && movePromo(prev) == NONE) {
                                    int& h = history[board.side][moveFrom(prev)][moveTo(prev)];
                                    h -= bonus + h * bonus / 10000;
                                }
                            }
                        }
                        
                        break;
                    }
                }
            }
        }
        
        // Store in TT
        storeTT(board.hash, depth, bestScore, bestMove, ttFlag, ply);
        
        return bestScore;
    }
    
    // Iterative deepening search
    Move search(int maxDepth = 0, int timeLimitMs = 0) {
        info.reset();
        info.startTime = std::chrono::steady_clock::now();
        info.timeLimit = timeLimitMs;
        info.depthLimit = maxDepth;
        rootSide = board.side; // for contempt: draws are bad for us when ahead
        
        // Age the TT so old entries from previous searches can be replaced
        ttAge++;
        
        if (maxDepth == 0) maxDepth = MAX_PLY;
        
        Move bestMove = MOVE_NONE;
        int bestScore = -INF_SCORE;
        
        // Aspiration window
        int aspAlpha = -INF_SCORE;
        int aspBeta = INF_SCORE;
        constexpr int ASP_WINDOW = 25;
        
        for (int depth = 1; depth <= maxDepth; depth++) {
            if (depth >= 4) {
                aspAlpha = bestScore - ASP_WINDOW;
                aspBeta = bestScore + ASP_WINDOW;
            }
            
            int score;
            int delta = ASP_WINDOW;
            
            while (true) {
                score = pvs(depth, aspAlpha, aspBeta, 0);
                
                if (info.stopped) break;
                
                if (score <= aspAlpha) {
                    // Failed low: widen alpha
                    aspBeta = (aspAlpha + aspBeta) / 2;
                    aspAlpha = std::max(score - delta, -INF_SCORE);
                    delta *= 2;
                } else if (score >= aspBeta) {
                    // Failed high: widen beta
                    aspBeta = std::min(score + delta, INF_SCORE);
                    delta *= 2;
                } else {
                    break; // score is within window
                }
                
                if (delta > 1000) {
                    // Window is too wide, search full window
                    aspAlpha = -INF_SCORE;
                    aspBeta = INF_SCORE;
                }
            }
            
            if (info.stopped) break;
            
            bestMove = info.bestMove;
            bestScore = score;
            info.bestScore = bestScore;
            
            // Check depth limit
            if (info.depthLimit > 0 && depth >= info.depthLimit) break;
            
            // If we found a forced mate, stop searching
            if (std::abs(bestScore) > MATE_THRESHOLD) break;
        }
        
        // Ensure we have a valid move
        if (bestMove == MOVE_NONE) {
            Move moves[MAX_MOVES];
            int n = board.generateMoves(moves);
            if (n > 0) bestMove = moves[0];
        }
        
        info.bestMove = bestMove;
        return bestMove;
    }
    
    // Get the formatted move string for the competition
    std::string bestMoveStr() {
        Move m = info.bestMove;
        if (m == MOVE_NONE) return "";
        
        int from = moveFrom(m);
        PieceType pt = pieceTypeFromId(board.mailbox[from]);
        
        // If promotion, output the promoted piece's ID (not the pawn's)
        PieceType outputType = (movePromo(m) != NONE) ? movePromo(m) : pt;
        
        return moveToStr(m, board.side, outputType);
    }
};
