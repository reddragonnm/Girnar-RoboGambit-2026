// Self-play data generator for Texel tuning
// Generates training data from Fischer Random 6x6 chess positions
// Output:
//   1. Supervised text format (one position per line, 39 fields):
//      board[36] side score_cp result
//   2. Optional JSONL format (full game trajectories for TD-Leaf)
//
// Usage: ./datagen [num_games] [search_depth] [output_file] [output_jsonl]

#include "types.h"
#include "attacks.h"
#include "zobrist.h"
#include "board.h"
#include "eval.h"
#include "search.h"
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <string>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <thread>
#include <mutex>
#include <sstream>
#include <cstdlib>

// ============================================================
// Data structures
// ============================================================
struct PositionRecord {
    int board[36];
    int side;
    float score; // search score in centipawns, from white's perspective
};

struct GameRecord {
    std::vector<PositionRecord> positions;
    double result;
};

struct WorkerResult {
    std::vector<GameRecord> games;
    int whiteWins = 0, blackWins = 0, draws = 0;
};

// ============================================================
// Fischer Random logic
// ============================================================
void generateFischerRandom(int boardArr[6][6], std::mt19937& rng) {
    memset(boardArr, 0, sizeof(int) * 36);
    for (int f = 0; f < 6; f++) {
        boardArr[1][f] = 1;  // White pawns
        boardArr[4][f] = 6;  // Black pawns
    }
    int backRank[6]; 
    memset(backRank, 0, sizeof(backRank));
    int darkFiles[] = {0, 2, 4};
    int lightFiles[] = {1, 3, 5};
    std::uniform_int_distribution<int> dist3(0, 2);
    backRank[darkFiles[dist3(rng)]] = BISHOP;
    backRank[lightFiles[dist3(rng)]] = BISHOP;
    std::vector<int> emptyFiles;
    for (int f = 0; f < 6; f++) if (backRank[f] == 0) emptyFiles.push_back(f);
    std::vector<int> remainingPieces = {KING, QUEEN, KNIGHT, KNIGHT};
    std::shuffle(remainingPieces.begin(), remainingPieces.end(), rng);
    for (int i = 0; i < 4; i++) backRank[emptyFiles[i]] = remainingPieces[i];
    for (int f = 0; f < 6; f++) {
        boardArr[0][f] = backRank[f];
        boardArr[5][f] = backRank[f] + 5;
    }
}

// ============================================================
// Self-play game loop
// ============================================================
double playGame(Search& search, std::mt19937& rng, int searchDepth, GameRecord& game) {
    int boardArr[6][6];
    generateFischerRandom(boardArr, rng);
    search.board.fromArray(boardArr);
    search.board.side = WHITE;
    search.clearTT();
    search.clearHistory();
    Eval::clearPawnHash();
    
    int moveCount = 0;
    const int MAX_GAME_MOVES = 200;
    std::uniform_int_distribution<int> randomMovesDist(4, 10);
    int randomMoves = randomMovesDist(rng);
    
    for (int i = 0; i < randomMoves; i++) {
        Move moves[MAX_MOVES];
        int numMoves = search.board.generateMoves(moves);
        if (numMoves == 0) break;
        std::uniform_int_distribution<int> moveDist(0, numMoves - 1);
        search.board.makeMove(moves[moveDist(rng)]);
        moveCount++;
    }
    
    while (moveCount < MAX_GAME_MOVES) {
        Move moves[MAX_MOVES];
        int numMoves = search.board.generateMoves(moves);
        if (numMoves == 0) return search.board.inCheck() ? ((search.board.side == WHITE) ? 0.0 : 1.0) : 0.5;
        if (search.board.halfmoveClock >= 100) return 0.5;
        if (search.board.isInsufficientMaterial()) return 0.5;
        // Repetition check
        int reps = 0;
        for (int i = search.board.undoCount - 2; i >= std::max(0, search.board.undoCount - search.board.halfmoveClock); i -= 2)
            if (search.board.undoStack[i].hash == search.board.hash && ++reps >= 2) return 0.5;

        // Record quiet positions only (not in check, no captures available)
        bool savedPos = false;
        if (!search.board.inCheck()) {
            Move caps[MAX_MOVES];
            if (search.board.generateCaptures(caps) == 0 &&
                std::abs(Eval::evaluate(search.board)) < 400000) {
                PositionRecord rec;
                for (int sq = 0; sq < 36; sq++) rec.board[sq] = search.board.mailbox[sq];
                rec.side = search.board.side;
                rec.score = 0.0f; // updated after search below
                game.positions.push_back(rec);
                savedPos = true;
            }
        }
        
        Move best = search.search(searchDepth, 0);

        // Record search score (white-perspective centipawns) for the saved position
        if (savedPos) {
            float stmScore = static_cast<float>(search.info.bestScore);
            game.positions.back().score = (search.board.side == WHITE) ? stmScore : -stmScore;
        }

        if (std::abs(search.info.bestScore) > MATE_THRESHOLD)
            return (search.info.bestScore > 0) ? ((search.board.side == WHITE) ? 1.0 : 0.0) : ((search.board.side == WHITE) ? 0.0 : 1.0);
        
        search.board.makeMove(best);
        moveCount++;
    }
    return 0.5;
}

void workerThread(int threadId, int numGames, int searchDepth, uint64_t seed, WorkerResult& result) {
    std::mt19937 rng(seed);
    Search search;
    for (int g = 0; g < numGames; g++) {
        GameRecord game;
        game.result = playGame(search, rng, searchDepth, game);
        result.games.push_back(game);
        if (game.result == 1.0) result.whiteWins++; else if (game.result == 0.0) result.blackWins++; else result.draws++;
        if ((g + 1) % 100 == 0)
            std::cout << "[Thread " << threadId << "] " << (g + 1) << "/" << numGames << " games done\n";
    }
}

int main(int argc, char** argv) {
    int numGames = 10000, searchDepth = 6, numThreads = std::max(1, (int)std::thread::hardware_concurrency() - 1);
    std::string outputFile = "tuning_data.txt", jsonlFile = "";
    if (argc > 1) numGames = std::atoi(argv[1]);
    if (argc > 2) searchDepth = std::atoi(argv[2]);
    if (argc > 3) outputFile = argv[3];
    if (argc > 4) jsonlFile = argv[4]; // Optional JSONL path
    
    Attacks::init(); Zobrist::init();

        auto startTime = std::chrono::steady_clock::now();
    std::vector<std::thread> threads;
    std::vector<WorkerResult> results(numThreads);
    std::mt19937 seedRng(std::chrono::steady_clock::now().time_since_epoch().count());
    for (int t = 0; t < numThreads; t++) {
        threads.emplace_back(workerThread, t, (numGames/numThreads) + (t < numGames%numThreads ? 1 : 0), searchDepth, (uint64_t)seedRng(), std::ref(results[t]));
    }
    for (auto& t : threads) t.join();

    std::ofstream out(outputFile);
    std::ofstream jout;
    if (!jsonlFile.empty()) jout.open(jsonlFile);

    size_t totalPos = 0;
    int tw = 0, tb = 0, td = 0;
    for (auto& r : results) {
        tw += r.whiteWins; tb += r.blackWins; td += r.draws;
        for (const auto& game : r.games) {
            for (const auto& pos : game.positions) {
                for (int sq = 0; sq < 36; sq++) out << pos.board[sq] << (sq < 35 ? " " : "");
                out << " " << pos.side << " " << pos.score << " " << game.result << "\n";
                totalPos++;
            }
            if (jout.is_open()) {
                jout << "{\"result\": " << game.result << ", \"positions\": [";
                for (size_t i = 0; i < game.positions.size(); i++) {
                    jout << "{\"board\": [";
                    for (int sq = 0; sq < 36; sq++) jout << game.positions[i].board[sq] << (sq < 35 ? ", " : "");
                    jout << "], \"side\": " << game.positions[i].side << ", \"score\": " << game.positions[i].score << "}" << (i < game.positions.size() - 1 ? ", " : "");
                }
                jout << "]}\n";
            }
        }
    }
    
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - startTime).count();
    std::cout << "\n=== Data Generation Complete ===\nGames: " << numGames << " (W:" << tw << " B:" << tb << " D:" << td << ")\n";
    std::cout << "Positions: " << totalPos << "\nTime: " << elapsed << "s\nOutput: " << outputFile << "\n";
    if (jout.is_open()) std::cout << "JSONL: " << jsonlFile << "\n";
    return 0;
}
