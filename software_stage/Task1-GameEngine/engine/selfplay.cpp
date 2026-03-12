#include "types.h"
#include "attacks.h"
#include "zobrist.h"
#include "board.h"
#include "eval.h"
#include "search.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <thread>
#include <mutex>
#include <atomic>

struct StoredPos {
    int8_t board[36];
    int8_t side;
};

std::mutex fileMutex;
std::atomic<int> gamesFinished(0);

void storePosition(const Board& b, std::vector<StoredPos>& positions) {
    StoredPos p;
    for (int sq = 0; sq < 36; sq++)
        p.board[sq] = b.mailbox[sq];
    p.side = b.side;
    positions.push_back(p);
}

void writeDataset(const std::vector<StoredPos>& positions, float result, std::ofstream& out) {
    std::lock_guard<std::mutex> lock(fileMutex);
    for (const auto& p : positions) {
        for (int i = 0; i < 36; i++)
            out << (int)p.board[i] << " ";
        out << (int)p.side << " " << result << "\n";
    }
}

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

bool isRepetition(const Board& board) {
    int reps = 0;
    int stackSize = board.undoCount;
    int limit = std::max(0, stackSize - board.halfmoveClock);
    for (int i = stackSize - 2; i >= limit; i -= 2) {
        if (board.undoStack[i].hash == board.hash) {
            reps++;
            if (reps >= 2) return true;
        }
    }
    return false;
}

void worker(int numGames, std::ofstream& dataset) {
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count() + std::hash<std::thread::id>{}(std::this_thread::get_id()));
    Search search;
    
    for (int i = 0; i < numGames; i++) {
        int boardArr[6][6];
        generateFischerRandom(boardArr, rng);
        search.board.fromArray(boardArr);
        search.board.side = WHITE;
        search.clearTT();
        search.clearHistory();
        
        std::vector<StoredPos> positions;
        int moveNum = 0;
        float result = 0.5f;

        while (moveNum < 150) {
            Move moves[MAX_MOVES];
            int numMoves = search.board.generateMoves(moves);
            if (numMoves == 0) {
                if (search.board.inCheck()) result = (search.board.side == WHITE) ? 0.0f : 1.0f;
                else result = 0.5f;
                break;
            }
            if (search.board.halfmoveClock >= 100 || isRepetition(search.board)) {
                result = 0.5f;
                break;
            }
            if (!search.board.inCheck()) {
                Move caps[MAX_MOVES];
                int eval = Eval::evaluate(search.board);
                if (search.board.generateCaptures(caps) == 0 && std::abs(eval) < 1500) {
                    storePosition(search.board, positions);
                }
            }
            Move best = search.search(0, 50); // Increased to 50ms for high quality refinement data
            if (std::abs(search.info.bestScore) > MATE_THRESHOLD) {
                if (search.info.bestScore > 0) result = (search.board.side == WHITE) ? 1.0f : 0.0f;
                else result = (search.board.side == WHITE) ? 0.0f : 1.0f;
                break;
            }
            search.board.makeMove(best);
            moveNum++;
        }
        writeDataset(positions, result, dataset);
        int total = ++gamesFinished;
        if (total % 100 == 0) {
            std::cout << "Games generated: " << total << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    Attacks::init();
    Zobrist::init();
    
    int numGames = 50000;
    if (argc > 1) numGames = std::atoi(argv[1]);
    
    std::ofstream dataset("tuning_data_frc.txt", std::ios::trunc);
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 1;

    std::cout << "Starting Fischer Random data generation (" << numGames << " games on " << numThreads << " threads)...\n";
    
    std::vector<std::thread> threads;
    int gamesPerThread = numGames / numThreads;
    for (unsigned int i = 0; i < numThreads; i++) {
        threads.emplace_back(worker, gamesPerThread, std::ref(dataset));
    }
    for (auto& t : threads) t.join();
    
    std::cout << "Dataset generation complete!\n";
    return 0;
}