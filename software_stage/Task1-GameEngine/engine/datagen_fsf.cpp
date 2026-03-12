#include "types.h"
#include "attacks.h"
#include "zobrist.h"
#include "board.h"
#include "eval.h"
#include "search.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <thread>
#include <mutex>
#include <atomic>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

std::mutex fileMutex;
std::atomic<int> gamesFinished(0);

// Global config for FSF-both mode
struct FsfBothConfig {
    bool enabled = false;
    int skillLevel = 20;    // strong side
    int skillMin = 5;       // weak side min
    int skillMax = 19;      // weak side max
    int mixPct = 30;        // % of games that are 20v20
    std::string jsonlFile;
};

struct StoredPos {
    int8_t board[36];
    int8_t side;
};

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

std::string boardToFen(const Board& b) {
    const char* pieceChars = ".PNBQKpnbqk";
    std::string fen = "";
    for (int r = 5; r >= 0; r--) {
        int emptyCount = 0;
        for (int f = 0; f < 6; f++) {
            int sq = makeSquare(f, r);
            int piece = b.mailbox[sq];
            if (piece == 0) {
                emptyCount++;
            } else {
                if (emptyCount > 0) {
                    fen += std::to_string(emptyCount);
                    emptyCount = 0;
                }
                fen += pieceChars[piece];
            }
        }
        if (emptyCount > 0) {
            fen += std::to_string(emptyCount);
        }
        if (r > 0) fen += "/";
    }
    fen += (b.side == WHITE) ? " w" : " b";
    fen += " - - 0 1";
    return fen;
}

int fileFromChar(char c) { return c - 'a'; }
int rankFromChar(char c) { return c - '1'; }

Move uciToMove(const std::string& uci, Board& b) {
    if (uci == "(none)" || uci == "0000" || uci.length() < 4) return MOVE_NONE;
    int fromSq = makeSquare(fileFromChar(uci[0]), rankFromChar(uci[1]));
    int toSq = makeSquare(fileFromChar(uci[2]), rankFromChar(uci[3]));
    PieceType promo = NONE;
    if (uci.length() > 4) {
        char p = uci[4];
        if (p == 'n') promo = KNIGHT;
        else if (p == 'b') promo = BISHOP;
        else if (p == 'q') promo = QUEEN;
    }
    
    Move moves[MAX_MOVES];
    int count = b.generateMoves(moves);
    for (int i = 0; i < count; i++) {
        if (moveFrom(moves[i]) == fromSq && moveTo(moves[i]) == toSq) {
            if (promo == NONE || movePromo(moves[i]) == promo) {
                return moves[i];
            }
        }
    }
    
    for (int i = 0; i < count; i++) {
        if (moveFrom(moves[i]) == fromSq && moveTo(moves[i]) == toSq) {
            return moves[i];
        }
    }
    return MOVE_NONE;
}

class UCIProcess {
    int pipe_in[2];
    int pipe_out[2];
    pid_t pid;
    FILE* out_stream;
    FILE* in_stream;
public:
    UCIProcess(const std::string& path, int skillLevel = 20) {
        if (pipe(pipe_in) < 0 || pipe(pipe_out) < 0) {
            throw std::runtime_error("Failed to create pipes");
        }
        pid = fork();
        if (pid < 0) {
            throw std::runtime_error("Failed to fork");
        }
        if (pid == 0) {
            dup2(pipe_in[0], STDIN_FILENO);
            dup2(pipe_out[1], STDOUT_FILENO);
            close(pipe_in[0]); close(pipe_in[1]);
            close(pipe_out[0]); close(pipe_out[1]);
            execlp(path.c_str(), path.c_str(), NULL);
            exit(1);
        }
        close(pipe_in[0]);
        close(pipe_out[1]);
        out_stream = fdopen(pipe_in[1], "w");
        in_stream = fdopen(pipe_out[0], "r");
        
        send("uci");
        while (true) {
            std::string line = readLine();
            if (line == "uciok") break;
        }
        
        char cwd[1024];
        if (getcwd(cwd, sizeof(cwd)) != NULL) {
            std::string variantsPath = std::string(cwd) + "/variants.ini";
            send("setoption name VariantPath value " + variantsPath);
        }
        
        send("setoption name UCI_Variant value chess6x6");
        send("setoption name Use NNUE value false");
        setSkillLevel(skillLevel);
        send("isready");
        while (true) {
            std::string line = readLine();
            if (line == "readyok") break;
        }
    }

    void setSkillLevel(int level) {
        send("setoption name Skill Level value " + std::to_string(level));
    }
    
    ~UCIProcess() {
        send("quit");
        fclose(out_stream);
        fclose(in_stream);
        waitpid(pid, NULL, 0);
    }
    
    void send(const std::string& cmd) {
        fprintf(out_stream, "%s\n", cmd.c_str());
        fflush(out_stream);
    }
    
    std::string readLine() {
        char buf[2048];
        if (fgets(buf, sizeof(buf), in_stream)) {
            std::string s(buf);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
            return s;
        }
        return "";
    }

    std::string getBestMove(const std::string& fen, int movetime) {
        send("position fen " + fen);
        send("go movetime " + std::to_string(movetime));
        while (true) {
            std::string line = readLine();
            if (line.substr(0, 8) == "bestmove") {
                size_t space = line.find(' ', 9);
                if (space != std::string::npos) return line.substr(9, space - 9);
                return line.substr(9);
            }
        }
    }
};

void writeJsonlGame(const std::vector<StoredPos>& positions, float result, std::ofstream& jout) {
    std::lock_guard<std::mutex> lock(fileMutex);
    jout << "{\"result\": " << result << ", \"positions\": [";
    for (size_t i = 0; i < positions.size(); i++) {
        jout << "{\"board\": [";
        for (int sq = 0; sq < 36; sq++)
            jout << (int)positions[i].board[sq] << (sq < 35 ? ", " : "");
        jout << "], \"side\": " << (int)positions[i].side << "}" << (i < positions.size() - 1 ? ", " : "");
    }
    jout << "]}\n";
}

void workerFsfBoth(int numGames, std::ofstream& dataset, std::ofstream* jsonlOut,
                   int timeMs, const std::string& fsfPath, const FsfBothConfig& cfg) {
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count() +
                     std::hash<std::thread::id>{}(std::this_thread::get_id()));

    // Two FSF instances for white/black
    UCIProcess fsfW(fsfPath, cfg.skillLevel);
    UCIProcess fsfB(fsfPath, cfg.skillLevel);
    Board board;

    std::uniform_int_distribution<int> pctDist(1, 100);
    std::uniform_int_distribution<int> skillDist(cfg.skillMin, cfg.skillMax);
    std::uniform_int_distribution<int> coinFlip(0, 1);

    for (int i = 0; i < numGames; i++) {
        int boardArr[6][6];
        generateFischerRandom(boardArr, rng);
        board.fromArray(boardArr);
        board.side = WHITE;

        // Decide skill levels for this game
        bool is20v20 = (pctDist(rng) <= cfg.mixPct);
        int whiteSkill = cfg.skillLevel, blackSkill = cfg.skillLevel;
        if (!is20v20) {
            int weakSkill = skillDist(rng);
            if (coinFlip(rng)) whiteSkill = weakSkill;
            else blackSkill = weakSkill;
        }
        fsfW.setSkillLevel(whiteSkill);
        fsfB.setSkillLevel(blackSkill);

        fsfW.send("ucinewgame"); fsfW.send("isready");
        while (fsfW.readLine() != "readyok") {}
        fsfB.send("ucinewgame"); fsfB.send("isready");
        while (fsfB.readLine() != "readyok") {}

        std::vector<StoredPos> positions;
        int moveNum = 0;
        float result = 0.5f;

        while (moveNum < 200) {
            Move moves[MAX_MOVES];
            int numMoves = board.generateMoves(moves);
            if (numMoves == 0) {
                if (board.inCheck()) result = (board.side == WHITE) ? 0.0f : 1.0f;
                else result = 0.5f;
                break;
            }
            if (board.halfmoveClock >= 100) { result = 0.5f; break; }
            // Repetition check
            int reps = 0;
            for (int j = board.undoCount - 2;
                 j >= std::max(0, board.undoCount - board.halfmoveClock); j -= 2) {
                if (board.undoStack[j].hash == board.hash && ++reps >= 2) break;
            }
            if (reps >= 2) { result = 0.5f; break; }

            if (!board.inCheck()) {
                Move caps[MAX_MOVES];
                if (board.generateCaptures(caps) == 0) {
                    // Filter: skip extreme material imbalance
                    static constexpr int MAT_VAL[] = {0, 100, 340, 350, 1000, 20000};
                    int matW = 0, matB = 0;
                    for (int sq = 0; sq < 36; sq++) {
                        int p = board.mailbox[sq];
                        if (p >= 1 && p <= 5) matW += MAT_VAL[p];
                        else if (p >= 6 && p <= 10) matB += MAT_VAL[p - 5];
                    }
                    if (std::abs(matW - matB) < 1500)
                        storePosition(board, positions);
                }
            }

            std::string fen = boardToFen(board);
            UCIProcess& active = (board.side == WHITE) ? fsfW : fsfB;
            std::string uciMove = active.getBestMove(fen, timeMs);
            Move best = uciToMove(uciMove, board);
            if (best == MOVE_NONE) { result = 0.5f; break; }

            board.makeMove(best);
            moveNum++;
        }

        writeDataset(positions, result, dataset);
        if (jsonlOut) writeJsonlGame(positions, result, *jsonlOut);
        int total = ++gamesFinished;
        if (total % 25 == 0) {
            std::cout << "Games generated: " << total << std::endl;
        }
    }
}

void worker(int numGames, std::ofstream& dataset, int timeMs, const std::string& fsfPath) {
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count() + std::hash<std::thread::id>{}(std::this_thread::get_id()));
    Search search;
    
    // Spawn fairy-stockfish
    UCIProcess fsf(fsfPath);
    
    for (int i = 0; i < numGames; i++) {
        int boardArr[6][6];
        generateFischerRandom(boardArr, rng);
        search.board.fromArray(boardArr);
        search.board.side = WHITE;
        search.clearTT();
        search.clearHistory();
        
        fsf.send("ucinewgame");
        fsf.send("isready");
        while (true) {
            if (fsf.readLine() == "readyok") break;
        }

        std::vector<StoredPos> positions;
        int moveNum = 0;
        float result = 0.5f;

        bool enginePlaysWhite = (i % 2 == 0);

        while (moveNum < 200) {
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
            
            Move best = MOVE_NONE;
            bool engineTurn = (search.board.side == WHITE) == enginePlaysWhite;
            
            if (engineTurn) {
                best = search.search(0, timeMs); 
                if (std::abs(search.info.bestScore) > MATE_THRESHOLD) {
                    if (search.info.bestScore > 0) result = (search.board.side == WHITE) ? 1.0f : 0.0f;
                    else result = (search.board.side == WHITE) ? 0.0f : 1.0f;
                    break;
                }
            } else {
                std::string fen = boardToFen(search.board);
                std::string uciMove = fsf.getBestMove(fen, timeMs);
                best = uciToMove(uciMove, search.board);
                if (best == MOVE_NONE) {
                    // Fairy-Stockfish crashed or gave invalid move
                    result = enginePlaysWhite ? 1.0f : 0.0f;
                    break;
                }
            }
            
            search.board.makeMove(best);
            moveNum++;
        }
        
        writeDataset(positions, result, dataset);
        int total = ++gamesFinished;
        if (total % 25 == 0) {
            std::cout << "Games generated: " << total << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    Attacks::init();
    Zobrist::init();
    
    int numGames = 5000;
    int timeMs = 30;
    std::string fsfPath = "./fairy-stockfish";
    std::string outFile = "tuning_data_fsf.txt";
    FsfBothConfig cfg;
    
    // Parse positional args (backward compatible)
    if (argc > 1 && argv[1][0] != '-') numGames = std::atoi(argv[1]);
    if (argc > 2 && argv[2][0] != '-') timeMs = std::atoi(argv[2]);
    if (argc > 3 && argv[3][0] != '-') outFile = argv[3];
    
    // Parse named flags
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--fsf-both") cfg.enabled = true;
        else if (arg == "--skill-level" && i + 1 < argc) cfg.skillLevel = std::atoi(argv[++i]);
        else if (arg == "--skill-min" && i + 1 < argc) cfg.skillMin = std::atoi(argv[++i]);
        else if (arg == "--skill-max" && i + 1 < argc) cfg.skillMax = std::atoi(argv[++i]);
        else if (arg == "--mix-pct" && i + 1 < argc) cfg.mixPct = std::atoi(argv[++i]);
        else if (arg == "--jsonl" && i + 1 < argc) cfg.jsonlFile = argv[++i];
        else if (arg == "--fsf-path" && i + 1 < argc) fsfPath = argv[++i];
    }
    
    std::ofstream dataset(outFile, std::ios::trunc);
    std::ofstream jsonlOut;
    if (!cfg.jsonlFile.empty()) jsonlOut.open(cfg.jsonlFile, std::ios::trunc);

    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 1;

    if (cfg.enabled) {
        std::cout << "Starting FSF-both-sides Data Generation (Fischer Random)\n";
        std::cout << "Games: " << numGames << "\n";
        std::cout << "Time per move: " << timeMs << " ms\n";
        std::cout << "Strong side skill: " << cfg.skillLevel << "\n";
        std::cout << "Weak side range: " << cfg.skillMin << "-" << cfg.skillMax << "\n";
        std::cout << "20v20 mix: " << cfg.mixPct << "%\n";
        std::cout << "Threads: " << numThreads << "\n";

        std::vector<std::thread> threads;
        int gamesRemaining = numGames;
        int gamesPerThread = std::max(1, numGames / (int)numThreads);
        for (unsigned int i = 0; i < numThreads && gamesRemaining > 0; i++) {
            int gamesThisThread = std::min(gamesPerThread, gamesRemaining);
            threads.emplace_back(workerFsfBoth, gamesThisThread, std::ref(dataset),
                                 jsonlOut.is_open() ? &jsonlOut : nullptr,
                                 timeMs, fsfPath, std::cref(cfg));
            gamesRemaining -= gamesThisThread;
        }
        for (auto& t : threads) t.join();
    } else {
        std::cout << "Starting Fischer Random Data Generation against Fairy-Stockfish\n";
        std::cout << "Games: " << numGames << "\n";
        std::cout << "Time per move: " << timeMs << " ms\n";
        std::cout << "Threads: " << numThreads << "\n";
        
        std::vector<std::thread> threads;
        int gamesPerThread = std::max(1, numGames / (int)numThreads);
        int gamesRemaining = numGames;
        for (unsigned int i = 0; i < numThreads && gamesRemaining > 0; i++) {
            int gamesThisThread = std::min(gamesPerThread, gamesRemaining);
            threads.emplace_back(worker, gamesThisThread, std::ref(dataset), timeMs, fsfPath);
            gamesRemaining -= gamesThisThread;
        }
        for (auto& t : threads) t.join();
    }
    
    std::cout << "Dataset generation complete!\n";
    if (jsonlOut.is_open()) std::cout << "JSONL: " << cfg.jsonlFile << "\n";
    return 0;
}
