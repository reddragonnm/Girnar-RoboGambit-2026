#include "types.h"
#include "attacks.h"
#include "zobrist.h"
#include "board.h"
#include "eval.h"
#include "search.h"
#include <iostream>
#include <sstream>
#include <cstdio>
#include <iomanip>

// Print board for debugging
void Board::print() const {
    const char* pieceChars = ".PNBQKpnbqk";
    
    std::cout << "\n  A B C D E F\n";
    for (int r = 5; r >= 0; r--) {
        std::cout << (r + 1) << " ";
        for (int f = 0; f < 6; f++) {
            int sq = makeSquare(f, r);
            int piece = mailbox[sq];
            if (piece == 0) {
                std::cout << ". ";
            } else {
                Color c = pieceColor(piece);
                PieceType pt = pieceTypeFromId(piece);
                char ch = pieceChars[pt];
                if (c == BLACK) ch = pieceChars[pt + 5]; // lowercase
                std::cout << ch << " ";
            }
        }
        std::cout << (r + 1) << "\n";
    }
    std::cout << "  A B C D E F\n";
    std::cout << (side == WHITE ? "White" : "Black") << " to move\n";
    std::cout << "Hash: " << std::hex << hash << std::dec << "\n\n";
}

// Initialize all subsystems
void initAll() {
    Attacks::init();
    Zobrist::init();
}

// Main engine interface: takes a 6x6 board array, returns move string
// board[rank][file] where rank 0 = rank 1 (bottom), with piece IDs 0-10
std::string getBestMove(const int boardArr[6][6], int sideToMove, int timeLimitMs = 2000) {
    Search search;
    search.board.fromArray(boardArr);
    search.board.side = (sideToMove == 0) ? WHITE : BLACK;
    
    // Recompute hash for side
    if (search.board.side == BLACK) {
        search.board.hash ^= Zobrist::sideKey;
    }
    
    Move bestMove = search.search(0, timeLimitMs);
    
    return search.bestMoveStr();
}

// Perft: count leaf nodes for move generation testing
uint64_t perft(Board& board, int depth) {
    if (depth == 0) return 1;
    
    Move moves[MAX_MOVES];
    int numMoves = board.generateMoves(moves);
    
    uint64_t nodes = 0;
    for (int i = 0; i < numMoves; i++) {
        board.makeMove(moves[i]);
        nodes += perft(board, depth - 1);
        board.unmakeMove();
    }
    
    return nodes;
}

// Run perft test from initial position
void testPerft() {
    // Standard Fischer Random starting position example:
    // Back rank: B N Q K N B (one possible arrangement with bishops on opposite colors)
    // White back rank (rank 1), Black back rank (rank 6)
    // Pawns on rank 2 (white) and rank 5 (black)
    
    // Piece IDs:
    // White: PAWN=1, KNIGHT=2, BISHOP=3, QUEEN=4, KING=5
    // Black: PAWN=6, KNIGHT=7, BISHOP=8, QUEEN=9, KING=10
    
    int boardArr[6][6] = {
        // Rank 1 (row 0): White back rank
        {3, 2, 4, 5, 2, 3},  // B N Q K N B
        // Rank 2 (row 1): White pawns
        {1, 1, 1, 1, 1, 1},
        // Rank 3-4: empty
        {0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0},
        // Rank 5 (row 4): Black pawns
        {6, 6, 6, 6, 6, 6},
        // Rank 6 (row 5): Black back rank
        {8, 7, 9, 10, 7, 8}  // b n q k n b
    };
    
    Board board;
    board.fromArray(boardArr);
    board.side = WHITE;
    board.print();
    
    std::cout << "Perft results:\n";
    for (int d = 1; d <= 6; d++) {
        auto start = std::chrono::steady_clock::now();
        uint64_t nodes = perft(board, d);
        auto end = std::chrono::steady_clock::now();
        int ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Depth " << d << ": " << nodes << " nodes";
        if (ms > 0) {
            std::cout << " (" << ms << "ms, " << (nodes * 1000 / ms) << " nps)";
        }
        std::cout << "\n";
    }
}

// Test search
void testSearch() {
    int boardArr[6][6] = {
        {3, 2, 4, 5, 2, 3},
        {1, 1, 1, 1, 1, 1},
        {0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0},
        {6, 6, 6, 6, 6, 6},
        {8, 7, 9, 10, 7, 8}
    };
    
    Board board;
    board.fromArray(boardArr);
    board.side = WHITE;
    board.print();
    
    Search search;
    search.board = board;
    
    std::cout << "Searching (5 seconds)...\n";
    Move best = search.search(0, 5000);
    
    int from = moveFrom(best);
    int to = moveTo(best);
    PieceType pt = pieceTypeFromId(board.mailbox[from]);
    
    std::cout << "Best move: " << search.bestMoveStr() << "\n";
    std::cout << "Score: " << search.info.bestScore << " cp\n";
    std::cout << "Nodes: " << search.info.nodes << "\n";
    std::cout << "TT hits: " << search.info.ttHits << "\n";
    std::cout << "Max depth: " << search.info.maxDepth << "\n";
}

// Bench: test search on multiple varied positions
void testBench() {
    struct BenchPos {
        int board[6][6];
        Color side;
        const char* description;
    };

    BenchPos positions[] = {
        // === OPENING POSITIONS (different Fischer Random back ranks) ===
        // 1. Standard opening: B N Q K N B
        {{{3,2,4,5,2,3},{1,1,1,1,1,1},{0,0,0,0,0,0},{0,0,0,0,0,0},{6,6,6,6,6,6},{8,7,9,10,7,8}},
         WHITE, "Opening: BNQKNB (standard)"},

        // 2. Fischer Random: N B K Q B N (bishops on opposite colors: B on file1=dark, B on file4=light)
        {{{2,3,5,4,3,2},{1,1,1,1,1,1},{0,0,0,0,0,0},{0,0,0,0,0,0},{6,6,6,6,6,6},{7,8,10,9,8,7}},
         WHITE, "Opening: NBKQBN (Fischer)"},

        // 3. Fischer Random: Q N B K B N (B on file2=light, B on file4=dark... wait, need opposite colors)
        // file0=dark,file1=light,file2=dark,file3=light,file4=dark,file5=light (rank1 a1=dark)
        // B on file1(light) and file4(dark) = opposite colors
        {{{4,3,2,5,3,2},{1,1,1,1,1,1},{0,0,0,0,0,0},{0,0,0,0,0,0},{6,6,6,6,6,6},{9,8,7,10,8,7}},
         WHITE, "Opening: QBNKBN (Fischer)"},

        // 4. Fischer Random: B K N Q B N (B on file0=dark, B on file4=dark... no, that's same color)
        // B on file0(dark) and file3(light) = opposite colors
        {{{3,5,2,3,4,2},{1,1,1,1,1,1},{0,0,0,0,0,0},{0,0,0,0,0,0},{6,6,6,6,6,6},{8,10,7,8,9,7}},
         WHITE, "Opening: BKNBQN (Fischer)"},

        // === MIDGAME POSITIONS ===
        // 5. Early midgame: some pawns advanced, all pieces remain
        {{{3,2,4,5,2,3},{1,0,1,1,1,1},{0,1,0,0,0,0},{0,0,0,0,6,0},{6,6,6,6,0,6},{8,7,9,10,7,8}},
         WHITE, "Midgame: early, pawns advanced"},

        // 6. Midgame with captures: missing some pawns and a knight
        {{{3,0,4,5,2,3},{1,0,1,0,1,1},{0,1,0,0,0,0},{0,0,0,6,0,0},{6,0,6,0,6,6},{8,7,9,10,0,8}},
         BLACK, "Midgame: some captures, Black to move"},

        // 7. Complex midgame: queens traded, bishops and knights remain
        // White: Kd1, Bc1, Ne3, Pawns a2,c2,f2  Black: Kd6, Bf6, Na5, Pawns a5,c5,f5
        {{{0,0,3,5,0,0},{1,0,1,0,0,1},{0,0,0,0,2,0},{0,0,0,0,0,0},{7,0,6,0,0,8},{0,0,0,10,0,0}},
         WHITE, "Midgame: no queens, minor pieces"},

        // 8. Tactical midgame: unbalanced material
        {{{0,0,4,5,2,3},{1,0,1,0,1,0},{0,1,0,0,0,0},{0,0,6,6,0,0},{0,6,0,0,6,6},{8,7,9,10,0,0}},
         WHITE, "Midgame: unbalanced, tactical"},

        // === ENDGAME POSITIONS ===
        // 9. Queen endgame: Kings + Queens + a few pawns
        {{{0,0,0,5,0,0},{0,0,1,0,1,0},{0,0,0,0,4,0},{0,9,0,0,0,0},{0,6,0,0,6,0},{0,0,0,10,0,0}},
         WHITE, "Endgame: Q+2P vs Q+2P"},

        // 10. Knight endgame: Kings + Knights + pawns
        {{{0,0,0,5,0,0},{0,1,0,0,1,0},{0,0,2,0,0,0},{0,0,0,7,0,0},{0,6,0,0,6,0},{0,0,0,10,0,0}},
         BLACK, "Endgame: N+2P vs N+2P, Black to move"},

        // 11. Lone king vs K+Q (should find mate)
        {{{0,0,0,5,4,0},{0,0,0,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0,0},{0,0,10,0,0,0}},
         WHITE, "Endgame: KQ vs K"},
    };

    int numPositions = sizeof(positions) / sizeof(positions[0]);
    int totalNodes = 0;
    int totalDepth = 0;
    int totalTimeMs = 0;
    int searchTimeMs = 5000; // 5 seconds per position

    std::cout << "=== Benchmark: " << numPositions << " positions, " 
              << searchTimeMs / 1000 << "s each ===\n\n";

    Search search;

    for (int i = 0; i < numPositions; i++) {
        Board board;
        board.fromArray(positions[i].board);
        board.side = positions[i].side;
        if (board.side == BLACK) {
            board.hash ^= Zobrist::sideKey;
        }

        // Verify the position is legal (king not in check by side NOT to move)
        if (board.isAttackedBy(board.kingSq[~board.side], board.side)) {
            std::cout << "[" << (i+1) << "/" << numPositions << "] " 
                      << positions[i].description << " -- SKIPPED (illegal position)\n";
            continue;
        }

        // Reset search state for each position
        search.clearHistory();
        search.clearTT();
        search.board = board;

        auto start = std::chrono::steady_clock::now();
        Move best = search.search(0, searchTimeMs);
        auto end = std::chrono::steady_clock::now();
        int ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        totalNodes += search.info.nodes;
        totalDepth += search.info.maxDepth;
        totalTimeMs += ms;

        std::cout << "[" << std::setw(2) << (i+1) << "/" << numPositions << "] " 
                  << std::left << std::setw(45) << positions[i].description
                  << " depth=" << std::setw(3) << search.info.maxDepth
                  << " nodes=" << std::setw(10) << search.info.nodes
                  << " score=" << std::setw(7) << search.info.bestScore
                  << " move=" << search.bestMoveStr()
                  << " time=" << ms << "ms"
                  << "\n";
    }

    std::cout << "\n=== Benchmark Summary ===\n";
    std::cout << "Positions:   " << numPositions << "\n";
    std::cout << "Total nodes: " << totalNodes << "\n";
    std::cout << "Total time:  " << totalTimeMs << "ms\n";
    std::cout << "Avg depth:   " << std::fixed << std::setprecision(1) 
              << (double)totalDepth / numPositions << "\n";
    std::cout << "Avg nodes:   " << totalNodes / numPositions << "\n";
    if (totalTimeMs > 0) {
        std::cout << "Overall NPS: " << (uint64_t)totalNodes * 1000 / totalTimeMs << "\n";
    }
}

// Interactive mode: play against the engine
void interactiveMode() {
    int boardArr[6][6] = {
        {3, 2, 4, 5, 2, 3},
        {1, 1, 1, 1, 1, 1},
        {0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0},
        {6, 6, 6, 6, 6, 6},
        {8, 7, 9, 10, 7, 8}
    };
    
    Search search;
    search.board.fromArray(boardArr);
    search.board.side = WHITE;
    
    std::cout << "RoboGambit 6x6 Chess Engine\n";
    std::cout << "Commands: 'move <from><to>' (e.g., 'move B1C3'), 'quit'\n";
    std::cout << "You play White, engine plays Black.\n\n";
    
    while (true) {
        search.board.print();
        
        if (search.board.side == WHITE) {
            // Human's turn
            std::cout << "Your move: ";
            std::string cmd;
            std::getline(std::cin, cmd);
            
            if (cmd == "quit") break;
            
            if (cmd.substr(0, 4) == "move" && cmd.length() >= 9) {
                std::string fromStr = cmd.substr(5, 2);
                std::string toStr = cmd.substr(7, 2);
                
                // Convert to uppercase
                if (fromStr[0] >= 'a') fromStr[0] -= 32;
                if (toStr[0] >= 'a') toStr[0] -= 32;
                
                int from = strToSquare(fromStr);
                int to = strToSquare(toStr);
                
                // Find matching legal move
                Move moves[MAX_MOVES];
                int numMoves = search.board.generateMoves(moves);
                
                bool found = false;
                for (int i = 0; i < numMoves; i++) {
                    if (moveFrom(moves[i]) == from && moveTo(moves[i]) == to) {
                        search.board.makeMove(moves[i]);
                        found = true;
                        break;
                    }
                }
                
                if (!found) {
                    std::cout << "Illegal move!\n";
                }
            }
        } else {
            // Engine's turn
            std::cout << "Engine thinking...\n";
            Move best = search.search(0, 3000);
            
            std::cout << "Engine plays: " << search.bestMoveStr() << "\n";
            std::cout << "Score: " << search.info.bestScore << " cp, Nodes: " << search.info.nodes << "\n";
            
            search.board.makeMove(best);
        }
        
        // Check game over
        Move moves[MAX_MOVES];
        int numMoves = search.board.generateMoves(moves);
        if (numMoves == 0) {
            if (search.board.inCheck()) {
                std::cout << (search.board.side == WHITE ? "Black" : "White") << " wins by checkmate!\n";
            } else {
                std::cout << "Draw by stalemate!\n";
            }
            search.board.print();
            break;
        }
    }
}

int main(int argc, char* argv[]) {
    initAll();
    
    if (argc > 1) {
        std::string mode = argv[1];
        if (mode == "perft") {
            testPerft();
        } else if (mode == "search") {
            testSearch();
        } else if (mode == "bench") {
            testBench();
        } else if (mode == "play") {
            interactiveMode();
        } else {
            std::cout << "Usage: " << argv[0] << " [perft|search|bench|play]\n";
        }
    } else {
        // Default: run both tests
        std::cout << "=== Perft Test ===\n";
        testPerft();
        std::cout << "\n=== Search Test ===\n";
        testSearch();
    }
    
    return 0;
}
