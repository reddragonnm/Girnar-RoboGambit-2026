// Texel Tuner for 6x6 Fischer Random Chess — Analytical Gradient Version
//
// Reads self-play training data and optimizes evaluation parameters
// using gradient descent to minimize MSE between sigmoid(K*eval) and game result.
//
// Key optimization: analytical gradients computed in a SINGLE PASS per epoch.
// For each position, we extract features on-the-fly (no pre-storage needed),
// compute eval = dot(features, params), then accumulate gradients.
//
// Fischer Random awareness:
//   - PSTs are tuned but with left-right symmetry enforced (pieces start anywhere)
//   - Material values, mobility, pawn structure, king safety get priority
//   - PSTs are regularized toward 0 to avoid overfitting to specific square placements
//
// Usage: ./tuner [data_file] [num_epochs] [learning_rate]
//   defaults: "tuning_data.txt", 500 epochs, lr=2.0

#include "types.h"
#include "attacks.h"
#include "zobrist.h"
#include "board.h"
#include "eval.h"
#include "search.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <random>
#include <sstream>
#include <iomanip>
#include <numeric>

void Board::print() const {}

// ============================================================
// Parameter layout (same as before)
// ============================================================
constexpr int NUM_PARAMS = 232;
constexpr int IDX_MATERIAL = 0;        // 4 values
constexpr int IDX_BISHOP_PAIR = 4;     // 1 value
constexpr int IDX_MOBILITY = 5;        // 3 values (N, B, Q)
constexpr int IDX_PASSED = 8;          // 2 values (mg_mult, eg_mult)
constexpr int IDX_DOUBLED = 10;        // 2 values
constexpr int IDX_ISOLATED = 12;       // 2 values
constexpr int IDX_KING_SAFETY = 14;    // 2 values
constexpr int IDX_PAWN_PST = 16;       // 36 values
constexpr int IDX_KNIGHT_PST = 52;     // 36 values
constexpr int IDX_BISHOP_PST = 88;     // 36 values
constexpr int IDX_QUEEN_PST = 124;     // 36 values
constexpr int IDX_KING_MG_PST = 160;   // 36 values
constexpr int IDX_KING_EG_PST = 196;   // 36 values

// PST base index for each piece type
constexpr int PST_IDX[PIECE_TYPE_COUNT] = {
    -1, IDX_PAWN_PST, IDX_KNIGHT_PST, IDX_BISHOP_PST, IDX_QUEEN_PST, IDX_KING_MG_PST
};

// Material parameter index for each piece type
constexpr int MAT_IDX[PIECE_TYPE_COUNT] = {-1, 0, 1, 2, 3, -1};

// Initial parameter values (stored for regularization toward starting point)
double initialParams[NUM_PARAMS];

// ============================================================
// Training position — compact representation
// ============================================================
struct TrainingPos {
    int8_t board[36];   // piece IDs (0-10), int8_t saves memory
    float result;       // 1.0=white win, 0.5=draw, 0.0=black win
};

// ============================================================
// Sparse feature: (parameter_index, coefficient)
// ============================================================
struct SparseFeature {
    int16_t idx;
    float val;
};

// ============================================================
// Load current engine parameters into the vector
// ============================================================
void loadCurrentParams(double params[NUM_PARAMS]) {
    params[0] = MATERIAL_VALUE[PAWN];
    params[1] = MATERIAL_VALUE[KNIGHT];
    params[2] = MATERIAL_VALUE[BISHOP];
    params[3] = MATERIAL_VALUE[QUEEN];

    params[IDX_BISHOP_PAIR] = Eval::BISHOP_PAIR_BONUS;

    params[IDX_MOBILITY + 0] = Eval::MOBILITY_WEIGHT[KNIGHT];
    params[IDX_MOBILITY + 1] = Eval::MOBILITY_WEIGHT[BISHOP];
    params[IDX_MOBILITY + 2] = Eval::MOBILITY_WEIGHT[QUEEN];

    params[IDX_PASSED + 0] = 12.53;
    params[IDX_PASSED + 1] = 36.53;
    params[IDX_DOUBLED + 0] = 14.85;
    params[IDX_DOUBLED + 1] = 17.17;
    params[IDX_ISOLATED + 0] = 6.79;
    params[IDX_ISOLATED + 1] = 4.43;

    params[IDX_KING_SAFETY + 0] = 0.00;
    params[IDX_KING_SAFETY + 1] = 60.00;

    for (int i = 0; i < 36; i++) {
        params[IDX_PAWN_PST + i] = PAWN_PST[i];
        params[IDX_KNIGHT_PST + i] = KNIGHT_PST[i];
        params[IDX_BISHOP_PST + i] = BISHOP_PST[i];
        params[IDX_QUEEN_PST + i] = QUEEN_PST[i];
        params[IDX_KING_MG_PST + i] = KING_PST_MIDGAME[i];
        params[IDX_KING_EG_PST + i] = KING_PST_ENDGAME[i];
    }
}

// ============================================================
// Extract sparse feature vector for one position and compute eval.
// Returns eval = dot(features, params).
// features are pushed onto the provided vector.
//
// The eval is:
//   mgScore = sum_pieces sign*(material + PST_mg) + bishop_pair + mobility + pawn_struct + king_safety
//   egScore = similar
//   score = (mgScore * phase + egScore * (TP - phase)) / TP
//
// Each feature[i] = d(score)/d(params[i]) = mg_coeff * tMg + eg_coeff * tEg
// ============================================================
inline double extractFeaturesAndEval(const TrainingPos& pos,
                                      const double params[NUM_PARAMS],
                                      SparseFeature* feats, int& nFeats) {
    nFeats = 0;

    const int phaseW[PIECE_TYPE_COUNT] = {0, 0, 1, 1, 3, 0};

    int phase = 0;
    int kingSq[2] = {-1, -1};
    int bishopCount[2] = {0, 0};
    uint64_t pawnBB[2] = {0, 0};
    uint64_t allOcc = 0;
    uint64_t colorOcc[2] = {0, 0};

    // First pass: build occupancy, phase, etc.
    for (int sq = 0; sq < 36; sq++) {
        int piece = pos.board[sq];
        if (piece == 0) continue;
        Color c = pieceColor(piece);
        PieceType pt = pieceTypeFromId(piece);
        allOcc |= sqBit(sq);
        colorOcc[c] |= sqBit(sq);
        phase += phaseW[pt];
        if (pt == KING) kingSq[c] = sq;
        if (pt == BISHOP) bishopCount[c]++;
        if (pt == PAWN) pawnBB[c] |= sqBit(sq);
    }
    if (phase > TOTAL_PHASE) phase = TOTAL_PHASE;

    double tMg = (double)phase / TOTAL_PHASE;
    double tEg = (double)(TOTAL_PHASE - phase) / TOTAL_PHASE;

    // We'll accumulate mg/eg coefficients, then taper.
    // Use small dense arrays for the 16 non-PST params, and handle PSTs inline.
    // Actually, for simplicity, use a small temp array for non-PST params (indices 0..15),
    // and emit PST features directly (at most ~20 pieces = ~20 PST features).

    double mgCoeff[16] = {};  // for params 0..15
    double egCoeff[16] = {};

    // Also accumulate eval directly
    double mgEval = 0.0, egEval = 0.0;

    // Second pass: material, PST, mobility
    int mobCount[2][PIECE_TYPE_COUNT] = {};

    for (int sq = 0; sq < 36; sq++) {
        int piece = pos.board[sq];
        if (piece == 0) continue;
        Color c = pieceColor(piece);
        PieceType pt = pieceTypeFromId(piece);
        double sign = (c == WHITE) ? 1.0 : -1.0;
        int pstSq = (c == WHITE) ? sq : (5 - rankOf(sq)) * 6 + fileOf(sq);

        // Material
        if (MAT_IDX[pt] >= 0) {
            int pidx = MAT_IDX[pt]; // 0..3
            mgCoeff[pidx] += sign;
            egCoeff[pidx] += sign;
            mgEval += sign * params[pidx];
            egEval += sign * params[pidx];
        }

        // PST — emit sparse feature directly
        if (pt == KING) {
            // King MG PST feature
            float fMg = (float)(sign * tMg);
            if (fMg != 0.0f) {
                feats[nFeats++] = {(int16_t)(IDX_KING_MG_PST + pstSq), fMg};
            }
            mgEval += sign * params[IDX_KING_MG_PST + pstSq];

            // King EG PST feature
            float fEg = (float)(sign * tEg);
            if (fEg != 0.0f) {
                feats[nFeats++] = {(int16_t)(IDX_KING_EG_PST + pstSq), fEg};
            }
            egEval += sign * params[IDX_KING_EG_PST + pstSq];
        } else {
            int pstBase = PST_IDX[pt];
            if (pstBase >= 0) {
                // Same PST for mg and eg: feature = sign * (tMg + tEg) = sign * 1.0
                float f = (float)sign;
                feats[nFeats++] = {(int16_t)(pstBase + pstSq), f};
                mgEval += sign * params[pstBase + pstSq];
                egEval += sign * params[pstBase + pstSq];
            }
        }

        // Mobility
        if (pt == KNIGHT || pt == BISHOP || pt == QUEEN) {
            uint64_t notOwn = ~colorOcc[c] & BOARD_MASK;
            int mob = 0;
            if (pt == KNIGHT) mob = popcount(Attacks::knightAttacks[sq] & notOwn);
            else if (pt == BISHOP) mob = popcount(Attacks::bishopAttacks(sq, allOcc) & notOwn);
            else mob = popcount(Attacks::queenAttacks(sq, allOcc) & notOwn);
            mobCount[c][pt] += mob;
        }
    }

    // Mobility features (indices 5..7)
    {
        double mN = (double)(mobCount[WHITE][KNIGHT] - mobCount[BLACK][KNIGHT]);
        mgCoeff[IDX_MOBILITY + 0] += mN;
        egCoeff[IDX_MOBILITY + 0] += mN;
        mgEval += mN * params[IDX_MOBILITY + 0];
        egEval += mN * params[IDX_MOBILITY + 0];

        double mB = (double)(mobCount[WHITE][BISHOP] - mobCount[BLACK][BISHOP]);
        mgCoeff[IDX_MOBILITY + 1] += mB;
        egCoeff[IDX_MOBILITY + 1] += mB;
        mgEval += mB * params[IDX_MOBILITY + 1];
        egEval += mB * params[IDX_MOBILITY + 1];

        double mQ = (double)(mobCount[WHITE][QUEEN] - mobCount[BLACK][QUEEN]);
        mgCoeff[IDX_MOBILITY + 2] += mQ;
        egCoeff[IDX_MOBILITY + 2] += mQ;
        mgEval += mQ * params[IDX_MOBILITY + 2];
        egEval += mQ * params[IDX_MOBILITY + 2];
    }

    // Bishop pair
    {
        double bpFeat = 0.0;
        if (bishopCount[WHITE] >= 2) bpFeat += 1.0;
        if (bishopCount[BLACK] >= 2) bpFeat -= 1.0;
        mgCoeff[IDX_BISHOP_PAIR] += bpFeat;
        egCoeff[IDX_BISHOP_PAIR] += bpFeat;
        mgEval += bpFeat * params[IDX_BISHOP_PAIR];
        egEval += bpFeat * params[IDX_BISHOP_PAIR];
    }

    // Pawn structure
    for (int c = 0; c < 2; c++) {
        double sign = (c == WHITE) ? 1.0 : -1.0;
        uint64_t pawns = pawnBB[c];
        uint64_t enemyPawns = pawnBB[1 - c];

        uint64_t tmp = pawns;
        while (tmp) {
            int sq = popLsb(tmp);
            int f = fileOf(sq);
            int r = rankOf(sq);

            // Passed pawn
            if (!(Attacks::passedPawnMask[c][sq] & enemyPawns)) {
                int distToPromo = (c == WHITE) ? (5 - r) : r;
                double bonus = (double)(5 - distToPromo)*1.5;
                mgCoeff[IDX_PASSED + 0] += sign * bonus;
                egCoeff[IDX_PASSED + 1] += sign * bonus;
                mgEval += sign * bonus * params[IDX_PASSED + 0];
                egEval += sign * bonus * params[IDX_PASSED + 1];
            }

            // Doubled pawn
            uint64_t sameFileFriends = pawns & Attacks::fileMask[f] & ~sqBit(sq);
            if (sameFileFriends) {
                int doubled = popcount(sameFileFriends);
                mgCoeff[IDX_DOUBLED + 0] -= sign * doubled;
                egCoeff[IDX_DOUBLED + 1] -= sign * doubled;
                mgEval -= sign * doubled * params[IDX_DOUBLED + 0];
                egEval -= sign * doubled * params[IDX_DOUBLED + 1];
            }

            // Isolated pawn
            if (!(pawns & Attacks::adjacentFileMask[f])) {
                mgCoeff[IDX_ISOLATED + 0] -= sign;
                egCoeff[IDX_ISOLATED + 1] -= sign;
                mgEval -= sign * params[IDX_ISOLATED + 0];
                egEval -= sign * params[IDX_ISOLATED + 1];
            }
        }
    }

    // King safety
    for (int c = 0; c < 2; c++) {
        double sign = (c == WHITE) ? 1.0 : -1.0;
        if (kingSq[c] < 0) continue;
        int ksq = kingSq[c];

        int shieldCount = popcount(Attacks::kingAttacks[ksq] & pawnBB[c]);
        mgCoeff[IDX_KING_SAFETY + 0] += sign * shieldCount;
        mgEval += sign * shieldCount * params[IDX_KING_SAFETY + 0];

        int kf = fileOf(ksq), kr = rankOf(ksq);
        // mild center king penalty
if (kf >= 2 && kf <= 3 && kr >= 2 && kr <= 3) {
    mgCoeff[IDX_KING_SAFETY + 1] -= 0.5 * sign;
    mgEval -= 0.5 * sign * params[IDX_KING_SAFETY + 1];
}
    }

    // Emit non-PST features (indices 0..15) as sparse
    for (int i = 0; i < 16; i++) {
        float f = (float)(mgCoeff[i] * tMg + egCoeff[i] * tEg);
        if (f != 0.0f) {
            feats[nFeats++] = {(int16_t)i, f};
        }
    }

    // Compute final tapered eval
    double eval = mgEval * tMg + egEval * tEg;
    return eval;
}

// ============================================================
// Sigmoid function
// ============================================================
inline double sigmoid(double K, double eval) {
    return 1.0 / (1.0 + pow(10.0, -K * eval / 200.0));
}

// ============================================================
// Enforce left-right symmetry on PSTs for Fischer Random
// ============================================================
void enforceSymmetry(double params[NUM_PARAMS]) {
    int pstOffsets[] = {IDX_KNIGHT_PST, IDX_BISHOP_PST, IDX_QUEEN_PST, IDX_KING_MG_PST, IDX_KING_EG_PST};

    for (int p : pstOffsets) {
        for (int rank = 0; rank < 6; rank++) {
            for (int file = 0; file < 3; file++) {
                int sq1 = rank * 6 + file;
                int sq2 = rank * 6 + (5 - file);
                double avg = (params[p + sq1] + params[p + sq2]) / 2.0;
                params[p + sq1] = avg;
                params[p + sq2] = avg;
            }
        }
    }

    // Pawn PST symmetry + force rank 0 and rank 5 to 0
    {
        int p = IDX_PAWN_PST;
        for (int rank = 0; rank < 6; rank++) {
            for (int file = 0; file < 3; file++) {
                int sq1 = rank * 6 + file;
                int sq2 = rank * 6 + (5 - file);
                double avg = (params[p + sq1] + params[p + sq2]) / 2.0;
                params[p + sq1] = avg;
                params[p + sq2] = avg;
            }
        }
        for (int file = 0; file < 6; file++) {
            params[p + file] = 0;
            params[p + 30 + file] = 0;
        }
    }
}

// ============================================================
// PST regularization: L2 penalty on PST values
// ============================================================
double pstRegularization(const double params[NUM_PARAMS], double lambda) {
    double reg = 0;
    int pstOffsets[] = {IDX_PAWN_PST, IDX_KNIGHT_PST, IDX_BISHOP_PST, IDX_QUEEN_PST, IDX_KING_MG_PST, IDX_KING_EG_PST};
    for (int p : pstOffsets) {
        for (int i = 0; i < 36; i++) {
            double diff = params[p + i] - initialParams[p + i];
            reg += diff * diff;
        }
    }
    return lambda * reg / (36 * 6);
}

// ============================================================
// Adam optimizer state
// ============================================================
struct AdamState {
    double m[NUM_PARAMS];
    double v[NUM_PARAMS];
    int t;

    void init() {
        memset(m, 0, sizeof(m));
        memset(v, 0, sizeof(v));
        t = 0;
    }

    double update(int idx, double grad, double lr, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8) {
        m[idx] = beta1 * m[idx] + (1.0 - beta1) * grad;
        v[idx] = beta2 * v[idx] + (1.0 - beta2) * grad * grad;
        double m_hat = m[idx] / (1.0 - pow(beta1, t));
        double v_hat = v[idx] / (1.0 - pow(beta2, t));
        return lr * m_hat / (sqrt(v_hat) + eps);
    }
};

// ============================================================
// Compute MSE + gradients in a single pass (on-the-fly features).
// If gradients is nullptr, only compute MSE.
// ============================================================
double computeErrorAndGradients(const std::vector<TrainingPos>& data,
                                 const double params[NUM_PARAMS],
                                 double K, double lambda,
                                 double gradients[NUM_PARAMS]) {
    bool computeGrad = (gradients != nullptr);
    if (computeGrad) memset(gradients, 0, sizeof(double) * NUM_PARAMS);

    const double Kfactor = K * log(10.0) / 200.0;
    const double N = (double)data.size();
    double totalError = 0.0;

    // Scratch buffer for sparse features (max ~60 per position is very generous)
    SparseFeature feats[128];

    for (size_t i = 0; i < data.size(); i++) {
        int nFeats = 0;
        double eval = extractFeaturesAndEval(data[i], params, feats, nFeats);
        double sig = sigmoid(K, eval);
        double diff = sig - data[i].result;
        totalError += diff * diff;

        if (computeGrad) {
            double common = 2.0 * diff * sig * (1.0 - sig) * Kfactor / N;
            for (int j = 0; j < nFeats; j++) {
                gradients[feats[j].idx] += common * feats[j].val;
            }
        }
    }

    double mse = totalError / N;

    // Add regularization gradient for PST params
    if (computeGrad) {
        int pstOffsets[] = {IDX_PAWN_PST, IDX_KNIGHT_PST, IDX_BISHOP_PST, IDX_QUEEN_PST, IDX_KING_MG_PST, IDX_KING_EG_PST};
        double regScale = 2.0 * lambda / (36.0 * 6.0);
        for (int p : pstOffsets) {
            for (int ii = 0; ii < 36; ii++) {
                gradients[p + ii] += regScale * (params[p + ii] - initialParams[p + ii]);
            }
        }
    }

    return mse + pstRegularization(params, lambda);
}

// ============================================================
// Compute MSE only (no gradients) — for K optimization
// ============================================================
double computeError(const std::vector<TrainingPos>& data, const double params[NUM_PARAMS], double K) {
    double totalError = 0.0;
    const double N = (double)data.size();

    // For K optimization we don't need features, just eval
    SparseFeature feats[128];
    for (size_t i = 0; i < data.size(); i++) {
        int nFeats = 0;
        double eval = extractFeaturesAndEval(data[i], params, feats, nFeats);
        double sig = sigmoid(K, eval);
        double diff = sig - data[i].result;
        totalError += diff * diff;
    }
    return totalError / N;
}

// ============================================================
// Find optimal K (sigmoid scaling constant)
// ============================================================
double findOptimalK(const std::vector<TrainingPos>& data, const double params[NUM_PARAMS]) {
    std::cout << "Finding optimal sigmoid K...\n";

    double lo = 0.1, hi = 3.0;

    for (int iter = 0; iter < 50; iter++) {
        double m1 = lo + (hi - lo) / 3.0;
        double m2 = hi - (hi - lo) / 3.0;
        double e1 = computeError(data, params, m1);
        double e2 = computeError(data, params, m2);
        if (e1 < e2) hi = m2;
        else lo = m1;
    }

    double K = (lo + hi) / 2.0;
    double error = computeError(data, params, K);
    std::cout << "Optimal K = " << std::fixed << std::setprecision(6) << K
              << " (MSE = " << std::setprecision(8) << error << ")\n";
    return K;
}

// ============================================================
// Print current parameters in C++ format
// ============================================================
void printParams(const double params[NUM_PARAMS], std::ostream& out) {
    out << "\n// ============================================================\n";
    out << "// Tuned parameters for 6x6 Fischer Random Chess\n";
    out << "// ============================================================\n\n";

    out << std::fixed;

    out << "constexpr float MATERIAL_VALUE[PIECE_TYPE_COUNT] = {\n";
    out << "    0.0f,       // NONE\n";
    out << "    " << std::setprecision(1) << params[0] << "f,     // PAWN\n";
    out << "    " << params[1] << "f,     // KNIGHT\n";
    out << "    " << params[2] << "f,     // BISHOP\n";
    out << "    " << params[3] << "f,    // QUEEN\n";
    out << "    20000.0f    // KING\n";
    out << "};\n\n";

    out << "constexpr float BISHOP_PAIR_BONUS = " << std::setprecision(1) << params[IDX_BISHOP_PAIR] << "f;\n\n";

    out << "constexpr float MOBILITY_WEIGHT[PIECE_TYPE_COUNT] = {\n";
    out << "    0.0f,   // NONE\n";
    out << "    0.0f,   // PAWN\n";
    out << "    " << std::setprecision(1) << params[IDX_MOBILITY + 0] << "f,  // KNIGHT\n";
    out << "    " << params[IDX_MOBILITY + 1] << "f,  // BISHOP\n";
    out << "    " << params[IDX_MOBILITY + 2] << "f,  // QUEEN\n";
    out << "    0.0f    // KING\n";
    out << "};\n\n";

    out << "// Passed pawn: bonus = (5-distToPromo) * mg_mult (mg), * eg_mult (eg)\n";
    out << "// Passed mg_mult = " << std::setprecision(2) << params[IDX_PASSED + 0] << "\n";
    out << "// Passed eg_mult = " << params[IDX_PASSED + 1] << "\n";
    out << "// Doubled penalty: mg=" << params[IDX_DOUBLED + 0]
        << " eg=" << params[IDX_DOUBLED + 1] << "\n";
    out << "// Isolated penalty: mg=" << params[IDX_ISOLATED + 0]
        << " eg=" << params[IDX_ISOLATED + 1] << "\n\n";

    out << "// King safety: pawn_shield=" << params[IDX_KING_SAFETY + 0]
        << " center_penalty_mg=" << params[IDX_KING_SAFETY + 1] << "\n\n";

    auto printPST = [&out](const char* name, const double* pst) {
        out << "constexpr float " << name << "[NUM_SQUARES] = {\n";
        for (int r = 0; r < 6; r++) {
            out << "    ";
            for (int f = 0; f < 6; f++) {
                double val = pst[r * 6 + f];
                if (f > 0) out << ",";
                out << std::setw(8) << std::setprecision(2) << val << "f";
            }
            if (r < 5) out << ",";
            out << "\n";
        }
        out << "};\n\n";
    };

    printPST("PAWN_PST", &params[IDX_PAWN_PST]);
    printPST("KNIGHT_PST", &params[IDX_KNIGHT_PST]);
    printPST("BISHOP_PST", &params[IDX_BISHOP_PST]);
    printPST("QUEEN_PST", &params[IDX_QUEEN_PST]);
    printPST("KING_PST_MIDGAME", &params[IDX_KING_MG_PST]);
    printPST("KING_PST_ENDGAME", &params[IDX_KING_EG_PST]);
}

// ============================================================
// Main tuning loop
// ============================================================
#ifndef TUNER_NO_MAIN
int main(int argc, char** argv) {
    std::string dataFile = "tuning_data.txt";
    int numEpochs = 500;
    double learningRate = 0.8;
    double pstLambda = 0.0002;

    if (argc > 1) dataFile = argv[1];
    if (argc > 2) numEpochs = std::atoi(argv[2]);
    if (argc > 3) learningRate = std::atof(argv[3]);

    std::cout << "=== Texel Tuner for 6x6 Fischer Random Chess ===\n";
    std::cout << "=== Analytical Gradient Version (On-the-fly) ===\n";
    std::cout << "Data file: " << dataFile << "\n";
    std::cout << "Epochs: " << numEpochs << "\n";
    std::cout << "Learning rate: " << learningRate << "\n";
    std::cout << "PST regularization lambda: " << pstLambda << "\n\n";

    Attacks::init();
    Zobrist::init();

    // ----------------------------------------------------------
    // Load training data
    // ----------------------------------------------------------
    std::cout << "Loading training data...\n";
    std::vector<TrainingPos> data;
    {
        std::ifstream in(dataFile);
        if (!in) {
            std::cerr << "Error: cannot open " << dataFile << "\n";
            return 1;
        }

        std::string line;
        while (std::getline(in, line)) {
            std::istringstream iss(line);
            TrainingPos pos;
            bool ok = true;
            for (int i = 0; i < 36; i++) {
                int val;
                if (!(iss >> val)) { ok = false; break; }
                pos.board[i] = (int8_t)val;
            }
            if (!ok) continue;
            int side;
            double result;
            if (!(iss >> side >> result)) continue;
            pos.result = (float)result;
            data.push_back(pos);
        }
    }

    std::cout << "Loaded " << data.size() << " positions\n";
    if (data.empty()) {
        std::cerr << "Error: no training data loaded\n";
        return 1;
    }

    double dataMB = (double)data.size() * sizeof(TrainingPos) / (1024.0 * 1024.0);
    std::cout << "Data memory: " << std::fixed << std::setprecision(1) << dataMB << " MB\n";

    int wWins = 0, bWins = 0, draws = 0;
    for (auto& p : data) {
        if (p.result > 0.9f) wWins++;
        else if (p.result < 0.1f) bWins++;
        else draws++;
    }
    std::cout << "Distribution: W=" << wWins << " D=" << draws << " B=" << bWins << "\n\n";

    // ----------------------------------------------------------
    // Initialize parameters
    // ----------------------------------------------------------
    double params[NUM_PARAMS];
    loadCurrentParams(params);
    enforceSymmetry(params);
    memcpy(initialParams, params, sizeof(params));

    // Find optimal K
    double K = findOptimalK(data, params);

    // Initialize Adam optimizer
    AdamState adam;
    adam.init();

    double bestError = computeErrorAndGradients(data, params, K, pstLambda, nullptr);
    double bestParams[NUM_PARAMS];
    memcpy(bestParams, params, sizeof(params));

    std::cout << "\nStarting optimization (analytical gradients, on-the-fly features)...\n";
    std::cout << "Initial error: " << std::fixed << std::setprecision(8) << bestError << "\n\n";

    auto startTime = std::chrono::steady_clock::now();

    // Parameter bounds
    double minBound[NUM_PARAMS], maxBound[NUM_PARAMS];
    for (int i = 0; i < NUM_PARAMS; i++) {
        minBound[i] = -500;
        maxBound[i] = 500;
    }
    minBound[0] = 50; maxBound[0] = 200;
    minBound[1] = 200; maxBound[1] = 500;
    minBound[2] = 200; maxBound[2] = 500;
    minBound[3] = 700; maxBound[3] = 1500;
    minBound[IDX_BISHOP_PAIR] = 0; maxBound[IDX_BISHOP_PAIR] = 100;
    for (int i = 0; i < 3; i++) {
        minBound[IDX_MOBILITY + i] = 0;
        maxBound[IDX_MOBILITY + i] = 25;
    }
    for (int i = 0; i < 2; i++) {
        minBound[IDX_PASSED + i] = 0; maxBound[IDX_PASSED + i] = 60;
        minBound[IDX_DOUBLED + i] = 0; maxBound[IDX_DOUBLED + i] = 50;
        minBound[IDX_ISOLATED + i] = 0; maxBound[IDX_ISOLATED + i] = 40;
    }
    minBound[IDX_KING_SAFETY + 0] = 0; maxBound[IDX_KING_SAFETY + 0] = 40;
    minBound[IDX_KING_SAFETY + 1] = 0; maxBound[IDX_KING_SAFETY + 1] = 60;
    for (int i = IDX_PAWN_PST; i < NUM_PARAMS; i++) {
        minBound[i] = -80;
        maxBound[i] = 80;
    }
    for (int i = IDX_KING_MG_PST; i < IDX_KING_MG_PST + 36; i++) {
        minBound[i] = -100;
        maxBound[i] = 100;
    }
    for (int i = IDX_KING_EG_PST; i < IDX_KING_EG_PST + 36; i++) {
        minBound[i] = -60;
        maxBound[i] = 60;
    }

    // Fixed params
    bool skipParam[NUM_PARAMS] = {};
    for (int i = IDX_PAWN_PST; i < IDX_PAWN_PST + 36; i++) {
        int rank = (i - IDX_PAWN_PST) / 6;
        if (rank == 0 || rank == 5) skipParam[i] = true;
    }

    // ============================================================
    // Optimization loop
    // ============================================================
    for (int epoch = 0; epoch < numEpochs; epoch++) {
        adam.t = epoch + 1;

        // Compute analytical gradients — SINGLE PASS over all data
        double gradients[NUM_PARAMS];
        double error = computeErrorAndGradients(data, params, K, pstLambda, gradients);

        // Zero out gradients for fixed params
        for (int i = 0; i < NUM_PARAMS; i++) {
            if (skipParam[i]) gradients[i] = 0.0;
        }

        // Update parameters with Adam
        for (int i = 0; i < NUM_PARAMS; i++) {
            if (skipParam[i] || gradients[i] == 0.0) continue;

            double delta = adam.update(i, gradients[i], learningRate);
            params[i] -= delta;

            if (params[i] < minBound[i]) params[i] = minBound[i];
            if (params[i] > maxBound[i]) params[i] = maxBound[i];
        }

        enforceSymmetry(params);

        if (error < bestError) {
            bestError = error;
            memcpy(bestParams, params, sizeof(params));
        }

        if ((epoch + 1) % 10 == 0 || epoch == 0) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - startTime).count();
            std::cout << "Epoch " << std::setw(4) << (epoch + 1)
                      << " | Error: " << std::fixed << std::setprecision(8) << error
                      << " | Best: " << bestError
                      << " | Time: " << std::setprecision(1) << elapsed << "s"
                      << " | Mat: " << (int)round(params[0]) << "/" << (int)round(params[1])
                      << "/" << (int)round(params[2]) << "/" << (int)round(params[3])
                      << " | BP:" << (int)round(params[IDX_BISHOP_PAIR])
                      << " | Mob:" << (int)round(params[IDX_MOBILITY]) << "/"
                      << (int)round(params[IDX_MOBILITY+1]) << "/" << (int)round(params[IDX_MOBILITY+2])
                      << "\n";
        }

        // Re-optimize K every 100 epochs
        if ((epoch + 1) % 100 == 0 && (epoch + 1) < numEpochs) {
            K = findOptimalK(data, params);
        }
    }

    // Re-optimize K with best parameters
    std::cout << "\nRe-optimizing K with best parameters...\n";
    K = findOptimalK(data, bestParams);
    double finalError = computeError(data, bestParams, K);
    std::cout << "Final MSE: " << std::fixed << std::setprecision(8) << finalError << "\n";

    std::cout << "\n=== BEST TUNED PARAMETERS ===\n";
    printParams(bestParams, std::cout);

    {
        std::ofstream out("tuned_params.txt");
        if (out) {
            printParams(bestParams, out);
            std::cout << "\nTuned parameters also written to tuned_params.txt\n";
        }
    }

    return 0;
}
#endif
