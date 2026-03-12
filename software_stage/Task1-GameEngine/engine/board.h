#pragma once
#include "attacks.h"
#include "pst.h"
#include "types.h"
#include "zobrist.h"
#include <array>
#include <cassert>
#include <cstring>
struct UndoInfo {
  Move move;
  int captured; // piece mailbox value at target square (0 if none)
  PieceType capturedType;
  Color capturedColor;
  uint64_t hash;
  int halfmoveClock;
};

// Fixed-size undo stack - eliminates heap allocations in makeMove/unmakeMove
constexpr int MAX_GAME_PLY = 512;

class Board {
public:
  // Bitboards per color per piece type
  Bitboard pieces[COLOR_COUNT][PIECE_TYPE_COUNT];

  // Occupancy bitboards
  Bitboard occupied[COLOR_COUNT];
  Bitboard allOccupied;

  // Mailbox: what piece is on each square (0 = empty, else pieceId 1-10)
  int mailbox[NUM_SQUARES];

  // Side to move
  Color side;

  // Zobrist hash
  uint64_t hash;

  // Track captured pieces for promotion rule
  // capturedCount[color][pieceType] = how many of that piece have been captured
  int capturedCount[COLOR_COUNT][PIECE_TYPE_COUNT];

  // How many of each piece type each side started with (for promotion logic)
  int initialCount[COLOR_COUNT][PIECE_TYPE_COUNT];

  // Half-move clock for 50-move rule (optional)
  int halfmoveClock;
  int fullmoveNumber;

  // Undo stack (fixed-size array - no heap allocations)
  UndoInfo undoStack[MAX_GAME_PLY];
  int undoCount;

  // King square cache
  int kingSq[COLOR_COUNT];

  // Incremental PST + material scores (from white's perspective)
  float mgPst[COLOR_COUNT]; // midgame material+PST score per color
  float egPst[COLOR_COUNT]; // endgame material+PST score per color
  int phase;                // game phase for tapered eval

  Board() { clear(); }

  void clear() {
    memset(pieces, 0, sizeof(pieces));
    memset(occupied, 0, sizeof(occupied));
    allOccupied = 0;
    memset(mailbox, 0, sizeof(mailbox));
    side = WHITE;
    hash = 0;
    memset(capturedCount, 0, sizeof(capturedCount));
    memset(initialCount, 0, sizeof(initialCount));
    halfmoveClock = 0;
    fullmoveNumber = 1;
    undoCount = 0;
    kingSq[WHITE] = kingSq[BLACK] = -1;
    mgPst[WHITE] = mgPst[BLACK] = 0.0f;
    egPst[WHITE] = egPst[BLACK] = 0.0f;
    phase = 0;
  }

  void putPiece(Color c, PieceType pt, int sq) {
    pieces[c][pt] |= sqBit(sq);
    occupied[c] |= sqBit(sq);
    allOccupied |= sqBit(sq);
    mailbox[sq] = pieceId(c, pt);
    hash ^= Zobrist::pieceKeys[c][pt][sq];
    if (pt == KING)
      kingSq[c] = sq;

    // Incremental PST+material update
    mgPst[c] += mgPstValue(pt, sq, c);
    egPst[c] += egPstValue(pt, sq, c);
    phase += PHASE_WEIGHT[pt];
  }

  void removePiece(Color c, PieceType pt, int sq) {
    pieces[c][pt] &= ~sqBit(sq);
    occupied[c] &= ~sqBit(sq);
    allOccupied &= ~sqBit(sq);
    mailbox[sq] = 0;
    hash ^= Zobrist::pieceKeys[c][pt][sq];

    // Incremental PST+material update
    mgPst[c] -= mgPstValue(pt, sq, c);
    egPst[c] -= egPstValue(pt, sq, c);
    phase -= PHASE_WEIGHT[pt];
  }

  // Optimized combined move: remove from 'from' and place on 'to' for same
  // piece type Saves redundant occupied/allOccupied clear+set, phase sub+add,
  // and mailbox zero+set
  void movePiece(Color c, PieceType pt, int from, int to) {
    Bitboard fromBit = sqBit(from);
    Bitboard toBit = sqBit(to);
    Bitboard toggle = fromBit | toBit;
    pieces[c][pt] ^= toggle;
    occupied[c] ^= toggle;
    allOccupied ^= toggle;
    mailbox[from] = 0;
    mailbox[to] = pieceId(c, pt);
    hash ^= Zobrist::pieceKeys[c][pt][from] ^ Zobrist::pieceKeys[c][pt][to];
    if (pt == KING)
      kingSq[c] = to;

    // Incremental PST update (material and phase cancel out for same piece
    // type)
    mgPst[c] += mgPstValue(pt, to, c) - mgPstValue(pt, from, c);
    egPst[c] += egPstValue(pt, to, c) - egPstValue(pt, from, c);
  }

  // Set up from a 6x6 numpy-style array (row 0 = rank 1)
  // Values: 0=empty, 1-5=white pieces, 6-10=black pieces
  void fromArray(const int board[6][6]) {
    clear();
    for (int rank = 0; rank < 6; rank++) {
      for (int file = 0; file < 6; file++) {
        int val = board[rank][file];
        if (val == 0)
          continue;
        Color c = pieceColor(val);
        PieceType pt = pieceTypeFromId(val);
        int sq = makeSquare(file, rank);
        putPiece(c, pt, sq);
      }
    }
    computeInitialCounts();
  }

  // Alternative: set up from flat array[36]
  void fromFlatArray(const int board[36]) {
    clear();
    for (int sq = 0; sq < 36; sq++) {
      int val = board[sq];
      if (val == 0)
        continue;
      Color c = pieceColor(val);
      PieceType pt = pieceTypeFromId(val);
      putPiece(c, pt, sq);
    }
    computeInitialCounts();
  }

  void computeInitialCounts() {
    // Count current pieces as initial (call at setup time — only valid
    // when the board is truly at the starting position!)
    for (int c = 0; c < COLOR_COUNT; c++)
      for (int pt = 1; pt < PIECE_TYPE_COUNT; pt++)
        initialCount[c][pt] = popcount(pieces[c][pt]);
  }

  // Set the known RoboGambit starting piece counts.
  // Every Fischer Random game starts with: 6P, 2N, 2B, 1Q, 1K per side.
  // Call this instead of computeInitialCounts() when setting up a mid-game
  // board received from the competition evaluator.
  void setRoboGambitInitialCounts() {
    for (int c = 0; c < COLOR_COUNT; c++) {
      initialCount[c][NONE] = 0;
      initialCount[c][PAWN] = 6;
      initialCount[c][KNIGHT] = 2;
      initialCount[c][BISHOP] = 2;
      initialCount[c][QUEEN] = 1;
      initialCount[c][KING] = 1;
    }
  }

  // Derive capturedCount from initialCount minus current piece counts.
  // Must be called after setRoboGambitInitialCounts() (or
  // computeInitialCounts() on a true starting position) and after all pieces
  // are placed on the board.
  void computeCapturedFromInitial() {
    for (int c = 0; c < COLOR_COUNT; c++) {
      for (int pt = 1; pt < PIECE_TYPE_COUNT; pt++) {
        int current = popcount(pieces[c][pt]);
        capturedCount[c][pt] = initialCount[c][pt] - current;
        if (capturedCount[c][pt] < 0)
          capturedCount[c][pt] = 0;
      }
    }
  }

  // Check if a square is attacked by the given color
  bool isAttackedBy(int sq, Color by) const {
    // Pawn attacks
    if (Attacks::pawnAttacks[~by][sq] & pieces[by][PAWN])
      return true;
    // Knight attacks
    if (Attacks::knightAttacks[sq] & pieces[by][KNIGHT])
      return true;
    // King attacks
    if (Attacks::kingAttacks[sq] & pieces[by][KING])
      return true;
    // Bishop attacks
    if (Attacks::bishopAttacks(sq, allOccupied) & pieces[by][BISHOP])
      return true;
    // Queen attacks
    if (Attacks::queenAttacks(sq, allOccupied) & pieces[by][QUEEN])
      return true;

    return false;
  }

  bool inCheck() const { return isAttackedBy(kingSq[side], ~side); }

  bool inCheck(Color c) const { return isAttackedBy(kingSq[c], ~c); }

  // Insufficient material draw detection.
  // Covers all cases where neither side can force checkmate:
  //   KvK, KNvK, KBvK, KNvKN, KBvKB (any combo), KNvKB
  // Positions with pawns or queens can always lead to mate — not a draw.
  bool isInsufficientMaterial() const {
    // Any pawn or queen means material is sufficient
    if (pieces[WHITE][PAWN] | pieces[BLACK][PAWN] |
        pieces[WHITE][QUEEN] | pieces[BLACK][QUEEN])
      return false;

    int wN = popcount(pieces[WHITE][KNIGHT]);
    int bN = popcount(pieces[BLACK][KNIGHT]);
    int wB = popcount(pieces[WHITE][BISHOP]);
    int bB = popcount(pieces[BLACK][BISHOP]);
    int wMinor = wN + wB;
    int bMinor = bN + bB;

    // KvK
    if (wMinor == 0 && bMinor == 0) return true;
    // KXvK (one minor piece total)
    if (wMinor + bMinor == 1) return true;
    // KNvKN, KNvKB, KBvKN, KBvKB — two minors total, one per side
    if (wMinor == 1 && bMinor == 1) return true;

    return false;
  }

  // Get least valuable attacker of a square for SEE
  Bitboard getLeastValuableAttacker(int sq, Bitboard occ, Color by,
                                    PieceType &pt) const {
    // Pawns
    Bitboard attackers = Attacks::pawnAttacks[~by][sq] & pieces[by][PAWN] & occ;
    if (attackers) {
      pt = PAWN;
      return attackers & (~attackers + 1);
    } // isolate LSB

    // Knights
    attackers = Attacks::knightAttacks[sq] & pieces[by][KNIGHT] & occ;
    if (attackers) {
      pt = KNIGHT;
      return attackers & (~attackers + 1);
    }

    // Bishops
    attackers = Attacks::bishopAttacks(sq, occ) & pieces[by][BISHOP] & occ;
    if (attackers) {
      pt = BISHOP;
      return attackers & (~attackers + 1);
    }

    // Queens
    attackers = Attacks::queenAttacks(sq, occ) & pieces[by][QUEEN] & occ;
    if (attackers) {
      pt = QUEEN;
      return attackers & (~attackers + 1);
    }

    // King
    attackers = Attacks::kingAttacks[sq] & pieces[by][KING] & occ;
    if (attackers) {
      pt = KING;
      return attackers & (~attackers + 1);
    }

    return 0;
  }

  // Static Exchange Evaluation
  // Returns the material gain/loss from a sequence of captures on target square
  // Positive = winning exchange, negative = losing exchange
  int see(Move m) const {
    static constexpr int SEE_VALUES[] = {0, 100, 340, 350, 1000, 20000};

    int from = moveFrom(m);
    int to = moveTo(m);

    int target = mailbox[to];
    PieceType attacker = pieceTypeFromId(mailbox[from]);

    if (target == 0 && movePromo(m) == NONE)
      return 0; // quiet move

    int gain[32];
    int depth = 0;
    Bitboard occ = allOccupied;

    // Initial capture value
    gain[0] = (target != 0) ? SEE_VALUES[pieceTypeFromId(target)] : 0;
    if (movePromo(m) != NONE) {
      gain[0] += SEE_VALUES[movePromo(m)] - SEE_VALUES[PAWN];
    }

    Color sideToCapture = ~pieceColor(mailbox[from]);
    occ ^= sqBit(from); // remove attacker from occupancy

    PieceType nextPt;

    while (true) {
      depth++;
      gain[depth] =
          SEE_VALUES[attacker] - gain[depth - 1]; // assume we lose the piece

      // Pruning: if even getting the piece isn't enough to beat, stop
      if (std::max(-gain[depth - 1], gain[depth]) < 0)
        break;

      Bitboard lva = getLeastValuableAttacker(to, occ, sideToCapture, nextPt);
      if (!lva)
        break;

      occ ^= lva; // remove this attacker
      attacker = nextPt;
      sideToCapture = ~sideToCapture;
    }

    // Minimax the gain array
    while (--depth) {
      gain[depth - 1] = -std::max(-gain[depth - 1], gain[depth]);
    }

    return gain[0];
  }

  // Get all available promotion piece types for a pawn of the given color
  // A pawn can only promote to a piece type that has been captured from the
  // SAME side i.e., you can promote to a piece type if you've lost one of those
  // More precisely: you can promote to type T if capturedCount[c][T] > 0
  // AND current count of T on board < initialCount[c][T]
  // Actually re-reading the rules: "Only available for pieces already lost"
  // This means: you can promote to piece type T if you currently have fewer of
  // T than you started with
  void getPromotionTypes(Color c, PieceType promos[], int &numPromos) const {
    numPromos = 0;
    for (int pt = KNIGHT; pt <= QUEEN; pt++) {
      int currentCount = popcount(pieces[c][pt]);
      if (currentCount < initialCount[c][pt]) {
        promos[numPromos++] = PieceType(pt);
      }
    }
  }

  // Compute pinned pieces and the ray mask along which they can move
  // Returns a bitboard of pinned pieces belonging to 'us'
  // pinRay[sq] is set for pinned piece squares to indicate the ray they must
  // stay on
  Bitboard computePinned(Color us, Bitboard pinRay[NUM_SQUARES]) const {
    Bitboard pinned = 0;
    int ksq = kingSq[us];
    Color them = ~us;

    // For each direction, check if there's exactly one of our pieces between
    // king and an enemy sliding attacker (bishop/queen on diagonals, queen on
    // straights)
    for (int dir = 0; dir < 8; dir++) {
      const Attacks::RayData &rd = Attacks::rays[ksq][dir];
      if (!rd.mask)
        continue;

      // What enemy pieces can attack along this direction?
      bool isDiag = (dir == 0 || dir == 2 || dir == 5 || dir == 7);
      bool isStraight = (dir == 1 || dir == 3 || dir == 4 || dir == 6);

      Bitboard enemySliders = 0;
      if (isDiag)
        enemySliders = pieces[them][BISHOP] | pieces[them][QUEEN];
      else
        enemySliders = pieces[them][QUEEN]; // no rooks in this variant

      if (!(rd.mask & enemySliders))
        continue; // no enemy slider on this ray

      // Get the actual attacks from king along this ray (considering all
      // occupancy)
      int idx = (int)pext_u64(allOccupied, rd.mask);
      Bitboard rayAtk =
          rd.attacks[idx]; // squares the king "sees" along this ray

      // Check for a potential pinner: an enemy slider on this ray
      // We need to look through our own pieces to find if there's a pinner
      // behind
      Bitboard ourOnRay = rayAtk & occupied[us];

      if (popcount(ourOnRay) != 1)
        continue; // need exactly 1 of our pieces blocking

      // Remove our blocker and see if an enemy slider is then visible
      int blockerSq = lsb(ourOnRay);
      Bitboard occWithoutBlocker = allOccupied ^ sqBit(blockerSq);
      int idx2 = (int)pext_u64(occWithoutBlocker, rd.mask);
      Bitboard extendedRay = rd.attacks[idx2];

      if (extendedRay & enemySliders) {
        // The piece at blockerSq is pinned
        pinned |= sqBit(blockerSq);
        // The pin ray includes the pinner's square and all squares between king
        // and pinner (including the blocker's square itself)
        pinRay[blockerSq] =
            extendedRay | sqBit(ksq); // include king sq for the line
      }
    }

    return pinned;
  }

  // Check if a move from 'from' to 'to' is along the pin ray
  // (the piece must stay on the line between king and pinner)
  static bool isAlongPinRay(int from, int to, Bitboard ray) {
    return (ray & sqBit(to)) != 0;
  }

  // Helper to add pawn moves with promotion handling
  inline int addPawnMoves(Move moves[], int count, int from, int to,
                          int promoRank, Color us) {
    if (rankOf(to) == promoRank) {
      PieceType promos[4];
      int numPromos;
      getPromotionTypes(us, promos, numPromos);
      for (int i = 0; i < numPromos; i++)
        moves[count++] = encodeMove(from, to, promos[i]);
    } else {
      moves[count++] = encodeMove(from, to);
    }
    return count;
  }

  // Generate all legal moves for the side to move
  int generateMoves(Move moves[MAX_MOVES]) {
    int count = 0;
    Color us = side;
    Color them = ~us;
    Bitboard ourPieces = occupied[us];
    Bitboard theirPieces = occupied[them];
    int ksq = kingSq[us];

    // Step 1: Compute checkers
    Bitboard checkers = attackersOf(ksq, them);
    int numCheckers = popcount(checkers);

    // Step 2: Generate king moves (always need isAttackedBy check, but at most
    // 8 moves)
    {
      Bitboard targets = Attacks::kingAttacks[ksq] & ~ourPieces;
      // Temporarily remove king from occupancy for isAttackedBy check
      // (a king can't block its own check by staying in line)
      Bitboard savedOcc = allOccupied;
      allOccupied ^= sqBit(ksq);
      occupied[us] ^= sqBit(ksq);
      pieces[us][KING] ^= sqBit(ksq);

      while (targets) {
        int to = popLsb(targets);
        if (!isAttackedBy(to, them)) {
          moves[count++] = encodeMove(ksq, to);
        }
      }

      allOccupied = savedOcc;
      occupied[us] |= sqBit(ksq);
      pieces[us][KING] |= sqBit(ksq);
    }

    // Double check: only king moves are legal
    if (numCheckers >= 2)
      return count;

    // Step 3: Compute pins (no memset needed - pinRayArr is only read for
    // pinned squares)
    Bitboard pinRayArr[NUM_SQUARES];
    Bitboard pinned = computePinned(us, pinRayArr);

    // Step 4: If in single check, non-king moves must block or capture the
    // checker
    Bitboard checkMask =
        ~0ULL; // mask of valid target squares for non-king pieces
    if (numCheckers == 1) {
      int checkerSq = lsb(checkers);
      PieceType checkerType = pieceTypeFromId(mailbox[checkerSq]);

      // Can capture the checker
      checkMask = checkers;

      // Can also block if the checker is a sliding piece (bishop or queen)
      if (checkerType == BISHOP || checkerType == QUEEN) {
        // Find squares between king and checker
        checkMask |= betweenMask(ksq, checkerSq);
      }
      // Knights and pawns can only be captured, not blocked (checkMask stays as
      // just the checker)
    }

    // Step 5: Generate non-king moves
    int promoRank = (us == WHITE) ? 5 : 0;
    int pawnDir = (us == WHITE) ? 6 : -6;

    // --- PAWNS ---
    {
      Bitboard pawns = pieces[us][PAWN];
      while (pawns) {
        int from = popLsb(pawns);
        bool isPinned = (pinned & sqBit(from)) != 0;

        // Forward move
        int to = from + pawnDir;
        if (to >= 0 && to < 36 && mailbox[to] == 0) {
          if (checkMask & sqBit(to)) {
            if (!isPinned || isAlongPinRay(from, to, pinRayArr[from])) {
              count = addPawnMoves(moves, count, from, to, promoRank, us);
            }
          }
        }

        // Captures
        Bitboard captures =
            Attacks::pawnAttacks[us][from] & theirPieces & checkMask;
        while (captures) {
          int capSq = popLsb(captures);
          if (!isPinned || isAlongPinRay(from, capSq, pinRayArr[from])) {
            count = addPawnMoves(moves, count, from, capSq, promoRank, us);
          }
        }
      }
    }

    // --- KNIGHTS ---
    {
      Bitboard knights =
          pieces[us][KNIGHT] & ~pinned; // pinned knights can never move
      while (knights) {
        int from = popLsb(knights);
        Bitboard targets =
            Attacks::knightAttacks[from] & ~ourPieces & checkMask;
        while (targets) {
          moves[count++] = encodeMove(from, popLsb(targets));
        }
      }
    }

    // --- BISHOPS ---
    {
      Bitboard bishops = pieces[us][BISHOP];
      while (bishops) {
        int from = popLsb(bishops);
        bool isPinned = (pinned & sqBit(from)) != 0;
        Bitboard targets =
            Attacks::bishopAttacks(from, allOccupied) & ~ourPieces & checkMask;
        if (isPinned)
          targets &= pinRayArr[from];
        while (targets) {
          moves[count++] = encodeMove(from, popLsb(targets));
        }
      }
    }

    // --- QUEEN ---
    {
      Bitboard queens = pieces[us][QUEEN];
      while (queens) {
        int from = popLsb(queens);
        bool isPinned = (pinned & sqBit(from)) != 0;
        Bitboard targets =
            Attacks::queenAttacks(from, allOccupied) & ~ourPieces & checkMask;
        if (isPinned)
          targets &= pinRayArr[from];
        while (targets) {
          moves[count++] = encodeMove(from, popLsb(targets));
        }
      }
    }

    return count;
  }

  // Get all pieces of a given color attacking a square
  Bitboard attackersOf(int sq, Color by) const {
    Bitboard att = 0;
    att |= Attacks::pawnAttacks[~by][sq] & pieces[by][PAWN];
    att |= Attacks::knightAttacks[sq] & pieces[by][KNIGHT];
    att |= Attacks::kingAttacks[sq] & pieces[by][KING];
    att |= Attacks::bishopAttacks(sq, allOccupied) & pieces[by][BISHOP];
    att |= Attacks::queenAttacks(sq, allOccupied) & pieces[by][QUEEN];
    return att;
  }

  // Compute squares strictly between sq1 and sq2 (on a line)
  // Returns 0 if they're not on a line or adjacent
  Bitboard betweenMask(int sq1, int sq2) const {
    Bitboard between = 0;
    int f1 = fileOf(sq1), r1 = rankOf(sq1);
    int f2 = fileOf(sq2), r2 = rankOf(sq2);

    int df = 0, dr = 0;
    if (f2 > f1)
      df = 1;
    else if (f2 < f1)
      df = -1;
    if (r2 > r1)
      dr = 1;
    else if (r2 < r1)
      dr = -1;

    // Verify they're on a line (same rank, file, or diagonal)
    if (df == 0 && dr == 0)
      return 0;
    if (df != 0 && dr != 0 && std::abs(f2 - f1) != std::abs(r2 - r1))
      return 0;
    if (df == 0 && f1 != f2)
      return 0;
    if (dr == 0 && r1 != r2)
      return 0;

    int cf = f1 + df, cr = r1 + dr;
    while (cf != f2 || cr != r2) {
      between |= sqBit(makeSquare(cf, cr));
      cf += df;
      cr += dr;
    }
    return between;
  }

  // Generate only captures and promotions (for quiescence search)
  int generateCaptures(Move moves[MAX_MOVES]) {
    int count = 0;
    Color us = side;
    Color them = ~us;
    Bitboard ourPieces = occupied[us];
    Bitboard theirPieces = occupied[them];
    int ksq = kingSq[us];

    Bitboard checkers = attackersOf(ksq, them);
    int numCheckers = popcount(checkers);

    // King captures
    {
      Bitboard targets = Attacks::kingAttacks[ksq] & theirPieces;
      Bitboard savedOcc = allOccupied;
      allOccupied ^= sqBit(ksq);
      occupied[us] ^= sqBit(ksq);
      pieces[us][KING] ^= sqBit(ksq);

      while (targets) {
        int to = popLsb(targets);
        if (!isAttackedBy(to, them)) {
          moves[count++] = encodeMove(ksq, to);
        }
      }

      allOccupied = savedOcc;
      occupied[us] |= sqBit(ksq);
      pieces[us][KING] |= sqBit(ksq);
    }

    if (numCheckers >= 2)
      return count;

    Bitboard pinRayArr[NUM_SQUARES];
    Bitboard pinned = computePinned(us, pinRayArr);

    Bitboard checkMask = ~0ULL;
    if (numCheckers == 1) {
      int checkerSq = lsb(checkers);
      PieceType checkerType = pieceTypeFromId(mailbox[checkerSq]);
      checkMask = checkers;
      if (checkerType == BISHOP || checkerType == QUEEN) {
        checkMask |= betweenMask(ksq, checkerSq);
      }
    }

    // Only generate captures (& promotions)
    Bitboard capTargets = theirPieces & checkMask;
    int promoRank = (us == WHITE) ? 5 : 0;
    int pawnDir = (us == WHITE) ? 6 : -6;

    // Pawn captures + promotions
    {
      Bitboard pawns = pieces[us][PAWN];
      while (pawns) {
        int from = popLsb(pawns);
        bool isPinned = (pinned & sqBit(from)) != 0;

        // Pawn push to promotion rank
        int to = from + pawnDir;
        if (to >= 0 && to < 36 && mailbox[to] == 0 && rankOf(to) == promoRank) {
          if (checkMask & sqBit(to)) {
            if (!isPinned || isAlongPinRay(from, to, pinRayArr[from])) {
              count = addPawnMoves(moves, count, from, to, promoRank, us);
            }
          }
        }

        // Captures
        Bitboard captures =
            Attacks::pawnAttacks[us][from] & theirPieces & checkMask;
        while (captures) {
          int capSq = popLsb(captures);
          if (!isPinned || isAlongPinRay(from, capSq, pinRayArr[from])) {
            count = addPawnMoves(moves, count, from, capSq, promoRank, us);
          }
        }
      }
    }

    // Knight captures
    {
      Bitboard knights = pieces[us][KNIGHT] & ~pinned;
      while (knights) {
        int from = popLsb(knights);
        Bitboard targets = Attacks::knightAttacks[from] & capTargets;
        while (targets) {
          moves[count++] = encodeMove(from, popLsb(targets));
        }
      }
    }

    // Bishop captures
    {
      Bitboard bishops = pieces[us][BISHOP];
      while (bishops) {
        int from = popLsb(bishops);
        bool isPinned = (pinned & sqBit(from)) != 0;
        Bitboard targets =
            Attacks::bishopAttacks(from, allOccupied) & capTargets;
        if (isPinned)
          targets &= pinRayArr[from];
        while (targets) {
          moves[count++] = encodeMove(from, popLsb(targets));
        }
      }
    }

    // Queen captures
    {
      Bitboard queens = pieces[us][QUEEN];
      while (queens) {
        int from = popLsb(queens);
        bool isPinned = (pinned & sqBit(from)) != 0;
        Bitboard targets =
            Attacks::queenAttacks(from, allOccupied) & capTargets;
        if (isPinned)
          targets &= pinRayArr[from];
        while (targets) {
          moves[count++] = encodeMove(from, popLsb(targets));
        }
      }
    }

    return count;
  }

  // Make a move (no legality check - assumes legal)
  void makeMove(Move m) {
    UndoInfo undo;
    undo.move = m;
    undo.hash = hash;
    undo.halfmoveClock = halfmoveClock;

    int from = moveFrom(m);
    int to = moveTo(m);
    PieceType promo = movePromo(m);

    Color us = side;
    Color them = ~us;

    int movingPiece = mailbox[from];
    PieceType movingType = pieceTypeFromId(movingPiece);

    // Handle capture
    int capturedPiece = mailbox[to];
    undo.captured = capturedPiece;
    if (capturedPiece) {
      undo.capturedColor = pieceColor(capturedPiece);
      undo.capturedType = pieceTypeFromId(capturedPiece);
      removePiece(undo.capturedColor, undo.capturedType, to);
      capturedCount[undo.capturedColor][undo.capturedType]++;
      halfmoveClock = 0;
    } else {
      undo.capturedType = NONE;
      undo.capturedColor = WHITE; // dummy
      if (movingType == PAWN)
        halfmoveClock = 0;
      else
        halfmoveClock++;
    }

    // Move piece
    if (promo != NONE) {
      // Promotion: remove pawn, place promoted piece
      removePiece(us, movingType, from);
      putPiece(us, promo, to);
    } else if (capturedPiece) {
      // Capture: destination already cleared, remove+put
      removePiece(us, movingType, from);
      putPiece(us, movingType, to);
    } else {
      // Quiet move: use optimized movePiece
      movePiece(us, movingType, from, to);
    }

    // Flip side
    side = them;
    hash ^= Zobrist::sideKey;

    if (us == BLACK)
      fullmoveNumber++;

    undoStack[undoCount++] = undo;
  }

  void unmakeMove() {
    UndoInfo &undo = undoStack[--undoCount];
    Move m = undo.move;

    int from = moveFrom(m);
    int to = moveTo(m);
    PieceType promo = movePromo(m);

    // Flip side back
    side = ~side;
    Color us = side;

    if (us == BLACK)
      fullmoveNumber--;

    int movingPiece = mailbox[to];
    PieceType currentType = pieceTypeFromId(movingPiece);
    PieceType originalType = (promo != NONE) ? PAWN : currentType;

    // Undo the move
    if (promo != NONE) {
      // Promotion: remove promoted piece, put pawn back
      removePiece(us, currentType, to);
      putPiece(us, originalType, from);
    } else if (undo.captured) {
      // Capture (non-promotion): remove piece from dest, put back at source
      removePiece(us, currentType, to);
      putPiece(us, originalType, from);
    } else {
      // Quiet move: use optimized movePiece (reverse direction)
      movePiece(us, currentType, to, from);
    }

    // Restore captured piece (covers both capture and promotion-capture)
    if (undo.captured) {
      putPiece(undo.capturedColor, undo.capturedType, to);
      capturedCount[undo.capturedColor][undo.capturedType]--;
    }

    hash = undo.hash;
    halfmoveClock = undo.halfmoveClock;
  }

  // Check for game-ending conditions
  bool isCheckmate() {
    if (!inCheck())
      return false;
    Move moves[MAX_MOVES];
    return generateMoves(moves) == 0;
  }

  bool isStalemate() {
    if (inCheck())
      return false;
    Move moves[MAX_MOVES];
    return generateMoves(moves) == 0;
  }

  // Simple material-only evaluation (from white's perspective)
  int materialScore() const {
    constexpr int VALUES[] = {0, 100, 320, 330, 900, 20000};
    int score = 0;
    for (int pt = PAWN; pt <= KING; pt++) {
      score += popcount(pieces[WHITE][pt]) * VALUES[pt];
      score -= popcount(pieces[BLACK][pt]) * VALUES[pt];
    }
    return score;
  }

  // Print board for debugging
  void print() const;
};
