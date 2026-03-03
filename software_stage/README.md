# ♟️ RoboGambit — Software Track
### Simulation-Based Autonomous Robotics Competition  
ARIES × Robotics Club IITD | 2025–26

This repository contains the software tasks for RoboGambit, focused on building a complete autonomous pipeline spanning:

- 🧠 AI-driven Game Decision Making  
- 👁️ ArUco-Based Perception  
- 🧩 State Estimation  
- 🔁 Reproducible Simulation Execution  

The software track is evaluated independently and forms the qualification stage for hardware execution.

---

# 🏗 Competition Structure

The software competition consists of **two tasks**:

| Task | Component | Marks |
|------|----------|-------|
| Task 1 | Autonomous Game Engine | 75 |
| Task 2 | ArUco-Based Perception & State Estimation | 25 |
| **Total** |  | **100** |

---

# 🧠 Task 1 — Autonomous Game Engine (75 Marks)

## Objective

Design a deterministic game engine capable of playing the RoboGambit 6×6 chess-inspired game.

This task focuses **only on decision-making logic** (no perception or motion execution).

---

## Input Format

- A `6×6 NumPy array`
- Values: `0–10`
  - `0` → empty cell  
  - `1–5` → white pieces  
  - `6–10` → black pieces  
- First element = **A1**
- Last element = **F6**

---

## Piece IDs

| Piece | ID |
|--------|----|
| White Pawn | 1 |
| White Knight | 2 |
| White Bishop | 3 |
| White Queen | 4 |
| White King | 5 |
| Black Pawn | 6 |
| Black Knight | 7 |
| Black Bishop | 8 |
| Black Queen | 9 |
| Black King | 10 |

---

## Required Output Format
<piece_id>:<source_cell>-><target_cell>


- Must strictly follow format
- Invalid or ambiguous strings are treated as illegal moves
- Engine must be callable as a standalone module

---

## Evaluation

- Engines compete in automated matches
- Ranked using **ELO-based system**
- Scores normalized relative to highest ELO

---

## Code Freeze Policy

After software evaluation:
- Game engine logic is **frozen**
- No changes to heuristics, models, or strategy allowed
- Only I/O adaptations permitted for hardware stage

Violation → Disqualification

---

# 👁️ Task 2 — ArUco-Based Perception (25 Marks)

## Objective

Develop a robust perception pipeline in ROS 2 + Gazebo that reconstructs the full game state using ArUco markers.

---

## Requirements

Pipeline must:

- Detect arena and piece ArUco markers
- Estimate arena pose (using 4 corner markers)
- Estimate pose of all pieces
- Reconstruct board state
- Output a 6×6 NumPy array (same format as Task 1)

No hardcoding, manual input, or bypassing perception allowed.

---

## Deliverables

- Perception source code
- 6×6 NumPy board output
- Text file with board position
- Demo video showing:
  - Marker detection
  - Pose estimation
  - State reconstruction
  - Brief explanation of approach

---

# 🖥 Environment Requirements

All submissions must:

- Run on **Ubuntu 22.04**
- Be compatible with organiser-provided systems
- Include all dependencies and setup instructions
- Include trained models (if used)
- Be reproducible without modification

---

# 📅 Timeline

- **4 March 2026** — Task 1 Release  
- **7 March 2026** — Task 2 Release  
- **12 March 2026** — Submission Deadline  
- **15 March 2026** — Final Deadline (20% penalty)

No submissions accepted after 15 March.

---

# 🚫 Restrictions

- No manual intervention during autonomous execution
- No hardcoding results
- No bypassing perception
- No external API calls to LLMs or pre-existing hosted models
- Code must be original work

---

# 🎯 What We Evaluate

- Correctness of implementation  
- Robustness and reproducibility  
- Strategic reasoning  
- Accuracy of perception  
- Clarity of design  
- System integration quality  

---

For complete rules and detailed specifications, refer to the official Software Rulebook included in this repository.