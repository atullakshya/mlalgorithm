"""
================================================================================
Q-LEARNING - Real-World Example: Warehouse Robot Navigation
================================================================================
REAL-TIME USE CASE:
    A warehouse robot must learn to navigate from the loading dock (start) to the
    shipping area (goal) while avoiding obstacles (shelves, walls). The robot
    gets rewards for reaching the goal and penalties for hitting obstacles.

ALGORITHM:
    Q-Learning (model-free, off-policy Reinforcement Learning):
    - Agent learns Q-values: Q(state, action) = expected future reward
    - Update rule: Q(s,a) = Q(s,a) + alpha * (reward + gamma*max(Q(s',a')) - Q(s,a))
    - Uses epsilon-greedy: explore randomly (epsilon) or exploit best known action
    - Does NOT need a model of the environment

MODEL TYPE AFTER TRAINING:
    -> A Q-TABLE: a 2D matrix of size (states x actions).
    -> Each cell = expected cumulative reward for taking action A in state S.
    -> The optimal POLICY is: in each state, pick the action with highest Q-value.
    -> Saved as .npy, contains the complete Q-table.
    -> VALUE-BASED RL model (learns values, derives policy from values).
================================================================================
"""
import numpy as np
import json
import os

# -------------------------------------------------------------------------
# STEP 1: Define the warehouse grid environment
# S = Start (loading dock), G = Goal (shipping area), X = Obstacle, . = Free
# -------------------------------------------------------------------------
GRID = [
    [".", ".", ".", "X", ".", "."],
    [".", "X", ".", ".", ".", "."],
    [".", ".", ".", "X", ".", "."],
    [".", "X", ".", ".", "X", "."],
    [".", ".", ".", ".", ".", "G"],
]
GRID_ROWS = len(GRID)
GRID_COLS = len(GRID[0])
START = (0, 0)
GOAL = (4, 5)
OBSTACLES = set()
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        if GRID[r][c] == "X":
            OBSTACLES.add((r, c))

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
ACTION_NAMES = ["Up", "Down", "Left", "Right"]
ACTION_SYMBOLS = ["^", "v", "<", ">"]

def is_valid(state):
    r, c = state
    return 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS and state not in OBSTACLES

def step(state, action_idx):
    """Environment step: returns (next_state, reward, done)."""
    dr, dc = ACTIONS[action_idx]
    new_state = (state[0] + dr, state[1] + dc)
    if not is_valid(new_state):
        return state, -1.0, False  # Hit wall/obstacle -> penalty, stay
    if new_state == GOAL:
        return new_state, 10.0, True  # Reached goal -> big reward
    return new_state, -0.1, False  # Normal move -> small penalty (encourages efficiency)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("=== STEP 1: Warehouse Grid Environment ===")
    print(f"Grid size: {GRID_ROWS}x{GRID_COLS}")
    print(f"Start: {START} (Loading Dock)")
    print(f"Goal : {GOAL} (Shipping Area)")
    print(f"Obstacles: {OBSTACLES}")
    print("\nGrid layout (S=Start, G=Goal, X=Obstacle):")
    for r in range(GRID_ROWS):
        row_str = "  "
        for c in range(GRID_COLS):
            if (r, c) == START:
                row_str += " S "
            elif (r, c) == GOAL:
                row_str += " G "
            elif (r, c) in OBSTACLES:
                row_str += " X "
            else:
                row_str += " . "
        print(row_str)

    # -------------------------------------------------------------------------
    # STEP 2: Initialize Q-Table (all zeros)
    # Shape: (rows, cols, 4 actions) - stores expected reward for each state-action pair
    # -------------------------------------------------------------------------
    Q = np.zeros((GRID_ROWS, GRID_COLS, len(ACTIONS)))
    print(f"\n=== STEP 2: Q-Table Initialized ===")
    print(f"Shape: {Q.shape} ({GRID_ROWS}*{GRID_COLS} states x {len(ACTIONS)} actions)")
    print(f"Total Q-values: {Q.size} (all initialized to 0)")

    # -------------------------------------------------------------------------
    # STEP 3: Set hyperparameters
    # -------------------------------------------------------------------------
    alpha = 0.1        # Learning rate (how fast Q-values update)
    gamma = 0.99       # Discount factor (importance of future rewards)
    epsilon = 1.0      # Exploration rate (start fully random)
    epsilon_decay = 0.995  # Reduce exploration over time
    epsilon_min = 0.01
    episodes = 5000
    max_steps = 200

    print(f"\n=== STEP 3: Hyperparameters ===")
    print(f"Learning rate (alpha)   : {alpha}")
    print(f"Discount factor (gamma) : {gamma}")
    print(f"Initial epsilon         : {epsilon}")
    print(f"Epsilon decay           : {epsilon_decay}")
    print(f"Episodes                : {episodes}")

    # -------------------------------------------------------------------------
    # STEP 4: Train the agent using Q-Learning
    # The agent explores the environment and updates Q-values after each step
    # Q(s,a) += alpha * (reward + gamma * max(Q(s')) - Q(s,a))
    # -------------------------------------------------------------------------
    rewards_per_episode = []
    steps_per_episode = []

    for ep in range(episodes):
        state = START
        total_reward = 0
        total_steps = 0

        for _ in range(max_steps):
            # Epsilon-greedy: explore randomly OR exploit best known action
            if np.random.random() < epsilon:
                action = np.random.randint(len(ACTIONS))  # Explore
            else:
                action = np.argmax(Q[state[0], state[1]])  # Exploit

            next_state, reward, done = step(state, action)

            # Q-Learning update (the core formula)
            best_next = np.max(Q[next_state[0], next_state[1]])
            Q[state[0], state[1], action] += alpha * (
                reward + gamma * best_next - Q[state[0], state[1], action]
            )

            state = next_state
            total_reward += reward
            total_steps += 1
            if done:
                break

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(total_steps)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print(f"\n=== STEP 4: Training Complete ===")
    for milestone in [100, 500, 1000, 2500, 5000]:
        if milestone <= episodes:
            idx = milestone - 1
            avg_r = np.mean(rewards_per_episode[max(0,idx-99):idx+1])
            avg_s = np.mean(steps_per_episode[max(0,idx-99):idx+1])
            print(f"  Episode {milestone:5d}: Avg Reward={avg_r:>7.2f}, Avg Steps={avg_s:>5.1f}")

    # -------------------------------------------------------------------------
    # STEP 5: Extract the learned policy (best action per state)
    # -------------------------------------------------------------------------
    print(f"\n=== STEP 5: Learned Policy (THE MODEL) ===")
    print("  Robot's optimal action in each cell:")
    for r in range(GRID_ROWS):
        row_str = "  "
        for c in range(GRID_COLS):
            if (r, c) == GOAL:
                row_str += " G "
            elif (r, c) in OBSTACLES:
                row_str += " X "
            else:
                best_a = np.argmax(Q[r, c])
                row_str += f" {ACTION_SYMBOLS[best_a]} "
        print(row_str)

    # -------------------------------------------------------------------------
    # STEP 6: Demonstrate the learned optimal path
    # -------------------------------------------------------------------------
    state = START
    path = [state]
    for _ in range(50):
        action = np.argmax(Q[state[0], state[1]])
        state, _, done = step(state, action)
        path.append(state)
        if done:
            break

    print(f"\n=== STEP 6: Optimal Path ===")
    print(f"Path length: {len(path)-1} steps")
    print(f"Path: {' -> '.join(str(p) for p in path)}")

    # -------------------------------------------------------------------------
    # STEP 7: Show Q-values for a sample state
    # -------------------------------------------------------------------------
    sample = (2, 2)
    print(f"\n=== STEP 7: Q-values at state {sample} ===")
    for i, (name, q_val) in enumerate(zip(ACTION_NAMES, Q[sample[0], sample[1]])):
        best = " <-- BEST" if i == np.argmax(Q[sample[0], sample[1]]) else ""
        print(f"  {name:>5s}: Q = {q_val:>8.4f}{best}")

    # -------------------------------------------------------------------------
    # STEP 8: Save the Q-table (THE MODEL)
    # -------------------------------------------------------------------------
    model_path = os.path.join(script_dir, "q_table.npy")
    np.save(model_path, Q)

    print(f"\n=== STEP 8: Model Saved ===")
    print(f"Saved to: {model_path}")
    print(f"\n--- WHAT IS THE TRAINED MODEL? ---")
    print(f"Type     : Q-Table (Value-Based Reinforcement Learning)")
    print(f"Contents : {Q.shape} matrix = {Q.size} Q-values")
    print(f"           Q(state, action) = expected cumulative future reward")
    print(f"Policy   : In each state, pick action with highest Q-value")
    print(f"Training : Agent explored {episodes} episodes, learning from rewards")
    print(f"Key diff : No labeled data! Agent learned by TRIAL AND ERROR")
    print(f"Size     : {Q.size * 8 / 1024:.1f} KB (one float per state-action pair)")

if __name__ == "__main__":
    main()
