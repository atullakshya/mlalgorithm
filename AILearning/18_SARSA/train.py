"""
================================================================================
SARSA - Real-World Example: Delivery Drone in Windy City
================================================================================
REAL-TIME USE CASE:
    A delivery drone must fly from a warehouse to a delivery point in a city
    with varying wind conditions. Wind pushes the drone upward in certain columns.
    SARSA learns a SAFE policy that accounts for the wind (more conservative than Q-Learning).

ALGORITHM:
    SARSA (State-Action-Reward-State-Action) - ON-POLICY RL:
    - Similar to Q-Learning, but updates using the ACTUAL next action taken
    - Q(s,a) += alpha * (reward + gamma * Q(s', a') - Q(s,a))
    - Key difference: uses Q(s', a') (actual) vs Q-Learning's max Q(s', a')
    - Results in a MORE CONSERVATIVE policy (avoids risky states)

MODEL TYPE AFTER TRAINING:
    -> Same as Q-Learning: a Q-TABLE with Q(state, action) values.
    -> But the learned policy is MORE CAUTIOUS because SARSA accounts for
       the exploration it does during training (on-policy).
    -> Q-Learning learns optimal policy regardless of exploration (off-policy).
================================================================================
"""
import numpy as np
import os

GRID_ROWS = 7
GRID_COLS = 10
START = (3, 0)   # Warehouse
GOAL = (3, 7)    # Delivery point
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]  # Upward wind per column

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["Up", "Down", "Left", "Right"]
ACTION_SYMBOLS = ["^", "v", "<", ">"]

def step(state, action_idx):
    """Move drone: wind pushes upward (negative row direction)."""
    dr, dc = ACTIONS[action_idx]
    new_r = state[0] + dr - WIND[state[1]]  # Wind effect
    new_c = state[1] + dc
    new_r = max(0, min(GRID_ROWS - 1, new_r))
    new_c = max(0, min(GRID_COLS - 1, new_c))
    new_state = (new_r, new_c)
    done = new_state == GOAL
    reward = 0.0 if done else -1.0  # -1 per step encourages shortest path
    return new_state, reward, done

def choose_action(Q, state, epsilon):
    """Epsilon-greedy action selection."""
    if np.random.random() < epsilon:
        return np.random.randint(len(ACTIONS))
    return np.argmax(Q[state[0], state[1]])

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # -------------------------------------------------------------------------
    # STEP 1: Environment setup
    # -------------------------------------------------------------------------
    print("=== STEP 1: Windy City Grid Environment ===")
    print(f"Grid: {GRID_ROWS}x{GRID_COLS}")
    print(f"Start (Warehouse) : {START}")
    print(f"Goal (Delivery)   : {GOAL}")
    print(f"Wind strength     : {WIND}")
    print(f"  Columns 3-5: wind=1 (mild), Columns 6-7: wind=2 (strong)")
    print("\nGrid (numbers = wind strength per column):")
    print("  Wind: " + " ".join(f"{w}" for w in WIND))
    for r in range(GRID_ROWS):
        row_str = "  "
        for c in range(GRID_COLS):
            if (r, c) == START:
                row_str += " S"
            elif (r, c) == GOAL:
                row_str += " G"
            else:
                row_str += " ."
        print(row_str)

    # -------------------------------------------------------------------------
    # STEP 2: Initialize Q-Table
    # -------------------------------------------------------------------------
    Q = np.zeros((GRID_ROWS, GRID_COLS, len(ACTIONS)))
    print(f"\n=== STEP 2: Q-Table Initialized ===")
    print(f"Shape: {Q.shape}")

    # -------------------------------------------------------------------------
    # STEP 3: Hyperparameters
    # -------------------------------------------------------------------------
    alpha = 0.5     # Higher learning rate than Q-Learning (SARSA is more stable)
    gamma = 1.0     # No discounting (all future rewards equally important)
    epsilon = 0.1   # Constant exploration (on-policy requires some exploration)
    episodes = 5000

    print(f"\n=== STEP 3: Hyperparameters ===")
    print(f"Alpha (learning rate): {alpha}")
    print(f"Gamma (discount)     : {gamma}")
    print(f"Epsilon (exploration): {epsilon}")
    print(f"Episodes             : {episodes}")

    # -------------------------------------------------------------------------
    # STEP 4: Train using SARSA
    # KEY DIFFERENCE from Q-Learning:
    #   Q-Learning: Q(s,a) += alpha * (r + gamma * MAX Q(s',a') - Q(s,a))
    #   SARSA:      Q(s,a) += alpha * (r + gamma * Q(s', a') - Q(s,a))
    #   SARSA uses the ACTUAL next action a' (not the best), making it on-policy
    # -------------------------------------------------------------------------
    steps_per_episode = []

    for ep in range(episodes):
        state = START
        action = choose_action(Q, state, epsilon)  # Choose first action
        total_steps = 0

        for _ in range(1000):
            next_state, reward, done = step(state, action)
            next_action = choose_action(Q, next_state, epsilon)  # Choose ACTUAL next action

            # SARSA update: uses Q(s', a') where a' is the ACTUAL next action
            Q[state[0], state[1], action] += alpha * (
                reward + gamma * Q[next_state[0], next_state[1], next_action]
                - Q[state[0], state[1], action]
            )

            state = next_state
            action = next_action  # This is what makes SARSA "on-policy"
            total_steps += 1
            if done:
                break

        steps_per_episode.append(total_steps)

    print(f"\n=== STEP 4: Training Complete ===")
    for milestone in [100, 500, 1000, 2500, 5000]:
        if milestone <= episodes:
            idx = milestone - 1
            avg = np.mean(steps_per_episode[max(0,idx-99):idx+1])
            print(f"  Episode {milestone:5d}: Avg Steps to Goal = {avg:>5.1f}")

    # -------------------------------------------------------------------------
    # STEP 5: Show learned policy
    # -------------------------------------------------------------------------
    print(f"\n=== STEP 5: Learned Policy ===")
    print("  Wind: " + " ".join(f"{w}" for w in WIND))
    for r in range(GRID_ROWS):
        row_str = "  "
        for c in range(GRID_COLS):
            if (r, c) == GOAL:
                row_str += " G"
            elif (r, c) == START:
                row_str += " S"
            else:
                best_a = np.argmax(Q[r, c])
                row_str += f" {ACTION_SYMBOLS[best_a]}"
        print(row_str)

    # -------------------------------------------------------------------------
    # STEP 6: Demonstrate optimal path (greedy policy, no exploration)
    # -------------------------------------------------------------------------
    state = START
    path = [state]
    for _ in range(100):
        action = np.argmax(Q[state[0], state[1]])
        state, _, done = step(state, action)
        path.append(state)
        if done:
            break

    print(f"\n=== STEP 6: Optimal Delivery Route ===")
    print(f"Steps: {len(path)-1}")
    print(f"Path: {' -> '.join(str(p) for p in path[:20])}")

    # -------------------------------------------------------------------------
    # STEP 7: SARSA vs Q-Learning comparison explanation
    # -------------------------------------------------------------------------
    print(f"\n=== STEP 7: SARSA vs Q-Learning ===")
    print(f"  Q-Learning (off-policy):")
    print(f"    Update: Q(s,a) += alpha * (r + gamma * MAX Q(s',a') - Q(s,a))")
    print(f"    Learns: The BEST possible policy (ignores exploration mistakes)")
    print(f"    Risk  : May learn paths near dangerous states")
    print(f"  SARSA (on-policy):")
    print(f"    Update: Q(s,a) += alpha * (r + gamma * Q(s',a') - Q(s,a))")
    print(f"    Learns: The policy it ACTUALLY follows (including exploration)")
    print(f"    Safer : Avoids risky states because it accounts for random actions")

    # -------------------------------------------------------------------------
    # STEP 8: Save model
    # -------------------------------------------------------------------------
    model_path = os.path.join(script_dir, "sarsa_q_table.npy")
    np.save(model_path, Q)

    print(f"\n=== STEP 8: Model Saved ===")
    print(f"\n--- WHAT IS THE TRAINED MODEL? ---")
    print(f"Type     : SARSA Q-Table (On-Policy RL)")
    print(f"Contents : {Q.shape} Q-table = {Q.size} state-action values")
    print(f"Policy   : Pick action with highest Q(s,a) in each state")
    print(f"Key diff : More CONSERVATIVE than Q-Learning")
    print(f"           Learns from what it ACTUALLY does (not what's theoretically best)")
    print(f"Best for : Safety-critical applications (drones, robotics, medical)")

if __name__ == "__main__":
    main()
