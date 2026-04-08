import os
import requests
from agent.llm_agent import decide
from memory.trajectory import TrajectoryMemory

from tasks.easy import get_easy_config
from tasks.medium import get_medium_config
from tasks.hard import get_hard_config

BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:7860")

EPISODES = 3
MAX_STEPS = 12


def run_episode(config, ep_num):
    memory = TrajectoryMemory()
    state = requests.post(f"{BASE}/reset", json=config).json()


    total_reward = 0

    for step in range(MAX_STEPS):
        action = decide(state, memory.get_recent())

        result = requests.post(f"{BASE}/step", json=action).json()

        print(f"[STEP] EP={ep_num} STEP={step} ACTION={action} REWARD={result['reward']}")

        memory.add(state, action, result["reward"])
        total_reward += result["reward"]
        state = result["state"]

        if result["done"]:
            break

    return total_reward


def evaluate_task(name, config_fn):
    print(f"\n[TASK] {name.upper()}")

    scores = []
    config = config_fn()

    for ep in range(EPISODES):
        score = run_episode(config, ep)
        scores.append(score)

    avg = sum(scores) / len(scores)
    best = max(scores)
    worst = min(scores)

    print(f"[RESULT] {name} AVG={avg:.2f} BEST={best:.2f} WORST={worst:.2f}")

    return avg


def run():
    print("[START]")

    easy_score = evaluate_task("easy", get_easy_config)
    medium_score = evaluate_task("medium", get_medium_config)
    hard_score = evaluate_task("hard", get_hard_config)

    final_score = (easy_score + medium_score + hard_score) / 3

    print(f"\n[END] FINAL_SCORE={final_score:.2f}")


if __name__ == "__main__":
    run()