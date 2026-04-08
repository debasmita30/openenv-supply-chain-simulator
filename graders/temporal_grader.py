def grade(trajectory):
    rewards = [step["reward"] for step in trajectory]

    # trend (improvement over time)
    trend = 0
    for i in range(1, len(rewards)):
        if rewards[i] > rewards[i-1]:
            trend += 1

    trend_score = trend / len(rewards)

    # stability (low variance)
    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    stability_score = max(0, 1 - variance)

    # final performance
    final_score = rewards[-1]

    score = (
        0.4 * final_score +
        0.3 * trend_score +
        0.3 * stability_score
    )

    return max(0, min(1, score))