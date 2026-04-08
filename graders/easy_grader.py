def grade(trajectory):
    final = trajectory[-1]["state"]

    backlog_score = max(0, 1 - final["backlog"] / 100)
    cost_score = max(0, 1 - final["transport_cost"] / 2)

    score = 0.6 * backlog_score + 0.4 * cost_score
    return min(1.0, max(0.0, score))