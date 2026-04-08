def grade(trajectory):
    final = trajectory[-1]["state"]

    backlog_score = max(0, 1 - final["backlog"] / 120)
    cost_score = max(0, 1 - final["transport_cost"] / 2.5)

    efficiency = len(trajectory)
    efficiency_score = max(0, 1 - efficiency / 12)

    score = (
        0.4 * backlog_score +
        0.4 * cost_score +
        0.2 * efficiency_score
    )

    return min(1.0, max(0.0, score))