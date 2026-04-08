class TrajectoryMemory:
    def __init__(self):
        self.history = []

    def add(self, state, action, reward):
        self.history.append({
            "state": state,
            "action": action,
            "reward": reward
        })

    def get_recent(self, k=3):
        return self.history[-k:]