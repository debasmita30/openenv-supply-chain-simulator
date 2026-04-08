import random
from models import StepResult

class SupplyChainEnv:
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.max_steps = 12

    def reset(self, config=None):
     
        self.true_demand = self.rng.randint(80, 120)

        if config:
            self.state = {
                "inventory": config["inventory"],
                "supplier_delay": config["supplier_delay"],
                "transport_cost": config["transport_cost"],
                "backlog": config["backlog"],
                "step": 0
            }
        else:
            self.state = {
                "inventory": 100,
                "supplier_delay": 2,
                "transport_cost": 1.0,
                "backlog": 0,
                "step": 0
            }

        self.pending_penalty = 0
        return self._observe()

    def _observe(self):
        obs = self.state.copy()
        obs["demand_signal"] = self.true_demand + self.rng.randint(-10, 10)
        return obs

    def step(self, action):
        s = self.state
        a = action["type"]

        # --- delayed penalty ---
        delayed_penalty = self.pending_penalty
        self.pending_penalty = 0

        # --- actions ---
        if a == "reroute":
            s["supplier_delay"] = max(0, s["supplier_delay"] - 1)
            s["transport_cost"] += 0.2

        elif a == "change_supplier":
            s["supplier_delay"] = self.rng.randint(1, 4)
            self.pending_penalty = 0.1

        elif a == "increase_inventory":
            s["inventory"] += 20
            s["transport_cost"] += 0.1

        elif a == "delay_orders":
            s["backlog"] += 5

        # --- stochastic disruption ---
        if self.rng.random() < 0.25:
            s["supplier_delay"] += 1

        # --- demand evolution (CONTROLLED) ---
        self.true_demand += self.rng.randint(-5, 10)
        self.true_demand = max(40, min(150, self.true_demand))

        # --- fulfillment (CONTROLLED DRAIN) ---
        fulfilled = min(s["inventory"], int(self.true_demand * 0.6))
        s["inventory"] -= fulfilled

        unmet = self.true_demand - fulfilled
        s["backlog"] = min(200, s["backlog"] + unmet)

        # --- cap cost ---
        s["transport_cost"] = min(2.5, s["transport_cost"])

        # --- reward shaping (FINAL FIX) ---
        fulfillment_ratio = fulfilled / (self.true_demand + 1)

        cost_norm = min(1.0, s["transport_cost"] / 3.0)
        backlog_norm = min(1.0, s["backlog"] / 150.0)

        reward = (
            0.7 * fulfillment_ratio +
            0.15 * (1 - cost_norm) +
            0.15 * (1 - backlog_norm)
        )

        reward = max(0.1, reward)

        # small penalties
        reward -= 0.005
        reward -= delayed_penalty * 0.5

        reward = min(1.0, reward)

        # --- step ---
        s["step"] += 1
        done = s["step"] >= self.max_steps

        return StepResult(
            state=self._observe(),
            reward=reward,
            done=done,
            info={"true_demand": self.true_demand}
        )

    def state(self):
        return self._observe()