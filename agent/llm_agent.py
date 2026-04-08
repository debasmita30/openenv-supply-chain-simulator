

import os
from dotenv import load_dotenv
from openai import OpenAI

# ===== LOAD ENV =====
load_dotenv()

API_BASE = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("HF_TOKEN", "")
MODEL = os.getenv("MODEL_NAME", "gpt-4o-mini")

# ===== ALLOWED ACTIONS =====
ALLOWED_ACTIONS = [
    "reroute",
    "change_supplier",
    "increase_inventory",
    "delay_orders"
]

# ===== INIT CLIENT =====
client = None
if API_KEY:
    client = OpenAI(base_url=API_BASE, api_key=API_KEY)


# ===== FALLBACK POLICY (SMART + SELF-CORRECTING) =====
def fallback_policy(state, memory):
    inv = state["inventory"]
    demand = state["demand_signal"]
    delay = state["supplier_delay"]
    backlog = state["backlog"]
    cost = state["transport_cost"]

    recent_rewards = [m["reward"] for m in memory] if memory else []
    recent_actions = [m["action"]["type"] for m in memory] if memory else []

    avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0

    # --- break loops ---
    if len(recent_actions) >= 3 and len(set(recent_actions[-3:])) == 1:
        last = recent_actions[-1]
        if last == "increase_inventory":
            return {"type": "reroute"}
        if last == "delay_orders":
            return {"type": "increase_inventory"}
        return {"type": "delay_orders"}

    # --- failure recovery ---
    if avg_reward < 0.1:
        if cost > 1.8:
            return {"type": "delay_orders"}
        if backlog > 40:
            return {"type": "change_supplier"}
        if delay > 2:
            return {"type": "reroute"}
        return {"type": "increase_inventory"}

    # --- main strategy ---
    if inv < demand * 0.6:
        if cost < 1.5:
            return {"type": "increase_inventory"}
        else:
            return {"type": "reroute"}

    if delay > 3:
        return {"type": "reroute"}

    if backlog > 60:
        return {"type": "change_supplier"}
    
    if cost > 2.0:
        if delay > 1:
            return {"type": "reroute"}
        return {"type": "delay_orders"}

    if cost > 2.2:
        return {"type": "delay_orders"}

    # --- controlled variation ---
    if len(memory) % 2 == 0:
        return {"type": "delay_orders"}
    else:
        return {"type": "reroute"}


# ===== LLM DECISION =====
def llm_decide(state, memory):
    SYSTEM_PROMPT = """
You are an expert supply chain optimizer.

Goals:
- maximize fulfillment
- minimize backlog
- minimize cost

Return ONLY one word:
reroute, change_supplier, increase_inventory, delay_orders
"""

    prompt = f"""
State:
{state}

Recent History:
{memory}

Best action?
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )

    action = response.choices[0].message.content.strip().lower()
    return action


# ===== SAFE DECIDE (FINAL) =====
def decide(state, memory):
    try:
        # no API → fallback
        if not client:
            return fallback_policy(state, memory)

        action = llm_decide(state, memory)

        # validate
        if action not in ALLOWED_ACTIONS:
            return fallback_policy(state, memory)

        return {"type": action}

    except Exception:
        # ANY failure → fallback
        return fallback_policy(state, memory)