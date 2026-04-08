from fastapi import FastAPI
from environment import SupplyChainEnv

app = FastAPI()
env = SupplyChainEnv()

@app.get("/reset")
def reset():
    return env.reset()

@app.post("/reset")
def reset_with_config(config: dict):
    return env.reset(config)

@app.post("/step")
def step(action: dict):
    return env.step(action).dict()

@app.get("/state")
def state():
    return env.state()