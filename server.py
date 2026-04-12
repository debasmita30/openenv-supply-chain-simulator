from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from environment import SupplyChainEnv

app = FastAPI()
env = SupplyChainEnv()

class ResetConfig(BaseModel):
    inventory: Optional[int] = None
    supplier_delay: Optional[int] = None
    transport_cost: Optional[float] = None
    backlog: Optional[int] = None
    task: Optional[str] = None

class ActionInput(BaseModel):
    type: str

@app.get("/reset")
def reset_get():
    return env.reset()

@app.post("/reset")
def reset_post(config: ResetConfig):
    return env.reset(config.dict(exclude_none=True))

@app.post("/step")
def step(action: ActionInput):
    return env.step(action.dict()).dict()

@app.get("/state")
def get_state():
    return env._observe()

@app.get("/health")
def health():
    return {"status": "ok"}
