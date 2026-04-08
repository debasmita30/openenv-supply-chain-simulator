from pydantic import BaseModel

class State(BaseModel):
    inventory: int
    supplier_delay: int
    transport_cost: float
    backlog: int
    step: int

class StepResult(BaseModel):
    state: dict
    reward: float
    done: bool
    info: dict