from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.solvers import UniversalPKSolver

app = FastAPI(title="Ray Lee's ellslab PK API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PKRequest(BaseModel):
    model_type: str
    weight: float
    dose: float
    tau: float
    tinf: float = 1.0
    thalf: float = 0.0 # 페니토인은 Ke가 아닌 Vmax/Km 사용[cite: 6]
    num_doses: int = 7

@app.post("/api/pk/simulate")
def simulate_pk(req: PKRequest):
    try:
        solver = UniversalPKSolver(req)
        result = solver.solve()
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))