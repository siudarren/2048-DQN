import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .dqn_educational import DQNAgent, board_to_planes
from .game_2048_env import Game2048Env

# ----------------- Paths -----------------

BACKEND_DIR = Path(__file__).resolve().parent   # /.../2048_game/Backend
ROOT_DIR = BACKEND_DIR.parent                   # /.../2048_game
MODEL_PATH = ROOT_DIR / "models" / "dqn_2048_parallel_ep32000.pt"
FRONTEND_BUILD_DIR = ROOT_DIR / "frontend" / "build"

# ----------------- FastAPI app -----------------

app = FastAPI()

# Static files / React build (optional)
if FRONTEND_BUILD_DIR.is_dir():
    app.mount(
        "/static",
        StaticFiles(directory=FRONTEND_BUILD_DIR / "static"),
        name="static",
    )

    @app.get("/")
    async def serve_index():
        return FileResponse(FRONTEND_BUILD_DIR / "index.html")
else:
    @app.get("/")
    async def root():
        return {"status": "ok", "message": "DQN-2048 backend is running"}

# ----------------- CORS -----------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:5175",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Schemas -----------------

class BoardRequest(BaseModel):
    board: List[List[int]]  # 4x4 array of tile values


class MoveResponse(BaseModel):
    action: int  # 0=up, 1=down, 2=left, 3=right


# ----------------- Agent load -----------------

agent = None


@app.on_event("startup")
def load_model():
    global agent
    agent = DQNAgent()
    print(f"Loading model from: {MODEL_PATH}")
    agent.load(str(MODEL_PATH))   # ensure string path
    agent.epsilon = 0.0
    agent.policy_net.eval()
    print("✅ DQN agent loaded and ready!")


# ----------------- Helpers -----------------

def get_valid_actions_from_board(board_np: np.ndarray) -> list[int]:
    env = Game2048Env()
    env.board = board_np.copy()
    valid = env.get_valid_actions()
    return valid


# ----------------- Main endpoint -----------------

@app.post("/move", response_model=MoveResponse)
def get_ai_move(req: BoardRequest):
    board_np = np.array(req.board, dtype=np.int32)

    valid_actions = get_valid_actions_from_board(board_np)
    if not valid_actions:
        return MoveResponse(action=-1)

    planes = board_to_planes(board_np)

    action = agent.select_action(
        planes,
        valid_actions=valid_actions,
        training=False,
    )

    return MoveResponse(action=int(action))
