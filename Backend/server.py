# server.py
import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from .dqn_educational import DQNAgent, board_to_planes  # reuse your code
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from .game_2048_env import Game2048Env
from pathlib import Path

FRONTEND_BUILD_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "build")

if os.path.isdir(FRONTEND_BUILD_DIR):
    app.mount(
        "/static",
        StaticFiles(directory=os.path.join(FRONTEND_BUILD_DIR, "static")),
        name="static",
    )

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(FRONTEND_BUILD_DIR, "index.html"))

app = FastAPI()
BACKEND_DIR = Path(__file__).resolve().parent   # /.../2048_game/Backend
ROOT_DIR = BACKEND_DIR.parent                   # /.../2048_game

MODEL_PATH = ROOT_DIR / "models" / "dqn_2048_parallel_ep12500.pt"
# ---- CORS ----
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
    allow_methods=["*"],  # important: lets OPTIONS through
    allow_headers=["*"],
)


# ----- Request / Response schemas -----

class BoardRequest(BaseModel):
    board: List[List[int]]  # 4x4 array of tile values


class MoveResponse(BaseModel):
    action: int  # 0=up, 1=down, 2=left, 3=right


# ----- Load model at startup -----

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


# ----- Helper: compute valid actions on backend (optional but nice) -----

def get_valid_actions_from_board(board_np: np.ndarray) -> list[int]:
    """
    Determine which moves actually change the board.
    Uses same logic as Game2048Env.move(), but doesn't modify score.
    """
    

    env = Game2048Env()
    env.board = board_np.copy()
    valid = env.get_valid_actions()
    return valid


# ----- Main endpoint -----

@app.post("/move", response_model=MoveResponse)
def get_ai_move(req: BoardRequest):
    """
    Given a 4x4 2048 board, return the best move according to the DQN.
    """

    # Convert incoming Python list -> numpy array
    board_np = np.array(req.board, dtype=np.int32)

    # Compute valid moves so the model never picks invalid ones
    valid_actions = get_valid_actions_from_board(board_np)
    if not valid_actions:
        # No moves available; return -1 to indicate "game over" / "no move"
        return MoveResponse(action=-1)

    # Convert to planes: (16, 4, 4)
    planes = board_to_planes(board_np)  # np.float32

    # Use your existing select_action() interface
    action = agent.select_action(
        planes,
        valid_actions=valid_actions,
        training=False,
    )

    return MoveResponse(action=int(action))
