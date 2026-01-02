# Backend/server.py

import os
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from Backend.dqn_educational import AfterstateAgent
from Backend.game_2048_env import Game2048Env

# ----------------- Paths -----------------

BACKEND_DIR = Path(__file__).resolve().parent      # /.../2048_game/Backend
ROOT_DIR = BACKEND_DIR.parent                      # /.../2048_game

# model you deploy with Heroku
MODEL_PATH = ROOT_DIR / "models" / "afterstate_2048_parallel_latest.pt"

# Vite default build output: Frontend/dist
FRONTEND_BUILD_DIR = ROOT_DIR / "Frontend" / "dist"

# ----------------- FastAPI app -----------------

app = FastAPI()

# ----------------- Static / Frontend -----------------

if FRONTEND_BUILD_DIR.is_dir():
    # Vite puts JS/CSS/etc in dist/assets
    assets_dir = FRONTEND_BUILD_DIR / "assets"
    if assets_dir.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=assets_dir),
            name="assets",
        )

    @app.get("/", include_in_schema=False)
    async def serve_index():
        """Serve the built React (Vite) app."""
        return FileResponse(FRONTEND_BUILD_DIR / "index.html")
else:
    @app.get("/", include_in_schema=False)
    async def root():
        """Fallback when frontend bundle is not present."""
        return {"status": "ok", "message": "DQN-2048 backend is running (no frontend bundle found)"}


# Optional explicit health endpoint for checks
@app.get("/api/health")
def health():
    return {"status": "ok"}


# ----------------- CORS -----------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5175",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        # you can add your Heroku frontend origin if you ever separate them
        # "https://your-frontend-domain.com",
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

agent: AfterstateAgent | None = None


@app.on_event("startup")
def load_model():
    global agent
    agent = AfterstateAgent(
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=0.0,  # no exploration at inference
        epsilon_end=0.0,
        epsilon_decay=1.0,
        batch_size=128,
        target_update_freq=500,
    )
    print(f"Loading model from: {MODEL_PATH}")
    agent.load(str(MODEL_PATH))
    agent.epsilon = 0.0
    agent.policy_net.eval()
    print(f"📂 Model loaded from {MODEL_PATH}")
    print("✅ Afterstate agent loaded and ready!")


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

    env = Game2048Env()
    env.board = board_np.copy()

    valid_actions = env.get_valid_actions()
    if not valid_actions:
        return MoveResponse(action=-1)

    gamma = agent.gamma
    best_action = None
    best_value = -float("inf")

    # Greedy afterstate search: a* = argmax_a [ merge_reward + γ V(afterstate) ]
    for action in valid_actions:
        moved, merge_reward, after_planes = env.get_afterstate(action)
        if not moved:
            continue

        after_tensor = torch.FloatTensor(after_planes).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            v_next = agent.policy_net(after_tensor).item()

        q_value = merge_reward + gamma * v_next

        if q_value > best_value:
            best_value = q_value
            best_action = action

    if best_action is None:
        # Fallback (shouldn't happen if valid_actions is correct)
        best_action = random.choice(valid_actions)

    return MoveResponse(action=int(best_action))
