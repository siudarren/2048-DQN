// src/components/GameContainer.jsx

import React, {useState, useEffect, useRef} from "react";
import Board from "./Board";
import Scoreboard from "./Scoreboard";
import {addRandomNumber, makeMove, checkGameover} from "../Utilities/utils.js";

function GameContainer() {
    const [board, setBoard] = useState([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]);
    const [score, setScore] = useState(0);
    const [gameover, setGameover] = useState(false);
    const [autoPlaying, setAutoPlaying] = useState(false);

    const boardRef = useRef(board);
    useEffect(() => {
        boardRef.current = board;
    }, [board]);

    // ---------- 1. Initial random tiles ----------
    useEffect(() => {
        let newBoard = addRandomNumber(board);
        newBoard = addRandomNumber(newBoard);
        setBoard(newBoard);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []); // run once on mount

    // ---------- 2. Keyboard input for human play ----------
    useEffect(() => {
        if (gameover) return;

        const handleKeyDown = (event) => {
            // Allow other shortcuts (like F5 dev tools etc.) to work
            if (!["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.key)) {
                return;
            }

            event.preventDefault();

            const result = makeMove(event.key, board);
            if (result.newState !== null) {
                const newBoardAddTile = addRandomNumber(result.newState);
                if (newBoardAddTile !== null) {
                    setBoard(newBoardAddTile);
                    setScore((prevScore) => prevScore + result.score);

                    // ✅ Check gameover on the *new* board, not the old one
                    if (checkGameover(newBoardAddTile)) {
                        setGameover(true);
                    }
                }
            }
        };

        window.addEventListener("keydown", handleKeyDown);
        return () => {
            window.removeEventListener("keydown", handleKeyDown);
        };
    }, [board, gameover]);

    // ---------- 3. Call backend to get AI move ----------
    async function requestAiMove(currentBoard) {
        try {
            const API_BASE =
                process.env.NODE_ENV === "production"
                    ? "" // same origin
                    : "http://localhost:8000"; // dev backend

            const res = await fetch(`${API_BASE}/move`, {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({board: currentBoard}),
            });

            if (!res.ok) {
                console.error("AI request failed:", res.status);
                return null;
            }

            const data = await res.json(); // { action: number }
            return data.action; // 0,1,2,3 or -1
        } catch (err) {
            console.error("Failed to call AI:", err);
            return null;
        }
    }

    useEffect(() => {
        if (!autoPlaying || gameover) return;

        let cancelled = false;

        const doStep = async () => {
            if (cancelled) return;

            const currentBoard = boardRef.current;

            const action = await requestAiMove(currentBoard);
            if (action === null || action === -1) {
                setAutoPlaying(false);
                return;
            }

            const keyMap = {
                0: "ArrowUp",
                1: "ArrowDown",
                2: "ArrowLeft",
                3: "ArrowRight",
            };

            const key = keyMap[action];
            const result = makeMove(key, currentBoard);

            if (result.newState !== null) {
                const newBoardAddTile = addRandomNumber(result.newState);
                if (newBoardAddTile !== null) {
                    setBoard(newBoardAddTile);
                    setScore((prev) => prev + result.score);

                    if (checkGameover(newBoardAddTile)) {
                        setGameover(true);
                        setAutoPlaying(false);
                        return;
                    }
                }
            }

            // 🐢 delay between moves
            setTimeout(doStep, 120); // 0.5s per move; change as you like
        };

        // start immediately:
        doStep();

        return () => {
            cancelled = true;
        };
    }, [autoPlaying, gameover]); // ❗ no "board" here

    const handleAiMoveClick = async () => {
        if (gameover) return;

        const action = await requestAiMove(board);
        if (action === null || action === -1) {
            // no move / error / gameover from backend
            return;
        }

        // Map numeric action -> Arrow keys expected by makeMove
        const keyMap = {
            0: "ArrowUp",
            1: "ArrowDown",
            2: "ArrowLeft",
            3: "ArrowRight",
        };

        const key = keyMap[action];
        if (!key) {
            console.error("Unknown action from AI:", action);
            return;
        }

        const result = makeMove(key, board);
        if (result.newState !== null) {
            const newBoardAddTile = addRandomNumber(result.newState);
            if (newBoardAddTile !== null) {
                setBoard(newBoardAddTile);
                setScore((prevScore) => prevScore + result.score);

                if (checkGameover(newBoardAddTile)) {
                    setGameover(true);
                }
            }
        }
    };

    // ---------- 4. Reset game ----------
    const resetGame = () => {
        const emptyBoard = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ];

        let newBoard = addRandomNumber(emptyBoard);
        newBoard = addRandomNumber(newBoard);

        setBoard(newBoard);
        setScore(0);
        setGameover(false);
    };

    return (
        <div
            style={{
                display: "grid",
                gridTemplateColumns: "2fr 3fr 2fr",
                justifyItems: "center",
            }}>
            <div></div>

            <div>
                <Scoreboard score={score} />
                <Board boardMatrix={board} />
            </div>

            <div style={{display: "flex", flexDirection: "column"}}>
                <button onClick={resetGame} style={{margin: "5rem 0 0 0", border: "1px solid black"}}>
                    Reset Game
                </button>

                <button
                    onClick={() => setAutoPlaying(true)}
                    disabled={autoPlaying}
                    style={{margin: "1rem 0", border: "1px solid black"}}>
                    Start Auto Play
                </button>

                <button
                    onClick={() => setAutoPlaying(false)}
                    disabled={!autoPlaying}
                    style={{margin: "0 0 1rem 0", border: "1px solid black"}}>
                    Stop
                </button>

                <button onClick={handleAiMoveClick} disabled={autoPlaying}>
                    Single AI Move
                </button>

                {gameover && <div>Game Over!</div>}
            </div>
        </div>
    );
}

export default GameContainer;
