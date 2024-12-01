// src/components/GameContainer.jsx

import React, {useState, useEffect, useContext} from "react";
import Board from "./Board";
import Scoreboard from "./Scoreboard";
import {addRandomNumber, makeMove, checkGameover} from "../Utilities/utils.js";
import axios from "axios";

function GameContainer() {
    const [board, setBoard] = useState([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]);
    const [score, setScore] = useState(0);
    const [gameover, setGameover] = useState(false);

    // Initial random number addition
    useEffect(() => {
        const newBoard = addRandomNumber(board);
        const newBoard2 = addRandomNumber(newBoard);
        setBoard(newBoard2);
    }, []);

    // Handle keyboard inputs for manual play
    useEffect(() => {
        if (!gameover) {
            const handleKeyDown = (event) => {
                event.preventDefault();
                const result = makeMove(event.key, board);
                if (result.newState !== null) {
                    const newBoardAddTile = addRandomNumber(result.newState);
                    if (newBoardAddTile !== null) {
                        setBoard(newBoardAddTile);
                        setScore((prevScore) => prevScore + result.score);
                    }

                    if (checkGameover(board)) {
                        setGameover(true);
                    }
                }
            };

            window.addEventListener("keydown", handleKeyDown);

            return () => {
                window.removeEventListener("keydown", handleKeyDown);
            };
        }
    }, [board, gameover]);

    // Function to get the next move from the backend (if needed)
    const getNextMove = async () => {
        try {
            const response = await axios.post("http://localhost:5000/api/get-move", {
                // Updated to port 5001
                gameState: board,
                currentScore: score,
                gameover: gameover,
            });
            const nextMove = response.data.move; // e.g., 'up', 'down', 'left', 'right'

            // Map the move to the key format expected by makeMove
            const keyMap = {
                up: "ArrowUp",
                down: "ArrowDown",
                left: "ArrowLeft",
                right: "ArrowRight",
            };
            const key = keyMap[nextMove];

            // Apply the move
            const result = makeMove(key, board);
            if (result.newState !== null) {
                const newBoardAddTile = addRandomNumber(result.newState);
                if (newBoardAddTile !== null) {
                    setBoard(newBoardAddTile);
                    setScore((prevScore) => prevScore + result.score);
                }

                if (checkGameover(newBoardAddTile)) {
                    setGameover(true);
                }
            }
        } catch (error) {
            console.error("Error fetching next move:", error);
        }
    };

    // Reset game function
    const resetGame = () => {
        const newBoard = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ];

        const newBoard1 = addRandomNumber(newBoard);
        setBoard(addRandomNumber(newBoard1));
        setScore(0);
        setGameover(false);
        setTrainingData(null);
    };

    return (
        <div style={{display: "grid", gridTemplateColumns: "2fr 3fr 2fr", justifyItems: "center"}}>
            <div></div>
            <div>
                <Scoreboard score={score} />
                <Board boardMatrix={board} />
            </div>
            <div style={{display: "flex", flexDirection: "column"}}>
                <button onClick={resetGame} style={{margin: "5rem 0 0 0", border: "1px solid black"}}>
                    Reset Game
                </button>
                <button onClick={getNextMove} style={{margin: "1rem 0", border: "1px solid black"}}>
                    Get Next Move
                </button>
                {gameover && <div>Game Over!</div>}
            </div>
        </div>
    );
}

export default GameContainer;
