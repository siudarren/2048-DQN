import React, {useState, useEffect} from "react";
import Board from "./Board";
import Scoreboard from "./Scoreboard";
import {addRandomNumber, makeMove, checkGameover} from "../Utilities/Utils";
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

    // Use useEffect to trigger the addRandomNumber function on initial mount
    useEffect(() => {
        const newBoard = addRandomNumber(board);
        setBoard(newBoard);
    }, []); // The empty dependency array means this effect runs only once, after the initial render

    useEffect(() => {
        if (!gameover) {
            const handleKeyDown = (event) => {
                event.preventDefault(); // Prevent default to stop any default actions triggered by key presses
                const result = makeMove(event.key, board);
                if (result.newState !== null) {
                    // Check if newState is not null
                    const newBoardAddTile = addRandomNumber(result.newState);
                    if (newBoardAddTile !== null) {
                        setBoard(newBoardAddTile);
                        setScore((prevScore) => prevScore + result.score); // Update the score
                    }

                    if (checkGameover(board)) {
                        setGameover(true);
                    }
                }
            };

            // Attach the event listener to the window
            window.addEventListener("keydown", handleKeyDown);

            // Cleanup function to remove the event listener when the component unmounts
            return () => {
                window.removeEventListener("keydown", handleKeyDown);
            };
        }
    }, [board]); // Dependency on board to ensure event listener updates with current state

    const getNextMove = async () => {
        try {
            const response = await axios.post("http://127.0.0.1:5000/api/get-move", {
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
        setBoard(addRandomNumber(newBoard)); // Reset board and add a random number
        setScore(0); // Reset score
        setGameover(false); // Reset gameover status
    };

    return (
        <div style={{display: "grid", gridTemplateColumns: "2fr 3fr 2fr", justifyItems: "center"}}>
            <div></div>
            <div>
                <Scoreboard score={score}></Scoreboard>
                <Board boardMatrix={board}></Board>
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
