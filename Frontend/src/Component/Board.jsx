// src/components/Board.jsx
import React from "react";
import Tile from "./Tile";

function Board({boardMatrix, lastSpawn}) {
    return (
        <div style={{display: "flex", flexDirection: "column"}}>
            {boardMatrix.map((row, rowIndex) => (
                <div key={rowIndex} style={{display: "flex"}}>
                    {row.map((value, colIndex) => {
                        const isSpawn = lastSpawn && lastSpawn.row === rowIndex && lastSpawn.col === colIndex;

                        return <Tile key={`${rowIndex}-${colIndex}`} value={value} isSpawn={isSpawn} />;
                    })}
                </div>
            ))}
        </div>
    );
}

export default Board;
