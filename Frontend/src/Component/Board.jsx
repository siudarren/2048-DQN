import React from "react";
import Tile from "./Tile";

function Board({boardMatrix}) {
    return <div style={{display: "flex", flexDirection: "column"}}>
        {boardMatrix.map((row, rowIndex) => (
            <div key={rowIndex} style={{display: "flex"}}>
                {row.map((value, colIndex) => (
                    <Tile key={`${rowIndex}-${colIndex}`} value={value} />
                ))}
            </div>
        ))}
    </div>;
}

export default Board;
