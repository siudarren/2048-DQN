// src/components/Tile.jsx

import React, {useEffect, useRef, useState} from "react";

const tileColors = {
    2: {background: "#eee4da", color: "#776e65"},
    4: {background: "#ede0c8", color: "#776e65"},
    8: {background: "#f2b179", color: "#f9f6f2"},
    16: {background: "#f59563", color: "#f9f6f2"},
    32: {background: "#f67c5f", color: "#f9f6f2"},
    64: {background: "#f65e3b", color: "#f9f6f2"},
    128: {background: "#edcf72", color: "#f9f6f2"},
    256: {background: "#edcc61", color: "#f9f6f2"},
    512: {background: "#edc850", color: "#f9f6f2"},
    1024: {background: "#edc53f", color: "#f9f6f2"},
    2048: {background: "#edc22e", color: "#f9f6f2"},
    4096: {background: "#3E3C36", color: "#f9f6f2"},
    8192: {background: "#3E3C36", color: "#f9f6f2"},
    16384: {background: "#edc53f", color: "#f9f6f2"},
    32768: {background: "#edc22e", color: "#f9f6f2"},
    65536: {background: "#3E3C36", color: "#f9f6f2"},
    131072: {background: "#3E3C36", color: "#f9f6f2"},
};

function Tile({value, isSpawn}) {
    // We’ll animate via transform scale
    const [scale, setScale] = useState(1);
    const [transition, setTransition] = useState("none");
    const prevValueRef = useRef(value);

    useEffect(() => {
        const prev = prevValueRef.current;

        // Debug (optional):
        // console.log("Tile change", { prev, value, isSpawn });

        if (value !== prev) {
            // SPAWN: this cell was 0, now >0, and it's the chosen spawn cell
            if (isSpawn && value !== 0 && prev === 0) {
                // Start tiny, then grow
                setTransition("none");
                setScale(0.5);

                requestAnimationFrame(() => {
                    requestAnimationFrame(() => {
                        setTransition("transform 300ms ease-out");
                        setScale(1);
                    });
                });
            }
            // MERGE: value increased at same cell (not spawn)
            else if (prev > 0 && value > prev) {
                setTransition("transform 120ms ease-out");
                setScale(1.1);

                setTimeout(() => {
                    setScale(1);
                }, 120);
            }
            // MOVE / clear: no animation
            else {
                setTransition("none");
                setScale(1);
            }

            prevValueRef.current = value;
        }
    }, [value, isSpawn]);

    const baseStyle = {
        width: "100px",
        height: "100px",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        margin: "5px",
        fontSize: "40px",
        fontWeight: "bold",
        borderRadius: "5px",
        boxSizing: "border-box",
        transform: `scale(${scale})`,
        transition,
    };

    const {background, color} = tileColors[value] || {background: "#cdc1b4", color: "#776e65"};

    const tileStyle = {
        ...baseStyle,
        backgroundColor: background,
        color,
    };

    return <div style={tileStyle}>{value !== 0 ? value : ""}</div>;
}

export default Tile;
