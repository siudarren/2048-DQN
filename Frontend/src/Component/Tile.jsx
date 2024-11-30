import React from "react";

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
    // Add more if the game includes tiles above 2048
};

function Tile({value}) {
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
    };

    // Get the appropriate color based on the tile's value or use default if the value is not defined
    const {background, color} = tileColors[value] || {background: "#cdc1b4", color: "#776e65"};

    // Combine base style with dynamic background and text color
    const tileStyle = {
        ...baseStyle,
        backgroundColor: background,
        color: color,
    };

    return <div style={tileStyle}>{value !== 0 ? value : ""}</div>;
}

export default Tile;
