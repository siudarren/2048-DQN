export function addRandomNumber(state) {
    // Step 1: Collect all the zero cells
    let zeroCells = [];
    for (let i = 0; i < state.length; i++) {
        for (let j = 0; j < state[i].length; j++) {
            if (state[i][j] === 0) {
                zeroCells.push([i, j]);
            }
        }
    }

    // Deep copy the state to avoid direct mutations
    const newState = state.map((row) => [...row]);

    // Step 2: Check if there are any zero cells
    if (zeroCells.length > 0) {
        // Step 3: Randomly select one zero cell
        const randomIndex = Math.floor(Math.random() * zeroCells.length);
        const [row, col] = zeroCells[randomIndex];

        // Step 4: Decide whether to place a 2 or a 4
        const randomNumber = Math.random() < 0.9 ? 2 : 4; // 90% chance of 2, 10% chance of 4

        // Step 5: Place the number in the selected cell
        newState[row][col] = randomNumber;
    }

    return newState;
}

export function checkGameover(state) {
    for (let i = 0; i < state.length; i++) {
        for (let j = 0; j < state[i].length; j++) {
            if (state[i][j] === 0) {
                return false;
            }
        }
    }

    if (moveUp(state) !== null) {
        return false;
    }

    if (moveDown(state) !== null) {
        return false;
    }

    if (moveLeft(state) !== null) {
        return false;
    }

    if (moveRight(state) !== null) {
        return false;
    }

    return true;
}

function moveUp(state) {
    if (!state || !Array.isArray(state) || !state.length || !Array.isArray(state[0])) {
        return {newState: null, score: 0}; // Return a default value if state is not valid
    }

    let hasChanged = false; // Flag to check if the board has changed
    let score = 0; // Initialize score
    let newState = state.map((row) => [...row]); // Create a new array to avoid direct mutation of the state

    // Process each column
    for (let col = 0; col < newState[0].length; col++) {
        // Extract the current column into a new array
        let column = [];
        for (let row = 0; row < newState.length; row++) {
            if (newState[row][col] !== 0) {
                column.push(newState[row][col]);
            }
        }

        // Combine numbers in the column
        let newColumn = [];
        for (let i = 0; i < column.length; i++) {
            if (i < column.length - 1 && column[i] === column[i + 1]) {
                let combinedValue = column[i] * 2;
                newColumn.push(combinedValue);
                score += combinedValue; // Add combined value to score
                i++; // Skip the next number since it's combined
                hasChanged = true; // Mark that a change occurred
            } else {
                newColumn.push(column[i]);
            }
        }

        // Fill the remaining spaces with zeros
        while (newColumn.length < newState.length) {
            newColumn.push(0);
        }

        // Check if this new column differs from the original column, if not yet marked as changed
        if (!hasChanged && !newColumn.every((val, index) => val === state[index][col])) {
            hasChanged = true;
        }

        // Place the newColumn back into the newState
        for (let row = 0; row < newState.length; row++) {
            newState[row][col] = newColumn[row];
        }
    }

    // Return the new state and score if changes were made, else return null
    return hasChanged ? {newState, score} : {newState: null, score: 0};
}

function rotate90Clockwise(state) {
    if (!state || !state.length) {
        return null;
    }
    const n = state.length;
    let rotated = Array.from({length: n}, () => Array(n).fill(0));
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            rotated[j][n - i - 1] = state[i][j];
        }
    }
    return rotated;
}

function rotate90CounterClockwise(state) {
    if (!state || !state.length) {
        return null;
    }
    const n = state.length;
    let rotated = Array.from({length: n}, () => Array(n).fill(0));
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            rotated[n - j - 1][i] = state[i][j];
        }
    }
    return rotated;
}

function moveLeft(state) {
    // Rotate the board 90 degrees clockwise
    let rotated = rotate90Clockwise(state);

    // Apply the moveUp function to the rotated state
    let result = moveUp(rotated); // result is { newState, score }

    // Check if a move occurred
    if (result.newState === null) {
        return {newState: null, score: 0};
    }

    // Rotate back 90 degrees counterclockwise to restore original orientation
    let finalState = rotate90CounterClockwise(result.newState);

    return {
        newState: finalState,
        score: result.score,
    };
}

function moveRight(state) {
    // Rotate the board 90 degrees counterclockwise
    let rotated = rotate90CounterClockwise(state);

    // Apply the moveUp function to the rotated state
    let result = moveUp(rotated);

    // Check if a move occurred
    if (result.newState === null) {
        return {newState: null, score: 0};
    }

    // Rotate back 90 degrees clockwise to restore original orientation
    let finalState = rotate90Clockwise(result.newState);

    return {
        newState: finalState,
        score: result.score,
    };
}

function moveDown(state) {
    // Rotate the board 180 degrees
    let rotated = rotate90Clockwise(rotate90Clockwise(state));

    // Apply the moveUp function to the rotated state
    let result = moveUp(rotated);

    // Check if a move occurred
    if (result.newState === null) {
        return {newState: null, score: 0};
    }

    // Rotate back 180 degrees to restore original orientation
    let finalState = rotate90CounterClockwise(rotate90CounterClockwise(result.newState));

    return {
        newState: finalState,
        score: result.score,
    };
}

export function makeMove(direction, board) {
    let result;
    switch (direction) {
        case "ArrowUp":
            result = moveUp(board);
            break;
        case "ArrowDown":
            result = moveDown(board);
            break;
        case "ArrowLeft":
            result = moveLeft(board);
            break;
        case "ArrowRight":
            result = moveRight(board);
            break;
        default:
            return {newState: null, score: 0};
    }
    // Return the result only if a move occurred
    if (result.newState !== null) {
        return result;
    } else {
        return {newState: null, score: 0};
    }
}
