from flask import Flask, jsonify, request
from flask_cors import CORS  # To handle Cross-Origin Resource Sharing

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

@app.route('/api/get-move', methods=['POST'])
def get_move():
    data = request.json
    # Placeholder for ML logic
    # You will process the game state and return the next move
    game_state = data.get('gameState')
    # For now, we'll return a random move
    move = 'up'  # This should be determined by your ML model
    return jsonify({'move': move})

if __name__ == '__main__':
    app.run(debug=True)