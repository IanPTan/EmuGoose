import torch as pt
import chess
import onnxruntime as ort
try:
    from ..utils.data import encode_board
except:
    from utils.data import encode_board


def make_nnue(onnx_path):
    session = ort.InferenceSession(onnx_path)
    return session


def batch_encode(boards):
    encodings = []
    for board in boards:
        encodings.append(encode_board(board))
    
    encodings = pt.stack(encodings).view(-1, 768)
    return encodings


class GameState:
    def __init__(self, nnue, board, wdl=None):
        self.board = board
        self.nnue = nnue
        self.wdl = wdl
    
    def evaluate(self):
        if self.wdl is not None:
            return self.wdl
        
        if self.is_done():
            outcome = self.board.outcome()
            self.wdl = 1 if outcome.winner == chess.WHITE else 0 if outcome.winner == chess.BLACK else 0.5
            return self.wdl
        
        encoding = encode_board(self.board).view(1, -1).numpy()
        # The model was exported with sigmoid, so we get the raw score.
        # The output from ONNX is a list of arrays.
        self.wdl = self.nnue.run(['output'], {'input': encoding})[0][0][0]

        return self.wdl
    
    def is_maximizing(self):
        return self.board.turn == chess.WHITE
    
    def is_done(self):
        return self.board.is_game_over()
    
    def get_moves(self):
        return list(self.board.generate_legal_moves())
    
    def move(self, move):
        new_board = self.board.copy()
        new_board.push(move)
        
        return GameState(self.nnue, new_board)