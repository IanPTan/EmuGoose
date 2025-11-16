import torch as pt
import chess
from ..NNUE.model import NNUE
from ..utils.data import encode_board


def make_nnue(nnue_path, config=[512, 32, 32]):
    nnue = NNUE(config)
    nnue.load(nnue_path)
    nnue.eval()
    return nnue


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
        
        encoding = encode_board(self.board).view(1, -1)
        self.wdl = self.nnue(encoding, sigmoid=False).item()

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