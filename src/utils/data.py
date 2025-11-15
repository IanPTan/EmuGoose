import chess
import torch as pt


CHANNELS = {
    'P': 0,  # White Pawns
    'N': 1,  # White Knights
    'B': 2,  # White Bishops
    'R': 3,  # White Rooks
    'Q': 4,  # White Queens
    'K': 5,  # White King
    'p': 6,  # Black Pawns
    'n': 7,  # Black Knights
    'b': 8,  # Black Bishops
    'r': 9,  # Black Rooks
    'q': 10, # Black Queens
    'k': 11, # Black King
}


def encode_board(board):
    encoding = pt.zeros((12, 8, 8), dtype=pt.float32)
    
    for r_idx in range(8):
        for f_idx in range(8):
            square_index = chess.square(f_idx, 7 - r_idx)
            piece = board.piece_at(square_index)

            if piece:
                channel = CHANNELS[piece.symbol()]
                encoding[channel, r_idx, f_idx] = 1
                
    return encoding


def encode_fen(fen):
    board = chess.Board(fen)
    encoding = encode_board(board)

    return encoding
