import chess
import math


def minimax(gamestate, depth=4):

    #if depth > 1:
        #print(f"depth = {depth}\n{gamestate.board}\n\n")

    if depth == 0 or gamestate.is_done():
        return gamestate.evaluate(), None
    
    maximizing = gamestate.is_maximizing()
    compare = (lambda a, b: a > b) if maximizing else (lambda a, b: a < b)
    
    moves = gamestate.get_moves()
    #print(f"moves: {len(moves)}")

    best_move = moves[0]
    best_eval, _ = minimax(gamestate.move(best_move), depth - 1)

    for move in moves[1:]:
        eval, _ = minimax(gamestate.move(move), depth - 1)

        if compare(eval, best_eval):
            best_eval = eval
            best_move = move
    
    return best_eval, best_move


def alphabeta(gamestate, depth, alpha, beta):
    """
    Performs the recursive minimax search with alpha-beta pruning.
    """
    # Base case: if we're at the max depth or the game is over
    if depth == 0 or gamestate.is_done():
        # Return the static evaluation of the board
        return gamestate.evaluate(), None

    moves = gamestate.get_moves()
    if not moves:
        # If there are no legal moves, it's stalemate or checkmate
        return gamestate.evaluate(), None

    # This is the move we will return
    best_move = moves[0]

    # --- Maximizing Player's Turn ---
    if gamestate.is_maximizing():
        best_eval = -math.inf  # Start with the worst possible score
        
        for move in moves:
            # Recursively call the function for the next state
            # Note: the next player is a minimizing player (is_maximizing=False)
            eval, _ = alphabeta(gamestate.move(move), depth - 1, alpha, beta)

            # Update the best evaluation found so far
            if eval > best_eval:
                best_eval = eval
                best_move = move

            # Update alpha (the best score for the maximizer)
            alpha = max(alpha, eval)
            
            # --- The Alpha-Beta Pruning ---
            # If our alpha is already greater than or equal to beta,
            # the minimizing player will never let us reach this node.
            # We can stop searching this branch.
            if alpha >= beta:
                break
        
        return best_eval, best_move

    # --- Minimizing Player's Turn ---
    else:
        best_eval = math.inf  # Start with the worst possible score
        
        for move in moves:
            # Recursively call the function for the next state
            # Note: the next player is a maximizing player (is_maximizing=True)
            eval, _ = alphabeta(gamestate.move(move), depth - 1, alpha, beta)

            # Update the best evaluation found so far
            if eval < best_eval:
                best_eval = eval
                best_move = move

            # Update beta (the best score for the minimizer)
            beta = min(beta, eval)
            
            # --- The Alpha-Beta Pruning ---
            # If our beta is already less than or equal to alpha,
            # the maximizing player will never choose this path.
            # We can stop searching this branch.
            if beta <= alpha:
                break
                
        return best_eval, best_move


if __name__ == "__main__":
    from utils.board import make_nnue, GameState

    nnue = make_nnue("NNUE/models/nnue.pth")
    board = chess.Board("r1b4r/1p3kb1/p2pp1p1/3q3p/3N1Pp1/2P3R1/PP1Q1BPP/4R1K1 w - - 6 23")
    gs = GameState(nnue, board)
    a, b = minimax(gs, 2)
    c, d = alphabeta(gs, 2, -math.inf, math.inf)