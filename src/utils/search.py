import chess
import math
import torch as pt
try:
    from .board import batch_encode
except:
    from utils.board import batch_encode
from collections import defaultdict


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


def iterative_search(board, nnue, max_depth=3):
    """
    Performs an iterative, breadth-first minimax search.
    It evaluates all leaf nodes at the final depth in a single batch.
    This is not a true alpha-beta search but is structured for batch evaluation.
    """
    if board.is_game_over():
        return 0, None

    # The tree stores tuples of (board, parent_index, move_that_led_here)
    # Level 0 is the root
    tree = [[(board, -1, None)]]

    # --- Build the tree level by level (Forward Pass) ---
    for depth in range(max_depth):
        next_level_nodes = []
        parent_nodes = tree[depth]

        for i, (parent_board, _, _) in enumerate(parent_nodes):
            if parent_board.is_game_over():
                continue  # Don't expand terminal nodes

            for move in parent_board.legal_moves:
                child_board = parent_board.copy()
                child_board.push(move)
                next_level_nodes.append((child_board, i, move))

        if not next_level_nodes:
            break  # Stop if no further moves are possible

        tree.append(next_level_nodes)

    # --- Evaluate leaf nodes and propagate scores (Backward Pass) ---
    leaf_nodes = tree[-1]
    leaf_boards = [node[0] for node in leaf_nodes]

    # Batch evaluate all unique leaf positions
    encodings = batch_encode(leaf_boards).numpy()
    # The ONNX model has sigmoid built-in
    evals = nnue.run(['output'], {'input': encodings})[0].flatten()

    # Store evaluations for the current (leaf) level
    level_evals = list(evals)

    # Propagate evaluations up the tree
    for depth in range(len(tree) - 2, -1, -1):
        parent_nodes = tree[depth]
        parent_evals = [-1] * len(parent_nodes)

        # Group children by parent
        child_groups = defaultdict(list)
        for i, (_, parent_idx, _) in enumerate(tree[depth + 1]):
            child_groups[parent_idx].append(level_evals[i])

        for i, (parent_board, _, _) in enumerate(parent_nodes):
            child_evals = child_groups.get(i)
            if not child_evals:
                outcome = parent_board.outcome()
                parent_evals[i] = 1.0 if outcome and outcome.winner else 0.0 if outcome and outcome.winner is False else 0.5
            elif parent_board.turn == chess.WHITE:  # Maximizing player
                parent_evals[i] = max(child_evals)
            else:  # Minimizing player
                parent_evals[i] = min(child_evals)
        level_evals = parent_evals

    # Find the best move from the root's children
    best_move = None
    best_eval = -math.inf if board.turn == chess.WHITE else math.inf
    compare = max if board.turn == chess.WHITE else min

    child_indices = [i for i, (_, parent_idx, _) in enumerate(tree[1]) if parent_idx == 0]
    if not child_indices:
        return 0, list(board.legal_moves)[0] if list(board.legal_moves) else None

    best_child_idx = compare(child_indices, key=lambda i: level_evals[i])
    best_eval = level_evals[best_child_idx]
    best_move = tree[1][best_child_idx][2]

    return best_eval, best_move


if __name__ == "__main__":
    from utils.board import make_nnue, GameState

    nnue = make_nnue("NNUE/models/nnue.onnx")
    board = chess.Board("r1b4r/1p3kb1/p2pp1p1/3q3p/3N1Pp1/2P3R1/PP1Q1BPP/4R1K1 w - - 6 23")
    gs = GameState(nnue, board)
    #a, b = minimax(gs, 2)
    a, b = alphabeta(gs, 3, -math.inf, math.inf)
    #a, b = iterative_search(board, nnue, max_depth=3)
