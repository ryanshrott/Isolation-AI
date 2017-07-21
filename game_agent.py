"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import numpy as np
import math

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return -np.inf

    if game.is_winner(player):
        return np.inf
    
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - 3*opp_moves)


def look_ahead(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return float('-inf')

    if game.is_winner(player):
        return float('inf')
    
    
    own_legal_moves = game.get_legal_moves(player)
    opp_legal_moves = game.get_legal_moves(game.get_opponent(player))

    own_moves_forecast = len(own_legal_moves)
    opp_moves_forecast = len(opp_legal_moves)

    for x,y in zip(own_legal_moves,opp_legal_moves):
        own_moves_forecast += len(game.get_moves(x))
        opp_moves_forecast += len(game.get_moves(y))

    return float(own_moves_forecast - opp_moves_forecast)

def corner(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return -np.inf

    if game.is_winner(player):
        return np.inf
    
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    value = float(own_moves - 3*opp_moves)
    penalized_value = value - 2
    
    if game.get_player_location(player) in  [(0, 0), (0, game.height - 1), (game.width - 1, 0), (game.width - 1, game.height - 1)]: # in the corner so we should penalize 
        #print('in corner')
        return penalized_value
    
    return value


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return -np.inf

    if game.is_winner(player):
        return np.inf
    
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    
    if own_moves != opp_moves:
        return float(own_moves - opp_moves)
    else:
        player_y_pos, player_x_pos = game.get_player_location(player)
        y_cent, x_cent = int(game.height / 2), int(game.width / 2)
        opp_y_pos, opp_x_pos = game.get_player_location(game.get_opponent(player))
        own_dist = abs(player_y_pos - y_cent) + abs(player_x_pos - x_cent)
        opp_dist = abs(opp_y_pos - y_cent) + abs(opp_x_pos - x_cent)
        return float(opp_dist - own_dist)/10


def squared_difference(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return -np.inf

    if game.is_winner(player):
        return np.inf
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    return float(own_moves**2 - (3*opp_moves)**2)

def division(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return -np.inf

    if game.is_winner(player):
        return np.inf
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    
    if own_moves == 0:
        return -np.inf
    if opp_moves == 0:
        return np.inf
    
    return float(own_moves/opp_moves)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, weight = 3, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.w = weight
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1,-1)
        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            best_move = self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move
    

    
    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()    
        
        def minimax_decision(state, depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()   
            if not state.get_legal_moves():
                return (-1,-1)
            actions = {a: min_value(state.forecast_move(a), depth-1) for a in state.get_legal_moves()}
            return max(actions, key=actions.get)
        
        def max_value(state, depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()   
            if not state.get_legal_moves() or depth==0:
                return self.score(state, self)
            v = -np.inf
            for a in state.get_legal_moves():
                v = max(v, min_value(state.forecast_move(a), depth-1))
            return v
        
        def min_value(state, depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()   
            if not state.get_legal_moves() or depth==0:
                return self.score(state, self)
            v = np.inf
            for a in state.get_legal_moves():
                v = min(v, max_value(state.forecast_move(a), depth-1))
            return v
    
        return minimax_decision(game, depth)

# create an isolation board (by default 7x7)


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            search_depth = 1
            while True: 
                best_move = self.alphabeta(game, search_depth)
                search_depth += 1

        except SearchTimeout:
            return best_move

        # Return the best move from the last completed search iteration

        return best_move
    


    def alphabeta(self, game, depth, alpha=-np.inf, beta=np.inf):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()    
        
        def alpha_beta_search(state, depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()   
            if not state.get_legal_moves():
                return (-1,-1)
            (v, action) = max_value(state, alpha, beta, depth)
            return action

        def max_value(state, alpha, beta, depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()   
            if not state.get_legal_moves() or depth==0:
                return (self.score(state, self), (-1,-1))
            legal_moves = state.get_legal_moves()
            v = -np.inf
            bestAction = legal_moves[0]
            for a in legal_moves:
                (newV,_) = min_value(state.forecast_move(a), alpha, beta, depth-1)
                if newV > v:
                    v = newV
                    bestAction = a
                if v >= beta: # prune
                    return (v, bestAction) 
                alpha = max(alpha, v)
            return (v, bestAction)
        
        def min_value(state, alpha, beta, depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()   
            if not state.get_legal_moves() or depth==0:
                return (self.score(state, self), (-1,-1))
            legal_moves = state.get_legal_moves()
            v = np.inf
            bestAction = legal_moves[0]
            for a in legal_moves:
                (newV,_) = max_value(state.forecast_move(a), alpha, beta, depth-1)
                if newV < v:
                    v = newV
                    bestAction = a
                if v <= alpha: # prune
                    return (v, bestAction) 
                beta = min(beta, v)
            return (v, bestAction)
    
        return alpha_beta_search(game, depth)
