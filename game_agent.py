"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score_2(game, player):
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
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # get the opp player
    opp = game.get_opponent(player)

    # legal moves  
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(opp)

    # number of legal moves
    own_len = len(own_moves)
    opp_len = len(opp_moves)

    # positions
    center_w, center_h = game.width / 2., game.height / 2.
    own_y, own_x = game.get_player_location(player)
    opp_y, opp_x = game.get_player_location(opp)
    max_cdist = center_w ** 2 + center_h ** 2

    # default score
    # try to have 3 or more legal moves (treat them equally)
    if own_len >= 3:
        score = 3
    else:
        score = own_len

    # also check if the opp moves left are further from the center
    # find the min center distance from all available opp moves
    # adding this to the score should corner the opp

    # do this only when the player took one of the opp's moves

    if (abs(own_x - opp_x) == 1 and abs(own_y - opp_y) == 2) \
    or (abs(own_x - opp_x) == 2 and abs(own_y - opp_y) == 1):

        min_cdist = max_cdist
        for m in opp_moves:
            cdist = (center_w - m[0]) ** 2 + (center_h - m[1]) ** 2
            # if the distance is minimum (which is 0), then cut off the loop
            if cdist == 0:
                return float(score)
            elif cdist < min_cdist:
                min_cdist = cdist

        # score should be less than 1 at max
        # in order not to override other scores
        score += min_cdist / max_cdist
    else:
        # if opp move not taken

        # player is about to lose when there's only one move left
        # and the opp can take that move
        if own_len == 1 and own_moves[0] in opp_moves:
            return -100.0

        # choose a position closer to the center
        cdist = (center_w - own_x) ** 2 + (center_h - own_y) ** 2
        score -= cdist / max_cdist

        # if a legal move can be reduced by the opp's next move
        # reduce that from the score and cut off the loop
        for m in own_moves:
            if m in opp_moves:
                return float(score - 1)

    return float(score)

def custom_score(game, player):
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
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    opp = game.get_opponent(player)

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(opp))

    # default score
    score = own_moves - opp_moves

    # try not to be diagonally close to the opp
    own_y, own_x = game.get_player_location(player)
    opp_y, opp_x = game.get_player_location(opp)

    if abs(own_x - opp_x) == 1 and abs(own_y - opp_y) == 1:
        score -= 1

    # also, choose a position closer to the center
    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)

    cdist = (w - x)**2 + (h - y)**2

    return float(score - cdist * 0.1)

def custom_score_3(game, player):
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
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    opp = game.get_opponent(player)

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(opp))

    # default score
    score = own_moves - opp_moves    

    # center coords
    w, h = game.width / 2., game.height / 2.

    # player coords
    y, x = game.get_player_location(player)

    # count blank spaces (separate them left and right sides)
    left_blanks = 0
    right_blanks = 0

    for blank in game.get_blank_spaces():
        if blank[0] < w:
            left_blanks += 1
        elif blank[0] > w:
            right_blanks += 1

    # try to move to the side with more blanks
    if left_blanks > right_blanks:
        if x < w:
            score += 1
    elif right_blanks > left_blanks:
        if x > w:
            score += 1

    return float(score)

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
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
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
        
        if len(game.get_legal_moves()) > 0:
            best_move = game.get_legal_moves()[0]
        else:
            best_move = (-1,-1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            best_move = self.minimax(game, self.search_depth)
            return best_move

        except SearchTimeout:
            return best_move

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

        # TODO: finish this function!
        
        # terminal test with depth checking
        def terminal_test(game, depth):
            # time check
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()            
            return depth <= 0 or not bool(game.get_legal_moves())

        # min value func
        def min_value(game, depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()                        
            if terminal_test(game, depth):
                # evauation as written in Notes
                return self.score(game, self)
            v = float("inf")
            for m in game.get_legal_moves():
                # reduce depth for next state
                v = min(v, max_value(game.forecast_move(m), depth - 1))
            return v

        # max value func
        def max_value(game, depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()                        
            if terminal_test(game, depth):
                return self.score(game, self)
            v = float("-inf")
            for m in game.get_legal_moves():
                v = max(v, min_value(game.forecast_move(m), depth - 1))
            return v

        # decision process
        best_score = float("-inf")
        if len(game.get_legal_moves()) > 0:
            best_move = game.get_legal_moves()[0]
        else:
            best_move = (-1,-1) # default move as in the description
        for m in game.get_legal_moves():
            v = min_value(game.forecast_move(m), depth - 1)
            if v > best_score:
                best_score = v
                best_move = m
        return best_move

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

        # TODO: finish this function!
        legal_moves = game.get_legal_moves()
        if len(legal_moves) > 0:
            best_move = legal_moves[0]
        else:
            best_move = (-1,-1)

        # Iterative Deepening Search
        depth = 0
        while self.time_left() > self.TIMER_THRESHOLD:
            try:
                best_move = self.alphabeta(game, depth)
                depth += 1
            except SearchTimeout:
                return best_move

        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
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

        # TODO: finish this function!

        '''
        # terminal test with depth checking
        def terminal_test(game, depth):
            # time check
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()            
            return depth <= 0 or not bool(game.get_legal_moves())
        '''
        # min value func
        def min_value(game, depth, alpha=float("-inf"), beta=float("inf")):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            legal_moves = game.get_legal_moves()
            if len(legal_moves) > 0:
                best_move = legal_moves[0]
            else:
                best_move = (-1,-1)
            if depth <= 0 or len(legal_moves) == 0:
                return self.score(game, self), best_move
            '''
            if terminal_test(game, depth):
                # evauation as written in Notes and default action
                return self.score(game, self), best_move
            '''
            best_score = float("inf")
            for m in legal_moves:
                # reduce depth for next state
                score, _ = max_value(game.forecast_move(m), depth - 1, alpha, beta)
                # update score and action
                if score < best_score:
                    best_score = score
                    best_move = m
                # if score's smaller than alpha, then cut off
                if score <= alpha:
                    return score, m
                # otherwise update beta
                beta = min(beta, score)
            return best_score, best_move

        # max value func
        def max_value(game, depth, alpha=float("-inf"), beta=float("inf")):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()                        
            legal_moves = game.get_legal_moves()
            if len(legal_moves) > 0:
                best_move = legal_moves[0]
            else:
                best_move = (-1,-1)
            if depth <= 0 or len(legal_moves) == 0:
                return self.score(game, self), best_move
            best_score = float("-inf")
            for m in legal_moves:
                # reduce depth for next state
                score, _ = min_value(game.forecast_move(m), depth - 1, alpha, beta)
                # update
                if score > best_score:
                    best_score = score
                    best_move = m
                # cut off
                if score >= beta:
                    return score, m
                # update alpha
                alpha = max(alpha, score)
            return best_score, best_move

        # decision process
        v, a = max_value(game, depth, alpha, beta)
        return a