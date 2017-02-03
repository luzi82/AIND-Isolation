"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import json
import numpy as np
from collections import deque
import functools
import isolation.isolation


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """

    utility = game.utility(player)
    if utility < -0.0001 or utility > 0.0001:
        return utility

    return custom_score_0(game, player)


knight_v=[(-1,-2),(1,-2),(-1,2),(1,2),(-2,-1),(2,-1),(-2,1),(2,1)]

def custom_score_0(game, player):
    p0 = player
    p1 = game.get_opponent(p0)
    board_score_vv_0 = get_cs0_board_score_vv_dict((game.width,game.height),game.get_player_location(p0))
    board_score_vv_1 = get_cs0_board_score_vv_dict((game.width,game.height),game.get_player_location(p1))
    board_score_vv_d = board_score_vv_0-board_score_vv_1
    board_state_vv = [[bs==isolation.isolation.Board.BLANK for bs in bs_v] for bs_v in game.__board_state__]
    board_state_vv = np.array(board_state_vv)
    return np.sum(board_state_vv*board_score_vv_d)

cs0_decay = 1./8

@functools.lru_cache(maxsize=None)
def get_cs0_board_score_vv_dict(size_wh,location):
    w, h = size_wh
    r, c = location
    ret = np.zeros((h,w))
    ret[r][c] = 1
    location_score_q = deque([(location,1)])
    while(len(location_score_q)>0):
        loc0, score = location_score_q.popleft()
        score = score*cs0_decay
        knight_move_v = get_knight_move_v(size_wh,loc0)
        for knight_move in knight_move_v:
            r, c = knight_move
            if ret[r][c] != 0:
                continue
            ret[r][c] = score
            location_score_q.append((knight_move,score))
    return ret

@functools.lru_cache(maxsize=None)
def get_knight_move_v(size_wh,location):
    w, h = size_wh
    r, c = location
    ret = []
    for knight in knight_v:
        rr, cc = knight
        rr += r
        cc += c
        if rr<0:
            continue
        if rr>=h:
            continue
        if cc<0:
            continue
        if cc>=w:
            continue
        ret.append((rr,cc))
    return ret

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

        self.search_fn = self.minimax if method == 'minimax' else self.alphabeta

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        ----------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        search_depth = self.search_depth if (self.search_depth >= 0) else game.width*game.height + 1

        ret = (-1,-1)
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.iterative:
                for i in range(search_depth):
                    _, ret = self.search_fn(game, i+1, legal_moves=legal_moves)
            else:
                _, ret = self.search_fn(game, search_depth, legal_moves=legal_moves)

        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        return ret

    def minimax(self, game, depth, maximizing_player=True, legal_moves=None):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if legal_moves == None:
            legal_moves = game.get_legal_moves()

        if len(legal_moves) <= 0:
            return self.score(game, self), (-1, -1)

        if depth == 0:
            return self.score(game, self), (-1, -1)

        ret_score = None
        ret_move_list = None
        for move in legal_moves:
            game_0 = game.forecast_move(move)
            tmp_score, _ = self.minimax(game_0, depth-1, not maximizing_player)
            if ret_score == None or ( maximizing_player and ( tmp_score>ret_score ) ) or ( (not maximizing_player) and ( tmp_score<ret_score ) ):
                ret_score = tmp_score
                ret_move_list = [move]
            elif tmp_score == ret_score:
                ret_move_list.append(move)

        return ret_score, random.choice(ret_move_list)

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True, legal_moves=None):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

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

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if legal_moves == None:
            legal_moves = game.get_legal_moves()

        if len(legal_moves) <= 0:
            return self.score(game, self), (-1, -1)

        if depth == 0:
            return self.score(game, self), (-1, -1)

        # copy from wikipedia pseudocode
        # https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
        if maximizing_player:
            v = float("-inf")
            ret_move_list = [(-1,-1)]
            for move in legal_moves:
                game_0 = game.forecast_move(move)
                vv, _ = self.alphabeta(game_0, depth-1, alpha, beta, not maximizing_player)
                if vv>v:
                    ret_move_list = [move]
                elif vv==v:
                    ret_move_list.append(move)
                v = max(v, vv)
                alpha = max(alpha, v)
                if beta <= alpha:
                    break
            return v, random.choice(ret_move_list)
        else:
            v = float("inf")
            ret_move_list = [(-1,-1)]
            for move in legal_moves:
                game_0 = game.forecast_move(move)
                vv, _ = self.alphabeta(game_0, depth-1, alpha, beta, not maximizing_player)
                if vv<v:
                    ret_move_list = [move]
                elif vv==v:
                    ret_move_list.append(move)
                v = min(v, vv)
                beta = min(beta, v)
                if beta <= alpha:
                    break
            return v, random.choice(ret_move_list)
