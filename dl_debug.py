import isolation.isolation
import deeplearn04.deeplearn04 as dl
import numpy as np
import json


class HumanDLPlayer():
    """Player that chooses a move according to user's input."""

    def __init__(self, dlscore):
        self.dlscore = dlscore

    def get_move(self, game, legal_moves, time_left):
        """
        Select a move from the available legal moves based on user input at the
        terminal.

        **********************************************************************
        NOTE: If testing with this player, remember to disable move timeout in
              the call to `Board.play()`.
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
            The move in the legal moves list selected by the user through the
            terminal prompt; automatically return (-1, -1) if there are no
            legal moves
        """
        if not legal_moves:
            return (-1, -1)

        print(game.to_string())

#         dl_state = dl.get_state(game)
#         dl_state = np.reshape(dl_state,(3,7,7))
#         print(dl_state)

        score_list = None
        if len(legal_moves) <= 8:
            print('score')
            score_list = self.dlscore.score_list(game,self)[0].tolist()
            print('\n'.join(['%d %s %f'%(i, str(dl.idx_to_rc(game,i)), score_list[i]) for i in range(len(score_list))]))
            print('choice')
            print(('\n'.join(['[%d] %s %d %f' % (i, str(move), dl.rc_to_idx(game,move), score_list[dl.rc_to_idx(game,move)]) for i, move in enumerate(legal_moves)])))
        else:
            print('choice')
            print(('\n'.join(['[%d] %s' % (i, str(move)) for i, move in enumerate(legal_moves)])))

        valid_choice = False
        while not valid_choice:
            try:
                index = int(input('Select move index:'))
                valid_choice = 0 <= index < len(legal_moves)

                if not valid_choice:
                    print('Illegal move! Try again.')

            except ValueError:
                print('Invalid index! Try again.')

        return legal_moves[index]

if __name__ == '__main__':
    arg_dict = {}
    arg_dict['output_path'] = None
    arg_dict['random_stddev'] = 0.1
    arg_dict['random_move_chance'] = 0.
    arg_dict['train_beta'] = 0.99
    arg_dict['continue'] = False
    arg_dict['train_memory'] = 10
    dll = dl.DeepLearn(arg_dict)
#        dll.load_sess('tensorflow_resource/dl04-100000')
    dll.load_sess('tensorflow_resource/dl04-732000')
    dlscore = dl.Score(dll)
    
    player1 = HumanDLPlayer(dlscore)
    player2 = HumanDLPlayer(dlscore)

    while(True):
        game = isolation.isolation.Board(player1,player2)
        game.play(time_limit=999999)
