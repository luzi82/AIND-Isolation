# ref: https://github.com/luzi82/codelog.tensorflow.tictactoe/blob/master/src/codelog/tensorflow/tictactoe/deeplearn/dl0021/deeplearn0021.py

import tensorflow as tf
import json
import time, os
import copy
import random
import logging
import argparse
import math
import numpy as np
import isolation.isolation
import collections

MY_NAME = os.path.basename(os.path.dirname(__file__))

OUTPUT_COUNT = 8
# RANDOM_STDDEV = 0.1
# RANDOM_MOVE_CHANCE = 0.05

def new_state_ph():
    return tf.placeholder(tf.float32, [None,3,7,7])

def new_model_var_dict():
    stddev = 0.4
    ret = {}
    ret['w0']=tf.Variable(tf.random_normal([147,148] ,stddev=stddev,dtype=tf.float32))
    ret['b1']=tf.Variable(tf.random_normal([148]     ,stddev=stddev,dtype=tf.float32))
    ret['w2']=tf.Variable(tf.random_normal([148,102]  ,stddev=stddev,dtype=tf.float32))
    ret['b3']=tf.Variable(tf.random_normal([102]      ,stddev=stddev,dtype=tf.float32))
    ret['w4']=tf.Variable(tf.random_normal([102,55]   ,stddev=stddev,dtype=tf.float32))
    ret['b5']=tf.Variable(tf.random_normal([55]      ,stddev=stddev,dtype=tf.float32))
    ret['w6']=tf.Variable(tf.random_normal([55,OUTPUT_COUNT]   ,stddev=stddev,dtype=tf.float32))
    ret['b7']=tf.Variable(tf.random_normal([OUTPUT_COUNT]      ,stddev=stddev,dtype=tf.float32))
    return ret

def new_sample_var_dict(arg_dict):
    ret = {}
    ret['state_0']          = tf.Variable(tf.zeros([arg_dict['train_memory'],3,7,7], dtype=tf.float32),trainable=False)
    ret['choice_0']         = tf.Variable(tf.zeros([arg_dict['train_memory']], dtype=tf.int32),trainable=False)
    ret['state_1']          = tf.Variable(tf.zeros([arg_dict['train_memory'],3,7,7], dtype=tf.float32),trainable=False)
    ret['choice_mask_1']    = tf.Variable(tf.zeros([arg_dict['train_memory'],OUTPUT_COUNT], dtype=tf.float32),trainable=False)
    ret['reward_1']         = tf.Variable(tf.zeros([arg_dict['train_memory']], dtype=tf.float32),trainable=False)
    ret['cont_1']           = tf.Variable(tf.zeros([arg_dict['train_memory']], dtype=tf.float32),trainable=False)
    return ret

def new_sample_input_ph_dict():
    ret = {}
    ret['state_0']=new_state_ph()
    ret['choice_0']=tf.placeholder(tf.int32,[None])
    ret['state_1']=new_state_ph()
    ret['choice_mask_1']=tf.placeholder(tf.float32,[None,OUTPUT_COUNT])
    ret['reward_1']=tf.placeholder(tf.float32,[None])
    ret['cont_1']=tf.placeholder(tf.float32,[None])
    return ret

#def model_lhs_dict_to_rhs_dict(var_dict):
#    ret = {}
#    for k in var_dict:
#        ret[k] = tf.Variable(tf.zeros(var_dict[k].get_shape(), dtype=tf.float32),trainable=False)
#    return ret

def get_buffer_var_dict(src_var_dict):
    ret = {}
    for k in src_var_dict:
        buf = tf.Variable(tf.zeros(src_var_dict[k].get_shape(), dtype=tf.float32),trainable=False)
        ret[k] = tf.assign(buf, src_var_dict[k])
    return ret

def get_push_train_sample_var_list(sample_var_dict, train_input_ph_dict, push_indices):
    ret = []
    for k in train_input_ph_dict:
        push_op = tf.scatter_update(ref=sample_var_dict[k],indices=push_indices,updates=train_input_ph_dict[k])
        ret.append(push_op)
    return ret

def get_q(state_ph,var_dict):
    mid = state_ph
    mid = tf.reshape(mid, [-1,147])
    mid = tf.matmul(mid,var_dict['w0'])
    mid = mid + var_dict['b1']
    mid = tf.nn.elu(mid)
    mid = tf.matmul(mid,var_dict['w2'])
    mid = mid + var_dict['b3']
    mid = tf.nn.elu(mid)
    mid = tf.matmul(mid,var_dict['w4'])
    mid = mid + var_dict['b5']
    mid = tf.nn.elu(mid)
    mid = tf.matmul(mid,var_dict['w6'])
    mid = mid + var_dict['b7']
    mid = tf.maximum(mid, -1)
    mid = tf.minimum(mid, 1)
    return mid

def get_train_choice(state_ph,var_dict,random_t,mask,arg_dict):
    score = get_q(state_ph,var_dict)
    mid = score
#     mid = mid + random_t
    mid = mid + (1 - mask) * 10
    mid = mid - tf.reduce_min(mid)
    mid = mid * mask
    mid = mid + mask * 0.00001
    mid = mid / tf.reduce_max(mid)
    mid = mid * (1-arg_dict['random_move_chance'])
    mid = mid + arg_dict['random_move_chance']
    mid = mid * mask
    weight = mid
    
    weight_sum = tf.reduce_sum(weight,reduction_indices=[1])
    high = tf.cumsum(weight, axis=1, exclusive=False)
    low = tf.cumsum(weight, axis=1, exclusive=True)
    sss0 = tf.reshape(weight_sum,[-1,1])
    high0 = high / sss0
    low0 = low / sss0
    r = tf.random_uniform(tf.shape(sss0), dtype=tf.float32)
    high1 = tf.less(r, high0)
    low1 = tf.less_equal(low0, r)
    good = tf.logical_and(high1,low1)
    good0 = tf.to_float(good)
    mid = tf.argmax(good0, dimension=1)
    train_choice = mid

    mid = score
    mid = mid + random_t
    mid = mid - tf.reduce_min(mid)
    #mid = tf.exp(mid)
    mid = mid * mask
    mid = mid + mask
    mid = tf.argmax(mid, dimension=1)
    cal_choice = mid

    return score, weight, train_choice, cal_choice

# TRAIN_BETA = 0.99
# ELEMENT_L2_FACTOR = 10.0
# L2_WEIGHT = 0.1

def get_train(sample_var_dict,var_dict,arg_dict):
    mid0 = tf.one_hot(sample_var_dict['choice_0'], OUTPUT_COUNT, axis=-1, dtype=tf.float32)
    mid0 = mid0 * get_q(sample_var_dict['state_0'],var_dict)
    mid0 = tf.reduce_sum(mid0, reduction_indices=[1])

    buffer_var_dict = get_buffer_var_dict(var_dict)
    mid1 = get_q(sample_var_dict['state_1'],buffer_var_dict)
    mid1 = mid1 + tf.constant(-100., dtype=tf.float32) * ( tf.constant(1., dtype=tf.float32) - sample_var_dict['choice_mask_1'] )
    mid1 = tf.reduce_max(mid1, reduction_indices=[1])  
    mid1 = mid1 * sample_var_dict['cont_1']
    mid1 = mid1 * tf.constant(arg_dict['train_beta'])

    mid = mid0+mid1-sample_var_dict['reward_1']
    mid = tf.abs(mid)
    mid = tf.reduce_mean(mid)
    score_diff = mid

    loss = mid

    mid = tf.train.AdadeltaOptimizer(learning_rate=0.01).minimize(mid,var_list=var_dict.values())
    train = mid
    
    return train, loss, score_diff

# TRAIN_MEMORY = 20000

class DeepLearn(object):
    
    def __init__(self,arg_dict):
        self.arg_dict = arg_dict
        
        self.var_dict = new_model_var_dict()
        self.queue = {
            'state_0':       [None]*arg_dict['train_memory'],
            'choice_0':      [None]*arg_dict['train_memory'],
            'state_1':       [None]*arg_dict['train_memory'],
            'choice_mask_1': [None]*arg_dict['train_memory'],
            'reward_1':      [None]*arg_dict['train_memory'],
            'cont_1':        [None]*arg_dict['train_memory'],
        }
        self.random_t = tf.random_normal([OUTPUT_COUNT], stddev=arg_dict['random_stddev'])
        self.mask = tf.placeholder(tf.float32, [None,OUTPUT_COUNT])

        # choice
        self.choice_state = new_state_ph()
        self.score, self.weight, self.train_choice, self.choice_cal = get_train_choice(self.choice_state,self.var_dict,self.random_t,self.mask, self.arg_dict)
        
        # train
        self.push_indices = tf.placeholder(tf.int32,[None])
        self.sample_input_ph_dict = new_sample_input_ph_dict()
        self.sample_var_dict = new_sample_var_dict(self.arg_dict)
        self.push_train_sample_var_list = get_push_train_sample_var_list(self.sample_var_dict, self.sample_input_ph_dict, self.push_indices)
        self.train, self.loss, self.score_diff = get_train(self.sample_var_dict,self.var_dict,self.arg_dict)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        self.train_count = 0
        self.saver = tf.train.Saver(self.var_dict,max_to_keep=None)
        self.timestamp = time.time()
        self.timestamp_last = self.timestamp
        
        self.push_done = 0
        self.push_idx = 0
        self.push_var_idx = 0

        if ('continue' in arg_dict) and (arg_dict['continue']):
            while os.path.isfile(os.path.join(os.path.join(arg_dict['output_path'],'sess','{}.index'.format(self.train_count+1000)))):
                self.train_count += 1000
            self.load_sess(os.path.join(arg_dict['output_path'],'sess',str(self.train_count)))
        
    def load_sess(self,filename):
        self.saver.restore(self.sess, filename)

    def cal_choice(self, state_0, mask, train_enable):
        if train_enable:
            #logging.debug("EAPDALXUMV mask: "+json.dumps(mask))
            score, choice_0, weight = self.sess.run([self.score, self.train_choice, self.weight],feed_dict={self.choice_state:[state_0],self.mask:[mask]})
            score = score[0].tolist()
            weight = weight[0].tolist()
            choice_0 = choice_0.tolist()[0]
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug("UBNHCJHT mask {}".format(json.dumps(mask)))
                logging.debug("VJJDHFUI score {}".format(json.dumps(score)))
                logging.debug("ALEGYXDQ weight {}".format(json.dumps(weight)))
                logging.debug("FMUZWHSY choice {}".format(json.dumps(choice_0)))
            return {
                'state_0': state_0,
                'choice_0': choice_0,
                'state_1': None,
                'choice_mask_1': None,
                'cont': None,
                'reward_1': None,
            }, score[choice_0]
        else:
            score, choice_0 = self.sess.run([self.score, self.choice_cal],feed_dict={self.choice_state:[state_0],self.mask:[mask]})
            score = score[0].tolist()
            choice_0 = choice_0.tolist()[0]
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug("FCUFWSMO score {}, choice {}".format(json.dumps(score),choice_0))
            return {
                'state_0': state_0,
                'choice_0': choice_0,
                'state_1': None,
                'choice_mask_1': None,
                'cont': None,
                'reward_1': None,
            }, score[choice_0]
            

    def push_train_dict(self, train_dict):
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("EECSQBUX push_train_dict: "+json.dumps(train_dict))
        if self.push_idx >= self.arg_dict['train_memory']:
            logging.warn('SKXGEZAE self.push_idx >= self.arg_dict["train_memory"]')
            return
        for k, v in self.queue.items():
            v[self.push_idx] = train_dict[k]
        self.push_idx += 1

    def do_train(self):
        if self.push_idx > 0:
            feed_dict = {}
            for k in self.sample_var_dict:
                feed_dict[self.sample_input_ph_dict[k]] = self.queue[k][0:self.push_idx]
            indices = list(range(self.push_var_idx,self.push_var_idx+self.push_idx))
            indices = [i%self.arg_dict['train_memory'] for i in indices]
            feed_dict[self.push_indices] = indices
            self.sess.run(self.push_train_sample_var_list,feed_dict=feed_dict)
            self.push_done += self.push_idx
            self.push_var_idx += self.push_idx
            self.push_var_idx %= self.arg_dict['train_memory']
            self.push_idx = 0
        if self.push_done < self.arg_dict['train_memory']:
            return
        _, loss, score_diff = self.sess.run([self.train,self.loss,self.score_diff])
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug('ZPDDPYFD loss '+str(loss)+' '+str(score_diff))
        self.train_count += 1
        if self.train_count % 1000 == 0:
            output_file_name = os.path.join(self.arg_dict['output_path'],'sess',str(self.train_count))
            timestamp_last = time.time()
            if logging.getLogger().isEnabledFor(logging.INFO):
                logging.info('CLPNAVGR save session: {}, loss: {}, -log(loss): {}, time: {}, time_i: {}'.format(output_file_name,loss,-math.log(loss),int((time.time()-self.timestamp)*1000),int((timestamp_last-self.timestamp_last)*1000)))
            self.timestamp_last = timestamp_last
            os.makedirs(os.path.dirname(output_file_name),exist_ok=True)
            self.saver.save(self.sess,output_file_name)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug('HZQQMSQT '+MY_NAME+' '+str(self.train_count))

    def close(self):
        self.sess.close()

REWARD_WIN = 1.
REWARD_LOSE = -1.
REWARD_DRAW = 0.
REWARD_STEP = -0.05

KNIGHT_IDX2RC_V=[(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1),(-2,1),(-1,2),(1,2)]
KNIGHT_RC2IDX_D={}
for knight_idx2rc_i in range(len(KNIGHT_IDX2RC_V)):
    KNIGHT_RC2IDX_D[KNIGHT_IDX2RC_V[knight_idx2rc_i]] = knight_idx2rc_i

class DLPlayer(object):

    def __init__(self, dl):
        self.dl = dl
        self.train_enable = True

    def get_move(self, game, legal_moves, time_left):
        if len(legal_moves) == 0:
            return (-1,-1)

        ret_move = None

        if len(legal_moves)<=8:

            state_0 = self.get_state(game)
            for move in legal_moves:
                choice_0 = self.rc_to_idx(game,move)
                game_1 = game.forecast_move(move)
                
                train_dict={}
                train_dict['state_0']       = state_0
                train_dict['choice_0']      = choice_0
                train_dict['state_1']       = self.get_state(game_1)
                train_dict['choice_mask_1'] = self.get_choice_mask(game_1)
                train_dict['reward_1']      = self.get_reward(game_1)
                train_dict['cont_1']        = self.get_cont(game_1)

                self.dl.push_train_dict(train_dict)

            self.dl.do_train()

            ret_move = random.choice(legal_moves)
        else:
            ret_move = random.choice(legal_moves)
    
        if ret_move not in legal_moves:
            print('1' if game.active_player == game.__player_1__ else '2')
            print(game.to_string())
            print(ret_move)
            raise Exception('CZQAZCWH ret_move not in legal_moves')
        return ret_move

    def rc_to_idx(self,game,rc):
        r, c = rc
        ar, ac = game.get_player_location(game.active_player)
        dr, dc = (r-ar, c-ac)
        idx = KNIGHT_RC2IDX_D.get((dr,dc))
        return idx

    def idx_to_rc(self,game,idx):
        dr, dc = KNIGHT_IDX2RC_V[idx]
        ar, ac = game.get_player_location(game.active_player)
        rc = (ar+dr, ac+dc)
        return rc

    def get_state(self, game):
        ret_state = np.zeros((3,7,7),dtype=np.float)
        board_state_vv = [[bs==isolation.isolation.Board.BLANK for bs in bs_v] for bs_v in game.__board_state__]
        np.copyto(ret_state[0], board_state_vv)
        aloc = game.get_player_location(game.active_player)
        if aloc != None:
            active_r, active_c = aloc
            ret_state[1][active_r][active_c] = 1.0
        iloc = game.get_player_location(game.inactive_player)
        if iloc != None:
            inactive_r, inactive_c = iloc
            ret_state[2][inactive_r][inactive_c] = 1.0
        return ret_state.tolist()

    def get_choice_mask(self, game):
        legal_moves = game.get_legal_moves()
        ret_vv = np.zeros(OUTPUT_COUNT,dtype=np.float)
        for move in legal_moves:
            ret_vv[self.rc_to_idx(game, move)] = 1.
        return ret_vv.tolist()

    def get_reward(self, game):
        utility = game.utility(game.inactive_player) # reward of last mover
        if utility > 0.001:
            return REWARD_WIN
        if utility < -0.001:
            return REWARD_LOSE
        return REWARD_STEP

    def get_cont(self, game):
        utility = game.utility(game.active_player)
        return (utility < 0.001) and (utility > -0.001)

    def close(self):
        self.dl.close()


def main(_):
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--output_path',
        type=str,
        help='The path to which checkpoints and other outputs '
        'should be saved. This can be either a local or GCS '
        'path.',
        default=None
    )
    argparser.add_argument(
        '--random_stddev',
        type=float,
        help='random_stddev',
        default=0.1
    )
    argparser.add_argument(
        '--random_move_chance',
        type=float,
        help='RANDOM_MOVE_CHANCE',
        default=0.05
    )
    argparser.add_argument(
        '--train_beta',
        type=float,
        help='TRAIN_BETA',
        default=0.99
    )
    argparser.add_argument(
        '--turn_count',
        type=int,
        help='turn_count',
        default=None
    )
    argparser.add_argument(
        '--device',
        type=str,
        help='device',
        default=None
    )
    argparser.add_argument(
        '--continue',
        action='store_true',
        help='continue'
    )
#     argparser.add_argument(
#         '--element_l2_factor',
#         type=float,
#         help='ELEMENT_L2_FACTOR',
#         default=10.0
#     )
#     argparser.add_argument(
#         '--l2_weight',
#         type=float,
#         help='L2_WEIGHT',
#         default=0.1
#     )
    argparser.add_argument(
        '--train_memory',
        type=int,
        help='TRAIN_MEMORY',
        default=10000
    )
    args, _ = argparser.parse_known_args()
    arg_dict = vars(args)
    if logging.getLogger().isEnabledFor(logging.INFO):
        logging.info('YGYMBFMN arg_dict {}'.format(json.dumps(arg_dict)))
    if(arg_dict['output_path']==None):
        if not arg_dict['continue']:
            timestamp = int(time.time())
            arg_dict['output_path'] = os.path.join('output',MY_NAME,str(timestamp),'deeplearn')
        else:
            filename_list = os.listdir(os.path.join('output',MY_NAME))
            filename_int_list = [util.to_int(filename,-1) for filename in filename_list]
            arg_timestamp = str(max(filename_int_list))
            arg_dict['output_path'] = os.path.join('output',MY_NAME,str(arg_timestamp),'deeplearn')

    if arg_dict['continue']:
        with open(os.path.join(arg_dict['output_path'],'input_arg_dict.json'),'r') as deeplearn_arg_dict_file:
            deeplearn_arg_dict = json.load(deeplearn_arg_dict_file)
        arg_dict = deeplearn_arg_dict
        arg_dict['continue'] = True
    else:
        os.makedirs(arg_dict['output_path'],exist_ok=True)
        with open(os.path.join(arg_dict['output_path'],'input_arg_dict.json'),'w') as out_file:
            json.dump(arg_dict,out_file)

    dl = DeepLearn(arg_dict)

    player1 = DLPlayer(dl)
    player2 = DLPlayer(dl)

    with tf.device(arg_dict['device']):
        while(True):
            game = isolation.isolation.Board(player1,player2)
            game.play(time_limit=999999)


if __name__ == '__main__':
    log_filename = os.path.join('log','{}-{}-deeplearn.log'.format(str(int(time.time())),MY_NAME))
    os.makedirs(os.path.dirname(log_filename),exist_ok=True)
    logging.basicConfig(level=logging.INFO,filename=log_filename)
#    logging.basicConfig(level=logging.DEBUG,filename=log_filename)
    tf.app.run()
