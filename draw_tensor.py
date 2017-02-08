import scipy.misc
import deeplearn08.deeplearn08 as dl

if __name__ == '__main__':
    arg_dict = {}
    arg_dict['output_path'] = None
    arg_dict['random_stddev'] = 0.1
    arg_dict['random_move_chance'] = 0.
    arg_dict['train_beta'] = 0.99
    arg_dict['continue'] = False
    arg_dict['train_memory'] = 10
    
    dll = dl.DeepLearn(arg_dict)
    dll.load_sess('tensorflow_resource/dl08-100000')

    key_list = list(dll.var_dict.keys())
    var_dict_list = [dll.var_dict[key] for key in key_list]
    x_list = dll.sess.run(var_dict_list)
    
    for i in range(len(key_list)):
        print(key_list[i])
        print(x_list[i].shape)
        if 'b' in key_list[i]:
            continue
        scipy.misc.toimage(x_list[i], cmin=-1, cmax=1).save('{}.png'.format(key_list[i]))
