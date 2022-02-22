import os
from itertools import chain
import torch

def get_args():
    system_args = {
        # expeirment info
        'project'       : 'SV_UNC_jhjw',
        'name'          : 'ResZNet_008',
        'tags'          : ['Proposed'],
        'description'   : 'ResZnet_with_skippath',

        # local
        'path_logging'  : '/results',

        # wandb
        'wandb_group'         : 'Juho', 
        'wandb_entity'        : 'ir-lab', 

        # VoxCeleb1 DB
        'path_vox1_train'   : '/datas/VoxCeleb1/train',
        'path_vox1_test'    : '/datas/VoxCeleb1/test',
        'path_vox1_trials'  : '/datas/VoxCeleb1/trials.txt',

        # musan DB
        'path_musan'        : '/datas/musan',

        # device
        'num_workers'   : 20,
        'usable_gpu'    : '0,1',
        'tqdm_ncols'    : 90,
        'path_scripts'     : os.path.dirname(os.path.realpath(__file__))
    }
    
    experiment_args = {
        # env
        'epoch'                     : 500,
        'batch_size'                : 120,
        'number_cycle'              : 80,
        'number_iteration_for_log'  : 50,
        'rand_seed'                 : 1234,
        'flag_reproduciable'        : True,
        
        # train process
        'do_train_feature_enhancement'  : True,
        'do_train_code_enhancement'     : True,

        # optimizer
        'optimizer'                 : 'adam',
        'amsgrad'                   : True,
        'learning_rate_scheduler'   : 'step',
        'lr_start'                  : 1e-3,
        'lr_end'                    : 1e-7,
        'weigth_decay'              : 1e-4,

        # criterion
        'classification_loss'                               : 'softmax',
        'enhancement_loss'                                  : 'mse',
        'code_enhacement_loss'                              : 'angleproto',
        'weight_classification_loss'                        : 1,
        'weight_code_enhancement_loss'                      : 1,
        'weight_feature_enhancement_loss'                   : 1,

        # model
        'first_kernel_size'     : 7,
        'first_stride_size'     : (2,1),
        'first_padding_size'    : 3,
        'l_channel'             : [16, 32, 64, 128],
        'l_num_convblocks'      : [3, 4, 6, 3],
        'code_dim'              : 128,
        'stride'                : [1,2,2,1],

        # data
        'nb_utt_per_spk'    : 2,
        'max_seg_per_spk'   : 500,
        'winlen'            : 400,
        'winstep'           : 160,
        'train_frame'       : 254,
        'nfft'              : 1024,
        'samplerate'        : 16000,
        'nfilts'            : 64,
        'premphasis'        : 0.97,
        'winfunc'           : torch.hamming_window,
        'test_frame'        : 382
    }

    # set args (system_args + experiment_args)
    args = {}
    for k, v in chain(system_args.items(), experiment_args.items()):
        args[k] = v

    return args, system_args, experiment_args