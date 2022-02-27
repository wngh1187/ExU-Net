import os
from itertools import chain
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type = str, required = True)
    parser.add_argument("-module_name", type=str, required=True)
    parser.add_argument("-model_name", type=str, required=True)

    #dir
    parser.add_argument('-path_logging', type = str, default = '/results')
    parser.add_argument('-path_vox1_train', type = str, default = '/datas/VoxCeleb1/train')
    parser.add_argument('-path_vox1_test', type = str, default = '/datas/VoxCeleb1/test')
    parser.add_argument('-path_vox1_trials', type = str, default = '/datas/VoxCeleb1/trials.txt')
    parser.add_argument('-path_musan', type = str, default = '/datas/musan')
    parser.add_argument('-path_musan_split', type = str, default = '/datas/musan_split')
    parser.add_argument('-load_model_path', type=str, default="../weights/Baseline.pt")

    #'tqdm_ncols'    : 90,
    #hyper-params
    parser.add_argument('-rand_seed', type = int, default = 1234) 
    parser.add_argument('-batch_size', type = int, default = 120)
    parser.add_argument('-epoch', type = int, default = 500)
    parser.add_argument('-optimizer', type = str, default = 'adam')
    parser.add_argument('-usable_gpu', type = str, default = '0,1')
    parser.add_argument('-lr_start', type = float, default = 1e-3)
    parser.add_argument('-lr_end', type = float, default = 1e-7)
    parser.add_argument('-weigth_decay', type = float, default = 1e-4)
    parser.add_argument('-learning_rate_scheduler', type = str, default = 'step')
    parser.add_argument('-number_cycle', type = int, default = 80) 
    parser.add_argument('-number_iteration_for_log', type = int, default = 50) 
    parser.add_argument('-num_workers', type = int, default = 20) 

    #loss
    parser.add_argument('-classification_loss', type = str, default = 'softmax')
    parser.add_argument('-feature_enhancement_loss', type = str, default = 'mse')
    parser.add_argument('-embd_enhancement_loss', type = str, default = 'angleproto')
    parser.add_argument('-weight_classification_loss', type = float, default = 1)
    parser.add_argument('-weight_feature_enhancement_loss', type = float, default = 1)
    parser.add_argument('-weight_embd_enhancement_loss', type = float, default = 1)

    #DNN args
    parser.add_argument("-m_l_channel", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("-m_l_num_convblocks", type=int, nargs="+", default=[3, 4, 6, 3])
    parser.add_argument("-m_l_stride", type=int, nargs="+", default=[1,2,2,1])
    parser.add_argument("-m_first_kernel_size", type=int, default=7)
    parser.add_argument("-m_first_stride_size", type=int, default=(2,1))
    parser.add_argument("-m_first_padding_size", type=int, default=3)
    parser.add_argument("-m_code_dim", type=int, default=128)

    #data
    parser.add_argument('-nb_utt_per_spk', type = int, default = 2)
    parser.add_argument('-max_seg_per_spk', type = int, default = 500)
    parser.add_argument('-train_frame', type = int, default = 254) 
    parser.add_argument('-test_frame', type = int, default = 382) 
    parser.add_argument('-winlen', type = int, default = 400) 
    parser.add_argument('-winstep', type = int, default = 160) 
    parser.add_argument('-nfft', type = int, default = 1024) 
    parser.add_argument('-samplerate', type = int, default = 16000) 
    parser.add_argument('-nfilts', type = int, default = 64) 

    #flag
    parser.add_argument('-amsgrad', type = str2bool, nargs='?', const=True, default = True)
    parser.add_argument('-do_train_feature_enhancement', type = str2bool, nargs='?', const=True, default = False)
    parser.add_argument('-do_train_embd_enhancement', type = str2bool, nargs='?', const=True, default = False)
    parser.add_argument('-train_only_clean', type = str2bool, nargs='?', const=True, default = False)
    parser.add_argument('-reproduciable', type = str2bool, nargs='?', const=True, default = True)
    parser.add_argument('-eval', type = str2bool, nargs='?', const=True, default = False)

    args = parser.parse_args()
    args.model = {}
    for k, v in vars(args).items():
        if k[:2] == 'm_':
            args.model[k[2:]] = v
    return args