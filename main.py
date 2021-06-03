import argparse
import os
import yaml
from torch.backends import cudnn
from solver import Solver
from data_loader import get_loader
def main(config):
    cudnn.benchmark = True
    if config.mode == 'train':
        train_data_loader = get_loader(config, config.face_image_path, config.face_crop_size,
                                   config.image_size, config.batch_size, 'face', config.mode)
        train_data_loaderB = get_loader(config,config.face_image_pathB, config.face_crop_size,
                                    config.image_size, config.batch_size, 'face','train_illumination')
        test_data_loader = get_loader(config,config.face_image_path_val, config.face_crop_size, config.image_size, 2, 'face'
                                      , 'val')
    elif config.mode == 'val':
        test_data_loader = get_loader(config, config.face_image_path_val, config.face_crop_size, config.image_size, 2, 'face', 'val')

        if not os.path.exists(config.result_path):
            os.makedirs(config.result_path)
    else:
        test_data_loader = get_loader(config,config.face_image_path, config.face_crop_size,
                                      config.image_size, config.batch_size, 'face', config.mode)
    # Solver

    if config.mode == 'train':

        solver = Solver(train_data_loader, train_data_loaderB, test_data_loader, config)
        solver.train()

    else :
        solver = Solver(None, None, test_data_loader, config)
        if config.single_out:
            solver.test_save_single_img()
        else:
            solver.test(0)



def str2bool(v):
    return v.lower() in ('true')

def merge_yaml_args(parser,filename):
    with open(filename, 'r') as f:
        yaml_cfg = yaml.load(f)
    for key_name in yaml_cfg.keys():
        if not key_name in exit_args:
            if isinstance(yaml_cfg[key_name], bool):
                parser.add_argument('--' + str(key_name), type=str2bool, default=yaml_cfg[key_name])
            else:
                parser.add_argument('--' + str(key_name), type=type(yaml_cfg[key_name]), default=yaml_cfg[key_name])
            exit_args.append(key_name)
    return parser

global exit_args
if __name__ == '__main__':

    exit_args = ['batch_size', 'save_dir', 'mode', 'pretrained_model', 'test_model', 'test_model']
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='tmp')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--pretrained_model', type=str, default="")
    parser.add_argument('--test_model', type=str, default="11-11")
    #parser = merge_yaml_args(parser, 'config/config_pose_VDL.yml')
    #parser = merge_yaml_args(parser, 'config/config_pose_SDL.yml')
    #parser = merge_yaml_args(parser, 'config/config_pose.yml')
    #parser = merge_yaml_args(parser, 'config/config_pose_TW1.yml')
    parser = merge_yaml_args(parser, 'config/config_pose_TW1.yml')
    config = parser.parse_args()
    config.model_save_path = os.path.join(config.save_dir, "models")
    config.sample_path = os.path.join(config.save_dir, "samples")
    config.result_path = os.path.join(config.save_dir, "results")
    config.logs_path = os.path.join(config.save_dir, "logs")
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)
    if not os.path.exists(config.sample_path):
        os.makedirs(config.sample_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    if not os.path.exists(config.logs_path):
        os.makedirs(config.logs_path)

    if config.mode == 'train':
        write_yaml = {}
        info = vars(config)
        for a in info:
            write_yaml[a] = info[a]
        with open(os.path.join(config.save_dir, 'data.yml'), 'w') as outfile:
            yaml.dump(write_yaml, outfile, default_flow_style=False)
    print(config)
    main(config)
