"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    This script generates a json file for the KITTI Depth Completion dataset.
"""

import os
import argparse
import random
import json
from random import choice

parser = argparse.ArgumentParser(
    description="KITTI Depth Completion jason generator")

parser.add_argument('--path_root', type=str, required=True,
                    help="Path to the KITTI Depth Completion dataset")

parser.add_argument('--path_out', type=str, required=False,
                    default='../data_json', help="Output path")
parser.add_argument('--name_out', type=str, required=False,
                    default='kitti_dc.json', help="Output file name")
parser.add_argument('--num_train', type=int, required=False,
                    default=int(1e10), help="Maximum number of train data")
parser.add_argument('--num_val', type=int, required=False,
                    default=int(1e10), help="Maximum number of val data")
parser.add_argument('--num_test', type=int, required=False,
                    default=int(1e10), help="Maximum number of test data")
parser.add_argument('--seed', type=int, required=False,
                    default=7240, help='Random seed')
parser.add_argument('--test_data', action='store_true',
                    default=False, help='json for DC test set generation')

args = parser.parse_args()

random.seed(args.seed)


# Some miscellaneous functions
def check_dir_existence(path_dir):
    assert os.path.isdir(path_dir), \
        "Directory does not exist : {}".format(path_dir)

def rgb_nearby(name):
    max_frame_diff = 3
    candidates = [
        i - max_frame_diff for i in range(max_frame_diff * 2 + 1)
        if i - max_frame_diff != 0
    ]

    random_offset = choice(candidates)
    name_near = '%010d.png' % (int(name.split(".")[0]) + random_offset)

    return name_near

def rgb_nearby_test(name):
    candidates = [-3, 3]
    random_offset = choice(candidates)
    name_near = '%010d.png' % (int(name.split(".")[0]) + random_offset)

    return name_near


def generate_json():
    os.makedirs(args.path_out, exist_ok=True)
    check_dir_existence(args.path_out)

    # For train/val splits
    dict_json = {}
    for split in ['train', 'val']:
        path_base = args.path_root + '/' + 'data_depth_annotated/' + split

        list_seq = os.listdir(path_base)
        list_seq = [seq for seq in list_seq if seq.endswith('_sync')]
        list_seq.sort()

        list_pairs = []
        for seq in list_seq:
            cnt_seq = 0
            # e.g., 2011_09_26
            seq_data = seq[:10]

            for cam in ['image_02', 'image_03']:
                list_depth = os.listdir(
                    path_base + '/' + seq
                    + '/proj_depth/groundtruth/{}'.format(cam))
                list_depth = [dp for dp in list_depth if dp.endswith('.png')]
                list_depth.sort()

                for name in list_depth:

                    path_rgb = 'rawdata/' +  seq_data + '/' + seq + '/' + cam + '/data/' + name
                    path_depth ='data_depth_velodyne/' +  split + '/' + seq \
                                 + '/proj_depth/velodyne_raw/' + cam + '/' \
                                 + name
                    path_gt ='data_depth_annotated/' +  split + '/' + seq + '/proj_depth/groundtruth/' \
                              + cam + '/' + name
                    path_rgb_near = 'rawdata/' +  seq_data + '/' + seq + '/' + cam + '/data/' + rgb_nearby(name)
                    path_calib = 'rawdata/' + seq_data + '/calib_cam_to_cam.txt'

                    dict_sample = {
                        'rgb': path_rgb,
                        'depth': path_depth,
                        'gt': path_gt,
                        'rgb_near': path_rgb_near,
                        'K': path_calib
                    }

                    flag_valid = True
                    for val in dict_sample.values():
                        flag_valid &= os.path.exists(args.path_root + '/' + val)
                        if not flag_valid:
                            # print(val)
                            break

                    if not flag_valid:
                        continue

                    list_pairs.append(dict_sample)
                    cnt_seq += 1



            print("{} : {} samples".format(seq, cnt_seq))

        dict_json[split] = list_pairs
        print("{} split : Total {} samples".format(split, len(list_pairs)))

    # For test split
    split = 'test'
    path_base = args.path_root + '/data_depth_selection/val_selection_cropped'

    list_depth = os.listdir(path_base + '/velodyne_raw')
    list_depth = [dp for dp in list_depth if dp.endswith('.png')]
    list_depth.sort()

    list_pairs = []
    s = 0
    for name in list_depth:
        name_base = name.split('velodyne_raw')
        base_near = name_base[1].split('_')
        base_near[1] = rgb_nearby_test(base_near[1]).split('.')[0]
        if base_near[-1][1] == '2':
            base_near[-1] = base_near[-1].replace('2', '3')
        else:
            base_near[-1] = base_near[-1].replace('3', '2')
        base_name = "_".join(base_near)
        name_base_near = name_base[0] + '{}' + base_name

        name_base = name_base[0] + '{}' + name_base[1]

        path_rgb = 'data_depth_selection/val_selection_cropped/image/' \
                   + name_base.format('image')
        path_depth = 'data_depth_selection/val_selection_cropped/velodyne_raw/' \
                     + name
        path_gt = 'data_depth_selection/val_selection_cropped/groundtruth_depth/' \
                  + name_base.format('groundtruth_depth')
        path_rgb_near = 'data_depth_selection/val_selection_cropped/image/' \
                        + name_base_near.format('image')
        path_calib = 'data_depth_selection/val_selection_cropped/intrinsics/' \
                     + name_base.format('image')[:-4] + '.txt'


        dict_sample = {
            'rgb': path_rgb,
            'depth': path_depth,
            'gt': path_gt,
            'rgb_near' : path_rgb_near,
            'K': path_calib
        }

        flag_valid = True

        for val in dict_sample.values():
            flag_valid &= os.path.exists(args.path_root + '/' + val)
            if not flag_valid:
                s += int(flag_valid)+1
                print(val)
                break

        if not flag_valid:
            continue

        list_pairs.append(dict_sample)
    print(s)

    dict_json[split] = list_pairs
    print("{} split : Total {} samples".format(split, len(list_pairs)))

    random.shuffle(dict_json['train'])

    # Cut if maximum is set
    for s in [('train', args.num_train), ('val', args.num_val),
              ('test', args.num_test)]:
        if len(dict_json[s[0]]) > s[1]:
            # Do shuffle
            random.shuffle(dict_json[s[0]])

            num_orig = len(dict_json[s[0]])
            dict_json[s[0]] = dict_json[s[0]][0:s[1]]
            print("{} split : {} -> {}".format(s[0], num_orig,
                                               len(dict_json[s[0]])))

    f = open(args.path_out + '/' + args.name_out, 'w')
    json.dump(dict_json, f, indent=4)
    f.close()

    print("Json file generation finished.")


def generate_json_test():
    check_dir_existence(args.path_out)

    dict_json = {}

    # For test split
    split = 'test'
    path_base = args.path_root \
                + '/data_depth_selection/test_depth_completion_anonymous'

    list_depth = os.listdir(path_base + '/velodyne_raw')
    list_depth.sort()

    list_pairs = []
    for name in list_depth:
        path_rgb = 'data_depth_selection/test_depth_completion_anonymous/image/' \
                   + name
        path_depth = \
            'data_depth_selection/test_depth_completion_anonymous/velodyne_raw/' \
            + name
        path_gt = path_depth
        path_calib = \
            'data_depth_selection/test_depth_completion_anonymous/intrinsics/' \
            + name[:-4] + '.txt'

        dict_sample = {
            'rgb': path_rgb,
            'depth': path_depth,
            'gt': path_gt,
            'K': path_calib
        }

        flag_valid = True
        for val in dict_sample.values():
            flag_valid &= os.path.exists(args.path_root + '/' + val)
            if not flag_valid:
                break

        if not flag_valid:
            continue

        list_pairs.append(dict_sample)

    dict_json[split] = list_pairs
    print("{} split : Total {} samples".format(split, len(list_pairs)))

    f = open(args.path_out + '/' + args.name_out, 'w')
    json.dump(dict_json, f, indent=4)
    f.close()

    print("Json file generation finished.")


if __name__ == '__main__':
    print('\nArguments :')
    for arg in vars(args):
        print(arg, ':', getattr(args, arg))
    print('')

    if args.test_data:
        generate_json_test()
    else:
        generate_json()