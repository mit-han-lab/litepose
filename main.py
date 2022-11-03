import argparse
import json
import yaml
import uuid
import shlex, subprocess
import torch
import importlib
import _init_paths
import os
from distutils.dir_util import copy_tree
from config import cfg



MODEL_TYPE_LIST = [
    "litepose_xs",
    "litepose_s",
    "litepose_m",
    "litepose_l",
    "higherhrnet_w32"
]

MODEL_TYPE_CFG = {
    "litepose_xs": {
        "cfg" : "experiments/crowd_pose/mobilenet/mobile.yaml",
        "weight" : "weights/LitePose-Auto-XS.pth.tar",
        "supercfg" : "mobile_configs/search-XS.json" 
    },
    "litepose_s": {
        "cfg" : "experiments/crowd_pose/mobilenet/mobile.yaml",
        "weight" : "weights/LitePose-Auto-S.pth.tar",
        "supercfg" : "mobile_configs/search-S.json" 
    },
    "litepose_m": {
        "cfg" : "experiments/crowd_pose/mobilenet/mobile.yaml",
        "weight" : "weights/LitePose-Auto-M.pth.tar",
        "supercfg" : "mobile_configs/search-M.json" 
    },
    "litepose_l": {
        "cfg" : "experiments/crowd_pose/mobilenet/mobile.yaml",
        "weight" : "weights/LitePose-Auto-L.pth.tar",
        "supercfg" : "mobile_configs/search-L.json" 
    },
    "higherhrnet_w32": {
        "cfg" : "w32_512_adam_lr1e-3.yaml",
        "weight" : "hrnet_w32-36af842e.pth",
        "supercfg" : None
    },
}



def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--json',
                        help='experiment configure file name',
                        required=True,
                        type=str)


    args = parser.parse_args()

    return args


def json_validation_check(json_dict):
    key_check = [
        "model_type" in json_dict, 
        "dataset_path" in json_dict,
        "num_joint" in json_dict,
        "dataset_format" in json_dict
    ]
    key_validation = all(key_check)
    
    if json_dict["model_type"] in MODEL_TYPE_LIST:
        model_type_validation = True
    else:
        model_type_validation = False
    
    if type(json_dict["num_joint"]) == int and json_dict["num_joint"] < 30:
        num_joint_validation = True
    else:
        num_joint_validation = False
    
    if json_dict["dataset_format"] in ["coco", "crowd_pose"]:
        dataset_format_validation = True
    else:
        dataset_format_validation = False
    print(key_validation)
    print(model_type_validation)
    print(num_joint_validation)
    print(dataset_format_validation)

    return all([key_validation, model_type_validation, num_joint_validation, dataset_format_validation])


def read_json_file(json_file_path):
    with open(json_file_path, 'r') as f_in:
        json_str = f_in.read()
    
    json_dict = json.loads(json_str)
    if not json_validation_check(json_dict) :
        raise Exception("json is not valid")
    else:
        return json_dict


def update_dict(cfg_dict, to_update):
    for first_key in to_update:
        if type(to_update[first_key]) == dict:
            for second_key in to_update[first_key]:
                cfg_dict[first_key][second_key] = to_update[first_key][second_key]
        else:
            cfg_dict[first_key] = to_update[first_key]

    return cfg_dict
    



def make_yaml_file(json_dict, task_id):
    # basic settings
    model_type = json_dict["model_type"]
    num_joint = json_dict["num_joint"]
    dataset_path = json_dict["dataset_path"]
    dataset_format = json_dict["dataset_format"]
    json_cfg = json_dict["cfg"]
    basic_cfg = MODEL_TYPE_CFG[model_type]["cfg"]
    weight = MODEL_TYPE_CFG[model_type]["weight"]
    supercfg = MODEL_TYPE_CFG[model_type]["supercfg"]

    # load yaml file
    with open(basic_cfg) as load_yaml:
        cfg_dict = yaml.load(load_yaml, Loader=yaml.FullLoader)
    
    cfg_dict["MULTIPROCESSING_DISTRIBUTED"] = False
    cfg_dict["DATASET"]["ROOT"] = dataset_path
    cfg_dict["OUTPUT_DIR"] = f"{task_id}/output"
    cfg_dict["LOG_DIR"] = f"{json_dict['output_path']}/{task_id}/log"
    cfg_dict["DATASET"]["DATASET"] = f"{dataset_format}_kpt"
    cfg_dict["DATASET"]["DATASET_TEST"] = dataset_format
    cfg_dict["MODEL"]["PRETRAINED"] = weight
    cfg_dict["MODEL"]["NUM_JOINTS"] = num_joint

    os.makedirs(cfg_dict["OUTPUT_DIR"], exist_ok=True)
    # override other settings to cfg
    cfg_dict = update_dict(cfg_dict, json_cfg)
    
    # write
    cfg_file_path = f"{task_id}/cfg.yaml"
    with open(cfg_file_path, 'w') as f:
        yaml.dump(cfg_dict, f)

    return cfg_file_path, supercfg


def make_onnx(cfg_file_path, supercfg, model_best_path, task_id, input_size, output_path):
    # onnx 
    f = open(supercfg, 'r' )
    cfg_arch = json.loads(f.read())
    cfg.defrost()
    cfg.merge_from_file(cfg_file_path)

    with open(cfg_file_path, 'r') as f:
        test_cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
    print(task_id)
    task_id = str(task_id)
    #import_task_id = importlib.import_module(task_id)
    import_str = f"from {task_id}.output.{test_cfg_dict['DATASET']['DATASET']}.{test_cfg_dict['MODEL']['NAME']}.cfg.{test_cfg_dict['MODEL']['NAME']} import get_pose_net"
    print(import_str)
    exec(import_str, globals())
    model = get_pose_net(cfg, is_train=True, cfg_arch = cfg_arch)
    model.eval()

    weight = torch.load(model_best_path)
    model.load_state_dict(weight)
    torch.onnx.export(model, torch.randn(*input_size), output_path)


def main(valid=False):
    # task id
    task_id = str(uuid.uuid4())
    task_id = "task_id_" + task_id.replace('-', '_')
    print(f"task_id : {task_id}")
    args = parse_args()
    json_file_path = args.json
    json_dict = read_json_file(json_file_path)

    # make yaml file
    cfg_file_path, supercfg = make_yaml_file(json_dict, task_id)

    # run training
    command_line = f"python dist_train.py --cfg {cfg_file_path} " 
    if supercfg:
        command_line += f"--superconfig {supercfg}"

    print(command_line)
    args = shlex.split(command_line)
    train_p = subprocess.Popen(args)
    train_p.wait()

    # validation
    
    with open(cfg_file_path, 'r') as f:
        test_cfg_dict = yaml.load(f, Loader=yaml.FullLoader)

    model_best_path = f"{test_cfg_dict['OUTPUT_DIR']}/{test_cfg_dict['DATASET']['DATASET']}/{test_cfg_dict['MODEL']['NAME']}/cfg/model_best.pth.tar"

    if valid:    
        command_line = f"python valid.py --cfg {cfg_file_path} "
        if supercfg:
            command_line += f"--superconfig {supercfg} "
        command_line += f"TEST.MODEL_FILE {model_best_path} "

        print(command_line)
        args = shlex.split(command_line)
        test_p = subprocess.Popen(args)
        test_p.wait()

    make_onnx(cfg_file_path, supercfg, model_best_path, task_id, [1,3,256,256], f"{test_cfg_dict['OUTPUT_DIR']}/{task_id}.onnx")
    output_path = f"{json_dict['output_path']}/{task_id}/output"
    copy_tree(test_cfg_dict["OUTPUT_DIR"], output_path)
    print(f"outputs are saved at {output_path}")


if __name__ == "__main__":
    main(valid=True)