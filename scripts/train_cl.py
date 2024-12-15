import copy
import datetime
import json
import os
import hydra
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from omegaconf import DictConfig, OmegaConf, open_dict

from bioscanclip.epoch.train_epoch import train_epoch
from inference_and_eval import get_features_and_label, inference_and_print_result
from bioscanclip.model.loss_func import ContrastiveLoss, ClipLoss
from bioscanclip.model.simple_clip import load_clip_model
from bioscanclip.util.util import set_seed
from bioscanclip.util.dataset import load_dataloader, load_insect_dataloader
from bioscanclip.util.util import scale_learning_rate
from bioscanclip.model.clibd_lightning import CLIBDLightning
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from bioscanclip.model.clibd_lightning import MyProgressBar


def print_when_rank_zero(message, rank=0):
    if rank is None or rank == 0:
        print(message)


def save_prediction(pred_list, gt_list, json_path):
    data = {
        "gt_labels": gt_list,
        "pred_labels": pred_list
    }

    with open(json_path, 'w') as json_file:
        json.dump(data, json_file)


def ddp_setup(rank: int, world_size: int, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def construct_key_dict(list_of_dict):
    key_dict = {}

    for curr_dict in list_of_dict:
        for a_kind_of_feature_or_label in curr_dict.keys():
            if a_kind_of_feature_or_label == "all_key_features" or a_kind_of_feature_or_label == "all_key_features_label":
                key_dict[a_kind_of_feature_or_label] = None
                continue

            if a_kind_of_feature_or_label not in key_dict.keys():
                key_dict[a_kind_of_feature_or_label] = curr_dict[a_kind_of_feature_or_label]
            else:
                if isinstance(curr_dict[a_kind_of_feature_or_label], list):
                    key_dict[a_kind_of_feature_or_label] = key_dict[a_kind_of_feature_or_label] + curr_dict[
                        a_kind_of_feature_or_label]
                else:
                    key_dict[a_kind_of_feature_or_label] = np.concatenate(
                        (key_dict[a_kind_of_feature_or_label], curr_dict[a_kind_of_feature_or_label]), axis=0)

    return key_dict


def main_process(args):
    if args.debug_flag:
        args.activate_wandb = False
        args.save_inference = False
        args.save_ckpt = False

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H%M%S")
    args = copy.deepcopy(args)

    with open_dict(args.model_config):
        if not hasattr(args.model_config, "for_open_clip"):
            args.model_config.for_open_clip = False

    # Load DATALOADER
    print("Construct dataloader...")
    # if hasattr(args.model_config, 'dataset') and args.model_config.dataset == "INSECT":
    #     insect_train_dataloader, insect_train_dataloader_for_key, insect_val_dataloader, insect_test_seen_dataloader, insect_test_unseen_dataloader = load_insect_dataloader(
    #         args)
    #     pre_train_dataloader = insect_train_dataloader
    # else:
    #     pre_train_dataloader, seen_val_dataloader, unseen_val_dataloader, all_keys_dataloader = load_dataloader(args)

    # pre_train_dataloader, seen_val_dataloader, unseen_val_dataloader, all_keys_dataloader = load_dataloader(args)

    # Debug with smaller dataset
    pre_train_dataloader, seen_val_dataloader, unseen_val_dataloader, all_keys_dataloader = load_dataloader(args,
                                                                                                            for_pretrain=False)

    # optional configs
    for_open_clip = False
    if hasattr(args.model_config, 'for_open_clip') and args.model_config.for_open_clip:
        for_open_clip = True

    all_gather = False
    if hasattr(args.model_config, 'all_gather') and args.model_config.all_gather:
        all_gather = True

    fix_temperature = None
    if hasattr(args.model_config, 'fix_temperature') and args.model_config.fix_temperature:
        fix_temperature = args.model_config.fix_temperature

    enable_amp = False
    if hasattr(args.model_config, 'amp') and args.model_config.amp:
        enable_amp = True

    eval_skip_epoch = -1
    if hasattr(args.model_config, 'eval_skip_epoch') and args.model_config.eval_skip_epoch:
        eval_skip_epoch = args.model_config.eval_skip_epoch

    scaler = GradScaler(enabled=enable_amp)

    # Load MODEL

    print("Initialize model...")

    wandb_logger = None
    if args.activate_wandb:
        wandb_logger = WandbLogger(project=args.model_config.wandb_project_name,
                                   name=args.model_config.model_output_name)
        wandb_logger.log_hyperparams(args)
    k_list = [1, 3, 5]
    model = CLIBDLightning(args, len_train_dataloader=len(pre_train_dataloader),
                           all_keys_dataloader=all_keys_dataloader,
                           seen_val_dataloader=seen_val_dataloader, unseen_val_dataloader=unseen_val_dataloader,
                           k_list=k_list)

    if hasattr(args.model_config, 'pretrained_ckpt_path'):
        model.load_from_checkpoint_with_path(args.model_config.pretrained_ckpt_path)

    # progress_bar = MyProgressBar()

    trainer = hydra.utils.instantiate(args.model_config.trainer, logger=wandb_logger)

    folder_path = os.path.join(args.project_root_path, args.model_output_dir,
                               args.model_config.model_output_name, formatted_datetime)
    os.makedirs(folder_path, exist_ok=True)

    OmegaConf.save(args, os.path.join(folder_path, 'config.yaml'))

    print("training...")
    trainer.fit(model=model, train_dataloaders=pre_train_dataloader, ckpt_path=None)



@hydra.main(config_path="../bioscanclip/config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    torch.cuda.empty_cache()
    args.project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    world_size = torch.cuda.device_count()
    print(f'world_size： {world_size}')

    default_seed = args.default_seed
    if hasattr(args.model_config, 'default_seed'):
        default_seed = args.model_config.default_seed

    if hasattr(args.model_config, 'random_seed') and args.model_config.random_seed:
        seed = set_seed();
        string = "random seed"
    else:
        seed = set_seed(seed=int(default_seed));
        string = "default seed"
    print("The module is run with %s: %d" % (string, seed))

    main_process(args=args)


if __name__ == '__main__':
    main()
