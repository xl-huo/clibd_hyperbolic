import copy
import datetime
import json
import os

import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.loggers import WandbLogger
from torch.cuda.amp import GradScaler

from bioscanclip.model.clibd_lightning import CLIBDLightning
from bioscanclip.util.dataset import load_dataloader
from bioscanclip.util.util import set_seed


def print_when_rank_zero(message, rank=0):
    if rank is None or rank == 0:
        print(message)



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

    pre_train_dataloader, seen_val_dataloader, unseen_val_dataloader, all_keys_dataloader = load_dataloader(args)

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
