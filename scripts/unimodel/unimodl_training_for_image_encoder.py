import os
import os

import h5py
import hydra
import torch
from omegaconf import DictConfig

from bioscanclip.model.simple_clip import load_vit_for_simclr_training
from bioscanclip.util.dataset import DatasetForSimCLRStyleTraining
from bioscanclip.util.simclr import SimCLR
from bioscanclip.util.util import set_seed


def print_when_rank_zero(message, rank=0):
    if rank is None or rank == 0:
        print(message)


def main_process(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(args.model_config, "dataset") and args.model_config.dataset == "bioscan_5m":
        hdf5_inputs_path = (args.bioscan_5m_data.path_to_smaller_hdf5_data
                            if hasattr(args.model_config,
                                       "train_with_small_subset") and args.model_config.train_with_small_subset
                            else args.bioscan_5m_data.path_to_hdf5_data)

    else:
        hdf5_inputs_path = args.bioscan_data.path_to_hdf5_data
    split = "no_split_and_seen_train"
    length_of_data = len(h5py.File(hdf5_inputs_path, "r", libver="latest")[split]['image'])

    train_dataset = DatasetForSimCLRStyleTraining(args, split, length=length_of_data, transform=None)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.model_config.batch_size, shuffle=True,
                                               num_workers=8, pin_memory=False, drop_last=True)

    model = load_vit_for_simclr_training(args, device=None)

    optimizer = torch.optim.Adam(model.parameters(), args.model_config.lr_config.lr,
                                 weight_decay=args.model_config.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)
    with torch.cuda.device(device):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, device=device, args=args)
        simclr.train(train_loader)


@hydra.main(config_path="../../bioscanclip/config", config_name="global_config", version_base="1.1")
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

    main_process(args)


if __name__ == '__main__':
    main()
