import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
import h5py

from bioscanclip.model.simple_clip import load_clip_model
from bioscanclip.util.dataset import load_dataloader_for_everything_in_5m
from bioscanclip.util.util import get_features_and_label

PLOT_FOLDER = "html_plots"
RETRIEVAL_FOLDER = "image_retrieval"

LEVELS = ["order", "family", "genus", "species"]


def convert_labels_to_four_list(list_of_dict):
    order_list = []
    family_list = []
    genus_list = []
    species_list = []
    for a_dict in list_of_dict:
        order_list.append(a_dict["order"])
        family_list.append(a_dict["family"])
        genus_list.append(a_dict["genus"])
        species_list.append(a_dict["species"])
    return order_list, family_list, genus_list, species_list

@hydra.main(config_path="../../bioscanclip/config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    args.project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    args.save_inference = True
    if os.path.exists(os.path.join(args.model_config.ckpt_path, "best.pth")):
        args.model_config.ckpt_path = os.path.join(args.model_config.ckpt_path, "best.pth")
    elif os.path.exists(os.path.join(args.model_config.ckpt_path, "last.pth")):
        args.model_config.ckpt_path = os.path.join(args.model_config.ckpt_path, "last.pth")

    folder_for_saving = os.path.join( args.project_root_path,
        "new_extracted_embedding", args.model_config.dataset, args.model_config.model_output_name
    )
    os.makedirs(folder_for_saving, exist_ok=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize model
    print("Initialize model...")

    model = load_clip_model(args, device)
    if hasattr(args.model_config, "load_ckpt") and args.model_config.load_ckpt is False:
        pass
    else:
        checkpoint = torch.load(args.model_config.ckpt_path, map_location="cuda:0")
        model.load_state_dict(checkpoint)
    # Load data
    args.model_config.batch_size = 24
    pre_train_dataloader, seen_val_dataloader, unseen_val_dataloader, seen_test_dataloader, unseen_test_dataloader, all_keys_dataloader, other_heldout_dataloader = load_dataloader_for_everything_in_5m(args)
    if hasattr(args.model_config, "dataset") and args.model_config.dataset == "bioscan_5m":
        dataloaders_that_need_to_be_process = [pre_train_dataloader, seen_val_dataloader, unseen_val_dataloader, seen_test_dataloader, unseen_test_dataloader, all_keys_dataloader, other_heldout_dataloader]

    extracted_features_path = os.path.join(folder_for_saving,
                                           f"extracted_features_for_all_5m_data.hdf5")

    feature_length = args.model_config.output_dim

    with h5py.File(extracted_features_path, "a") as hdf5_file:
        for index, dataloader in enumerate(dataloaders_that_need_to_be_process):
            embedding_dict = get_features_and_label(dataloader, model, device)
            labels = convert_labels_to_four_list(embedding_dict["label_list"])
            label_names = ["order_list", "family_list", "genus_list", "species_list"]
            encoded_names = ["encoded_image_feature", "encoded_dna_feature", "encoded_language_feature"]

            for label_name, label_data in zip(label_names, labels):
                label_data = np.array([s.encode('utf-8') for s in label_data])
                if label_name in hdf5_file:
                    original_len = hdf5_file[label_name].shape[0]
                    hdf5_file[label_name].resize((original_len + len(label_data),))
                    hdf5_file[label_name][original_len:] = label_data
                else:
                    hdf5_file.create_dataset(label_name, data=label_data, maxshape=(None,))

            for encoded_name in encoded_names:
                feature_data = np.array(embedding_dict[encoded_name])
                if encoded_name in hdf5_file:
                    original_len = hdf5_file[encoded_name].shape[0]
                    hdf5_file[encoded_name].resize((original_len + feature_data.shape[0], feature_length))
                    hdf5_file[encoded_name][original_len:] = feature_data
                else:
                    hdf5_file.create_dataset(encoded_name, data=feature_data, maxshape=(None, feature_length))

    print("Done!")
    print(f"Saved to {extracted_features_path}")

if __name__ == "__main__":
    main()
