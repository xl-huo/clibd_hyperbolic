import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
import h5py

from bioscanclip.epoch.inference_epoch import get_feature_and_label
from bioscanclip.model.simple_clip import load_clip_model
from bioscanclip.util.dataset import load_bioscan_dataloader_all_small_splits

PLOT_FOLDER = "html_plots"
RETRIEVAL_FOLDER = "image_retrieval"
All_TYPE_OF_FEATURES_OF_QUERY = [
    "encoded_image_feature",
    "encoded_dna_feature",
    "encoded_language_feature",
    "averaged_feature",
    "concatenated_feature",
]
All_TYPE_OF_FEATURES_OF_KEY = [
    "encoded_image_feature",
    "encoded_dna_feature",
    "encoded_language_feature",
    "averaged_feature",
    "concatenated_feature",
    "all_key_features",
]
LEVELS = ["order", "family", "genus", "species"]


def get_features_and_label(dataloader, model, device, for_key_set=False):
    _, encoded_language_feature, _ = get_feature_and_label(
        dataloader, model, device, type_of_feature="text", multi_gpu=False
    )

    _, encoded_dna_feature, _ = get_feature_and_label(
        dataloader, model, device, type_of_feature="dna", multi_gpu=False
    )

    file_name_list, encoded_image_feature, label_list = get_feature_and_label(
        dataloader, model, device, type_of_feature="image", multi_gpu=False
    )

    averaged_feature = None
    concatenated_feature = None
    all_key_features = None
    all_key_features_label = None
    if encoded_dna_feature is not None and encoded_image_feature is not None:
        averaged_feature = np.mean([encoded_image_feature, encoded_dna_feature], axis=0)
        concatenated_feature = np.concatenate((encoded_image_feature, encoded_dna_feature), axis=1)

    dictionary_of_split = {
        "file_name_list": file_name_list,
        "encoded_dna_feature": encoded_dna_feature,
        "encoded_image_feature": encoded_image_feature,
        "encoded_language_feature": encoded_language_feature,
        "averaged_feature": averaged_feature,
        "concatenated_feature": concatenated_feature,
        "label_list": label_list,
    }

    if (
        for_key_set
        and encoded_image_feature is not None
        and encoded_dna_feature is not None
        and encoded_language_feature is not None
    ):
        for curr_feature in [encoded_image_feature, encoded_dna_feature, encoded_language_feature]:
            if all_key_features is None:
                all_key_features = curr_feature
                all_key_features_label = label_list
            else:
                all_key_features = np.concatenate((all_key_features, curr_feature), axis=0)
                all_key_features_label = all_key_features_label + label_list

    dictionary_of_split["all_key_features"] = all_key_features
    dictionary_of_split["all_key_features_label"] = all_key_features_label

    return dictionary_of_split

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

@hydra.main(config_path="../bioscanclip/config", config_name="global_config", version_base="1.1")
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
    train_seen_dataloader, seen_val_dataloader, unseen_val_dataloader, seen_test_dataloader, unseen_test_dataloader, seen_keys_dataloader, val_unseen_keys_dataloader, test_unseen_keys_dataloader, all_keys_dataloader = load_bioscan_dataloader_all_small_splits(
        args)
    if hasattr(args.model_config, "dataset") and args.model_config.dataset == "bioscan_5m":
        dataloaders_that_need_to_be_process = {"seen_val_dataloader": seen_val_dataloader,
                                               "unseen_val_dataloader": unseen_val_dataloader,
                                               "seen_test_dataloader": seen_test_dataloader,
                                               "unseen_test_dataloader": unseen_test_dataloader,
                                               "seen_keys_dataloader": seen_keys_dataloader,
                                               "unseen_keys_dataloader": val_unseen_keys_dataloader,
                                               "all_keys_dataloader": all_keys_dataloader}
    else:
        dataloaders_that_need_to_be_process = {"train_seen_dataloader": train_seen_dataloader,
                                               "seen_val_dataloader": seen_val_dataloader,
                                               "unseen_val_dataloader": unseen_val_dataloader,
                                               "seen_test_dataloader": seen_test_dataloader,
                                               "unseen_test_dataloader": unseen_test_dataloader,
                                               "seen_keys_dataloader": seen_keys_dataloader,
                                               "val_unseen_keys_dataloader": val_unseen_keys_dataloader,
                                               "test_unseen_keys_dataloader": test_unseen_keys_dataloader,
                                               "all_keys_dataloader": all_keys_dataloader}

    for key, dataloader in dataloaders_that_need_to_be_process.items():
        print(f"Processing {key}...")
        split = key.split('_')[:-1]
        split_name = "_".join(split)
        extracted_features_path = os.path.join(folder_for_saving,
                                               f"extracted_features_of_{split_name}.hdf5")
        labels_path = os.path.join(folder_for_saving, f"labels_{key}.npy")
        if args.save_inference and not (os.path.exists(extracted_features_path) and os.path.exists(labels_path)):

            embedding_dict = get_features_and_label(dataloader, model, device)

            new_file = h5py.File(extracted_features_path, "w")

            order_list, family_list, genus_list, species_list = convert_labels_to_four_list(embedding_dict["label_list"])

            order_list = [s.encode('utf-8') for s in order_list]
            new_file.create_dataset("order_list", data=order_list)

            family_list = [s.encode('utf-8') for s in family_list]
            new_file.create_dataset("family_list", data=family_list)

            genus_list = [s.encode('utf-8') for s in genus_list]
            new_file.create_dataset("genus_list", data=genus_list)

            species_list = [s.encode('utf-8') for s in species_list]
            new_file.create_dataset("species_list", data=species_list)

            new_file.create_dataset("encoded_image_feature", data=embedding_dict["encoded_image_feature"])
            new_file.create_dataset("encoded_dna_feature", data=embedding_dict["encoded_dna_feature"])
            if hasattr(args.model_config, "language"):
                new_file.create_dataset("encoded_language_feature", data=embedding_dict["encoded_language_feature"])

if __name__ == "__main__":
    main()
