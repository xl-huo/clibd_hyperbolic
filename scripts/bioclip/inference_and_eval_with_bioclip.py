import json
import json
import os
import contextlib
import h5py
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from bioscanclip.util.dataset import load_dataloader, load_bioscan_dataloader_with_train_seen_and_separate_keys, \
    load_bioscan_dataloader_all_small_splits
import open_clip
import torch.nn.functional as F
from collections import defaultdict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



openai_templates = [
    lambda c: f"a bad photo of a {c}.",
    lambda c: f"a photo of many {c}.",
    lambda c: f"a sculpture of a {c}.",
    lambda c: f"a photo of the hard to see {c}.",
    lambda c: f"a low resolution photo of the {c}.",
    lambda c: f"a rendering of a {c}.",
    lambda c: f"graffiti of a {c}.",
    lambda c: f"a bad photo of the {c}.",
    lambda c: f"a cropped photo of the {c}.",
    lambda c: f"a tattoo of a {c}.",
    lambda c: f"the embroidered {c}.",
    lambda c: f"a photo of a hard to see {c}.",
    lambda c: f"a bright photo of a {c}.",
    lambda c: f"a photo of a clean {c}.",
    lambda c: f"a photo of a dirty {c}.",
    lambda c: f"a dark photo of the {c}.",
    lambda c: f"a drawing of a {c}.",
    lambda c: f"a photo of my {c}.",
    lambda c: f"the plastic {c}.",
    lambda c: f"a photo of the cool {c}.",
    lambda c: f"a close-up photo of a {c}.",
    lambda c: f"a black and white photo of the {c}.",
    lambda c: f"a painting of the {c}.",
    lambda c: f"a painting of a {c}.",
    lambda c: f"a pixelated photo of the {c}.",
    lambda c: f"a sculpture of the {c}.",
    lambda c: f"a bright photo of the {c}.",
    lambda c: f"a cropped photo of a {c}.",
    lambda c: f"a plastic {c}.",
    lambda c: f"a photo of the dirty {c}.",
    lambda c: f"a jpeg corrupted photo of a {c}.",
    lambda c: f"a blurry photo of the {c}.",
    lambda c: f"a photo of the {c}.",
    lambda c: f"a good photo of the {c}.",
    lambda c: f"a rendering of the {c}.",
    lambda c: f"a {c} in a video game.",
    lambda c: f"a photo of one {c}.",
    lambda c: f"a doodle of a {c}.",
    lambda c: f"a close-up photo of the {c}.",
    lambda c: f"a photo of a {c}.",
    lambda c: f"the origami {c}.",
    lambda c: f"the {c} in a video game.",
    lambda c: f"a sketch of a {c}.",
    lambda c: f"a doodle of the {c}.",
    lambda c: f"a origami {c}.",
    lambda c: f"a low resolution photo of a {c}.",
    lambda c: f"the toy {c}.",
    lambda c: f"a rendition of the {c}.",
    lambda c: f"a photo of the clean {c}.",
    lambda c: f"a photo of a large {c}.",
    lambda c: f"a rendition of a {c}.",
    lambda c: f"a photo of a nice {c}.",
    lambda c: f"a photo of a weird {c}.",
    lambda c: f"a blurry photo of a {c}.",
    lambda c: f"a cartoon {c}.",
    lambda c: f"art of a {c}.",
    lambda c: f"a sketch of the {c}.",
    lambda c: f"a embroidered {c}.",
    lambda c: f"a pixelated photo of a {c}.",
    lambda c: f"itap of the {c}.",
    lambda c: f"a jpeg corrupted photo of the {c}.",
    lambda c: f"a good photo of a {c}.",
    lambda c: f"a plushie {c}.",
    lambda c: f"a photo of the nice {c}.",
    lambda c: f"a photo of the small {c}.",
    lambda c: f"a photo of the weird {c}.",
    lambda c: f"the cartoon {c}.",
    lambda c: f"art of the {c}.",
    lambda c: f"a drawing of the {c}.",
    lambda c: f"a photo of the large {c}.",
    lambda c: f"a black and white photo of a {c}.",
    lambda c: f"the plushie {c}.",
    lambda c: f"a dark photo of a {c}.",
    lambda c: f"itap of a {c}.",
    lambda c: f"graffiti of the {c}.",
    lambda c: f"a toy {c}.",
    lambda c: f"itap of my {c}.",
    lambda c: f"a photo of a cool {c}.",
    lambda c: f"a photo of a small {c}.",
    lambda c: f"a tattoo of the {c}.",
]

def make_image_key_features(model, all_keys_dataloader):
    key_labels = {}
    with torch.no_grad():
        key_features = []
        autocast = get_autocast("amp")
        pbar = tqdm(all_keys_dataloader)
        for batch in pbar:
            pbar.set_description("Encode key feature...")
            file_name_batch, image_input_batch, dna_batch, input_ids, token_type_ids, attention_mask, label_batch = batch
            image_input_batch = image_input_batch.to(DEVICE)
            with autocast():
                image_features = model.encode_image(image_input_batch)
                image_features = F.normalize(image_features, dim=-1)
            image_features = image_features.to("cpu")
            for image_feature in image_features:
                # print(image_feature.shape)
                # exit()
                key_features.append(image_feature)
            for key in label_batch.keys():
                if key not in key_labels:
                    key_labels[key] = []
                key_labels[key] = key_labels[key] + label_batch[key]

        key_features = torch.stack(key_features, dim=1).to(DEVICE)
    return key_features, key_labels

def make_txt_features(model, classnames, templates):
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
    with torch.no_grad():
        txt_features = []
        for classname in tqdm(classnames):
            classname = " ".join(word for word in classname.split("_") if word)
            texts = [template(classname) for template in templates]  # format with class
            texts = tokenizer(texts).to(DEVICE)  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            txt_features.append(class_embedding)
        txt_features = torch.stack(txt_features, dim=1).to(DEVICE)
    return txt_features

def get_all_unique_species_from_dataloader(dataloader):
    all_species = []
    species_to_other = {}
    for batch in dataloader:
        file_name_batch, image_input_batch, dna_batch, input_ids, token_type_ids, attention_mask, label_batch = batch
        all_species = all_species + label_batch['species']

        for idx, species in enumerate(label_batch['species']):
            if species not in species_to_other.keys():
                species_to_other[species] = {"order": label_batch['order'][idx], "family": label_batch['family'][idx],
                                             "genus": label_batch['genus'][idx], 'species': species}

    all_species = list(set(all_species))
    all_species.sort()

    all_labels_in_dict = []

    for species in all_species:
        all_labels_in_dict.append(species_to_other[species])

    return all_species, all_labels_in_dict


def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return contextlib.suppress

def make_prediction(logits, key_labels, topk=(1,)):
    pred_index = logits.topk(max(topk), dim=1).indices
    prediction = [key_labels[label] for label in pred_index]
    return prediction

def calculate_macro_accuracy(all_pred_labels, all_gt_labels):

    levels = ['order', 'family', 'genus', 'species']
    correct_counts = {level: defaultdict(int) for level in levels}
    total_counts = {level: defaultdict(int) for level in levels}

    for pred, gt in zip(all_pred_labels, all_gt_labels):
        for level in levels:
            if pred[level] == gt[level]:
                correct_counts[level][gt[level]] += 1
            total_counts[level][gt[level]] += 1

    macro_accuracies = {}
    for level in levels:
        level_accuracies = [correct_counts[level][cls] / total_counts[level][cls]
                            for cls in total_counts[level]]
        macro_accuracies[level] = sum(level_accuracies) / len(level_accuracies) if level_accuracies else 0

    return macro_accuracies

def process_all_gt_labels(all_gt_labels):
    all_gt_labels_in_list = []
    total_len = len(all_gt_labels['species'])
    for idx in range(total_len):
        curr_labels = {}
        for key in all_gt_labels.keys():
            curr_labels[key] = all_gt_labels[key][idx]
        all_gt_labels_in_list.append(curr_labels)
    return all_gt_labels_in_list


def encode_image_features_and_make_prediction(model, query_dataloader, key_features, key_labels):
    autocast = get_autocast("amp")

    pbar = tqdm(query_dataloader, desc="Encoding image features...")
    all_pred_labels = []
    all_gt_labels = {}
    for batch in pbar:
        file_name_batch, image_input_batch, dna_batch, input_ids, token_type_ids, attention_mask, label_batch = batch
        for key in label_batch.keys():
            if key not in all_gt_labels:
                all_gt_labels[key] = []
            all_gt_labels[key] = all_gt_labels[key] + label_batch[key]
        image_input_batch = image_input_batch.to(DEVICE)
        with autocast():
            image_features = model.encode_image(image_input_batch)
            image_features = F.normalize(image_features, dim=-1)
            logits = model.logit_scale.exp() * image_features @ key_features

        # based on the logits, make prediction
        pred = make_prediction(logits, key_labels, topk=(1,))
        all_pred_labels = all_pred_labels + pred


    all_gt_labels = process_all_gt_labels(all_gt_labels)
    macro_accuracies = calculate_macro_accuracy(all_pred_labels, all_gt_labels)
    return macro_accuracies

def harmonic_mean(numbers):
    if any(n == 0 for n in numbers):
        raise ValueError("All numbers must be non-zero.")

    num_count = len(numbers)
    inverse_sum = sum(1 / n for n in numbers)

    return num_count / inverse_sum

@hydra.main(config_path="../../bioscanclip/config", config_name="global_config", version_base="1.1")
def main(args: DictConfig) -> None:
    args.save_inference = True

    folder_for_saving = os.path.join(
        args.visualization.output_dir, args.model_config.model_output_name, "features_and_prediction"
    )
    os.makedirs(folder_for_saving, exist_ok=True)

    # initialize model
    print("Initialize model...")
    model, _, _ = open_clip.create_model_and_transforms(
        "hf-hub:imageomics/bioclip"
    )
    model.to(DEVICE)

    # Load data
    args.model_config.batch_size = 24
    _, _, _, seen_test_dataloader, unseen_test_dataloader, seen_keys_dataloader, val_unseen_keys_dataloader, test_unseen_keys_dataloader, all_keys_dataloader = load_bioscan_dataloader_all_small_splits(
        args)
    _, seen_val_dataloader, unseen_val_dataloader, all_keys_dataloader = load_dataloader(args)

    txt_features_path = os.path.join(folder_for_saving, "txt_features.pth")
    labels_dict_path = os.path.join(folder_for_saving, "labels_dict.pth")
    all_species, all_labels_in_dict = get_all_unique_species_from_dataloader(all_keys_dataloader)

    if os.path.exists(txt_features_path) and os.path.exists(labels_dict_path):
        txt_features_of_all_species = torch.load(txt_features_path)
        all_labels_in_dict = torch.load(labels_dict_path)
    else:
        classnames = [name.replace("_", " ") for name in all_species]
        txt_features_of_all_species = make_txt_features(model, classnames, openai_templates)
        torch.save(txt_features_of_all_species, txt_features_path)
        torch.save(all_labels_in_dict, labels_dict_path)

    text_dict = {'features': txt_features_of_all_species, 'labels': all_labels_in_dict}
    seen_macro_accuracies = encode_image_features_and_make_prediction(model, seen_test_dataloader,  text_dict['features'], text_dict['labels'])


    unseen_macro_accuracies = encode_image_features_and_make_prediction(model, unseen_test_dataloader, text_dict['features'], text_dict['labels'])

    print("For image to text: ")
    for level in ['order', 'family', 'genus', 'species']:
        print(f"Level: {level}")
        seen_acc = seen_macro_accuracies[level]
        unseen_acc = unseen_macro_accuracies[level]
        harmoinc_mean_acc = harmonic_mean([seen_acc, unseen_acc])
        print(f"Seen acc：{seen_acc} || Unseen acc：{unseen_acc} || Harmonic mean acc：{harmoinc_mean_acc}")
        print(f"{round(seen_acc*100, 1)} & {round(unseen_acc*100, 1)} & {round(harmoinc_mean_acc*100, 1)}")


if __name__ == "__main__":
    main()
