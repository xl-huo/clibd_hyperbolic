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
from bioscanclip.util.dataset import load_dataloader, load_bioscan_dataloader_with_train_seen_and_separate_keys, load_bioscan_dataloader_all_small_splits
import open_clip
import torch.nn.functional as F
from collections import defaultdict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_image_key_features(model, all_keys_dataloader):
    all_labels_in_dict = {}
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
            for image_feature in image_features:
                key_features.append(image_feature)
            for key in label_batch.keys():
                if key not in all_labels_in_dict:
                    all_labels_in_dict[key] = []
                all_labels_in_dict[key] = all_labels_in_dict[key] + label_batch[key]

        key_features = torch.stack(key_features, dim=1).to(DEVICE)
    all_labels_in_dict = process_all_gt_labels(all_labels_in_dict)
    return key_features, all_labels_in_dict


def make_prediction(logits, key_labels, topk=(1,)):
    pred_index = logits.topk(max(topk), dim=1).indices
    prediction = [key_labels[label] for label in pred_index]
    return prediction

def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return contextlib.suppress

def compute_accuracy(predictions, ground_truth):
    top_1_correct = 0
    top_3_correct = 0
    top_5_correct = 0
    total_samples = len(ground_truth)
    class_correct_and_total_count = {}
    for i, truth in enumerate(ground_truth):
        # Top-1 accuracy check
        if truth not in class_correct_and_total_count.keys():
            class_correct_and_total_count[truth] = {'top1_c':0.0, 'top3_c':0.0, 'top5_c':0.0, 'total':0.0}
        class_correct_and_total_count[truth]['total'] = class_correct_and_total_count[truth]['total'] + 1

        if predictions[i][0] == truth:
            top_1_correct += 1
            class_correct_and_total_count[truth]['top1_c'] = class_correct_and_total_count[truth]['top1_c'] + 1

        if truth in predictions[i][0:3]:
            top_3_correct += 1
            class_correct_and_total_count[truth]['top3_c'] = class_correct_and_total_count[truth]['top3_c'] + 1

        # Top-5 accuracy check
        if truth in predictions[i]:
            top_5_correct += 1
            class_correct_and_total_count[truth]['top5_c'] = class_correct_and_total_count[truth]['top5_c'] + 1

    # Calculate accuracies
    top_1_accuracy = top_1_correct / total_samples
    top_3_accuracy = top_3_correct / total_samples
    top_5_accuracy = top_5_correct / total_samples
    print('For micro acc')
    print(f"Top-1 Accuracy: {top_1_accuracy * 100:.2f}%")
    print(f"Top-3 Accuracy: {top_3_accuracy * 100:.2f}%")
    print(f"Top-5 Accuracy: {top_5_accuracy * 100:.2f}%")

    top_1_class_acc_list = []
    top_3_class_acc_list = []
    top_5_class_acc_list = []



    for i in class_correct_and_total_count.keys():
        top_1_class_acc_list.append(class_correct_and_total_count[i]['top1_c']*1.0/class_correct_and_total_count[i]['total'])
        top_3_class_acc_list.append(
            class_correct_and_total_count[i]['top3_c'] * 1.0 /class_correct_and_total_count[i]['total'])
        top_5_class_acc_list.append(
            class_correct_and_total_count[i]['top5_c'] * 1.0 /class_correct_and_total_count[i]['total'])

    macro_top_1_accuracy = sum(top_1_class_acc_list) * 1.0 / len(top_1_class_acc_list)
    macro_top_3_accuracy = sum(top_3_class_acc_list) * 1.0 / len(top_1_class_acc_list)
    macro_top_5_accuracy = sum(top_5_class_acc_list) * 1.0 / len(top_1_class_acc_list)

    print('For macro acc')
    print(f"Top-1 Accuracy: {macro_top_1_accuracy * 100:.2f}%")
    print(f"Top-3 Accuracy: {macro_top_3_accuracy * 100:.2f}%")
    print(f"Top-5 Accuracy: {macro_top_5_accuracy * 100:.2f}%")

def process_all_gt_labels(all_gt_labels):
    all_gt_labels_in_list = []
    total_len = len(all_gt_labels['species'])
    for idx in range(total_len):
        curr_labels = {}
        for key in all_gt_labels.keys():
            curr_labels[key] = all_gt_labels[key][idx]
        all_gt_labels_in_list.append(curr_labels)
    return all_gt_labels_in_list

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


def encode_image_feature_and_calculate_accuracy(model, query_dataloader, key_features, key_labels):
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
    print("Initialize dataloader...")
    args.model_config.batch_size = 24
    _, _, _, seen_test_dataloader, unseen_test_dataloader, seen_keys_dataloader, val_unseen_keys_dataloader, test_unseen_keys_dataloader, all_keys_dataloader = load_bioscan_dataloader_all_small_splits(
        args)
    _, seen_val_dataloader, unseen_val_dataloader, all_keys_dataloader = load_dataloader(args)

    image_key_feature_path = os.path.join(folder_for_saving, "image_key_features.pth")
    key_labels_path = os.path.join(folder_for_saving, "key_labels.pth")
    if os.path.exists(image_key_feature_path) and os.path.exists(key_labels_path):
        key_features = torch.load(image_key_feature_path)
        key_labels = torch.load(key_labels_path)
    else:
        key_features, key_labels = make_image_key_features(model, all_keys_dataloader)
        torch.save(key_features, image_key_feature_path)
        torch.save(key_labels, key_labels_path)

    seen_macro_accuracies = encode_image_feature_and_calculate_accuracy(model, seen_test_dataloader, key_features, key_labels)

    unseen_macro_accuracies = encode_image_feature_and_calculate_accuracy(model, unseen_test_dataloader, key_features, key_labels)

    print("For image to image: ")
    for level in ['order', 'family', 'genus', 'species']:
        print(f"Level: {level}")
        seen_acc = seen_macro_accuracies[level]
        unseen_acc = unseen_macro_accuracies[level]
        harmoinc_mean_acc = harmonic_mean([seen_acc, unseen_acc])
        print(f"Seen acc：{seen_acc} || Unseen acc：{unseen_acc} || Harmonic mean acc：{harmoinc_mean_acc}")
        print(f"{round(seen_acc*100, 1)} & {round(unseen_acc*100, 1)} & {round(harmoinc_mean_acc*100, 1)}")






if __name__ == "__main__":
    main()
