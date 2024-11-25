import os
import csv
import math
import json
import yaml
import shutil
import numpy as np
import pandas as pd
from argparse import ArgumentParser

path_template = 'logs/{}_val_top1_{}_results_with_nearest.csv'
file_name_list = ["before", "IT", "ID", "IDT"]
seen_type_list = ["seen", "unseen"]


# main figure
img_seen_list = [
    'BIOUG76074-A02',
    'BIOUG66353-C06',
    'BIOUG75991-A10',
    '4107604',
]
img_unseen_list = [
    'BIOUG68137-A07',
    '4842849',
    'BIOUG71679-F07',
    '4282372',
]

# supplementary figure
img_seen_list_sup = [
    'BIOUG68700-H02',
    'BIOUG78896-C06',
    'BIOUG71429-F12',
    '3669757',
    '5941497',
    '4804097',
    'BIOUG67833-H04',
    'BIOUG66355-C07',
    'BIOUG71682-D01',
    '4765649',
    '4808300',
]

img_unseen_list_sup = [
    '5012285',
    '4264242',
    'BIOUG73265-H02',
    'BIOUG70033-B02',
    'BIOUG71526-D06',
    '5500864',
    '6110806',
    'BIOUG79506-G04',
    'BIOUG66056-G05',
    '4220099',
    'BIOUG66030-F09',
]

flag_original = False
os.makedirs("logs/retrieval", exist_ok=True)


latex_template = ""
# for seen_id, unseen_id in zip(img_seen_list, img_unseen_list):
for seen_id, unseen_id in zip(img_seen_list_sup, img_unseen_list_sup):

    if flag_original:
        pass


    latex_seen_string = "\\small nearest retrieval & \n"; latex_unseen_string = "\\small nearest retrieval & \n"
    for file_name in file_name_list:
        seen_result_path = path_template.format(file_name, seen_type_list[0])
        unseen_result_path = path_template.format(file_name, seen_type_list[1])

        seen_df = pd.read_csv(seen_result_path)
        seen_nearest_id = seen_df.loc[seen_df['GT_id'] == seen_id]['Pred_id'].values[0]

        latex_seen_string += f"\\includegraphics{{figs/images/attention_maps/retrieval/{seen_nearest_id}.png}}"
        latex_seen_string += " & \n"

        seen_image_path_save = f"logs/retrieval/{seen_nearest_id}.png"
        seen_image_path_origin = f"representation_visualization/all_key/{seen_nearest_id}.png"
        if not os.path.exists(seen_image_path_save):
            shutil.copyfile(seen_image_path_origin, seen_image_path_save)


        unseen_df = pd.read_csv(unseen_result_path)
        unseen_nearest_id = unseen_df.loc[unseen_df['GT_id'] == unseen_id]['Pred_id'].values[0]
        latex_unseen_string += f"\\includegraphics{{figs/images/attention_maps/retrieval/{unseen_nearest_id}.png}}"
        latex_unseen_string += " & \n" if file_name != file_name_list[-1] else "\\\\ \n"

        unseen_image_path_save = f"logs/retrieval/{unseen_nearest_id}.png"
        unseen_image_path_origin = f"representation_visualization/all_key/{unseen_nearest_id}.png"
        if not os.path.exists(unseen_image_path_save):
            shutil.copyfile(unseen_image_path_origin, unseen_image_path_save)

    latex_template += latex_seen_string; latex_template += latex_unseen_string
    
    latex_template += "\n\n"

print(latex_template)