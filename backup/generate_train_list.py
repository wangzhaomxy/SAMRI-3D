# Import necessary libraries
import pandas as pd
import pickle
from tqdm import tqdm

f_path = "/Users/wangzhao/Library/CloudStorage/OneDrive-TheUniversityofQueensland/PhD/CodePractice/samri_results/"

# Function to get the balance factor
def get_balance_factor(result):
    ds_uniq = result.groupby("Dataset").emb_path.nunique()
    max_num = ds_uniq.max()
    balance_factor = {ds_name: round(max_num / (10 * ds_uniq[ds_name]) + 0.5) for ds_name in list(ds_uniq.index)}
    return balance_factor

# Function to get the balanced data
def get_balanced_data(data):
    balance_factor = get_balance_factor(data)
    balanced_data = []
    for dataset, group in data.groupby("Dataset"):
        unique_emb_paths = group[["emb_path", "labels"]].drop_duplicates().to_dict(orient="records")
        balanced_data += unique_emb_paths * balance_factor[dataset]
    return balanced_data

# Load the vitb_train_result CSV file
vitb_train_result = pd.read_csv(f_path + "20250518samri_sam_medsam/" + "sam_vit_b_01ec64_train_ds.csv")
vitb_train_result["emb_path"] = vitb_train_result.img_path.apply(lambda x: 
    "/scratch/user/s4670484/Embedding/" + 
    x.split("/")[-3] + "/" + 
     x.split("/")[-1][:-7] + ".npz")

# Filter results with b_dice > 0.2 and <= 0.6
result_20_60 = vitb_train_result[
    (vitb_train_result["b_dice"] > 0.2) & 
    (vitb_train_result["b_dice"] <= 0.4) 
][["Dataset", "labels", "emb_path"]]

# Filter results with b_dice > 0.4
result_60_100 = vitb_train_result[
    (vitb_train_result["b_dice"] > 0.4) 
][["Dataset", "labels", "emb_path"]]

# Get the balance data
train_60_down = get_balanced_data(result_20_60)
train_60_up = get_balanced_data(result_60_100)

# Get the balanced data 
train_list = {
    "train_60_down": train_60_down,
    "train_60_up": train_60_up
}

# Save the train list to a pickle file
with open(f_path + "train_list.pkl", "wb") as f:
    pickle.dump(train_list, f)