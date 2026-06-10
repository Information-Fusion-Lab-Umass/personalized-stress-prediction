import os
import sys
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, average_precision_score

from calm_net import *
# from calm_net_gen import *

# srun -p gpu-preempt --gres=gpu:1 --mem=16G --pty bash

np.random.seed(42)
torch.manual_seed(42)

class WESAD(Dataset):
    def __init__(self, fnames, 
                 sample_root="data/samples",
                 covs_path='data/RegularizedData4Hz.pkl'):  
        self.fnames = fnames
        self.sample_root = sample_root

        with open(covs_path, "rb") as f:
            self.covs = pickle.load(f)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        # load sample data
        with open(os.path.join(self.sample_root, self.fnames[idx]), "rb") as f:
            sample = pickle.load(f)
        sub_id = self.fnames[idx].split("_")[0]



        '''
        Deprecated: Semantic portrait embeddings
        
        This block previously loaded subject-level "semantic portraits"
        derived from demographic text. It is retained here for reference
        in case this approach is revisited in future research.
        
        Original procedure:
        1. Encode each subject's demographic text using a pretrained
           language model (e.g., ClinicalBERT or BioBART).
        2. Obtain a subject-level embedding by applying global mean
           pooling over token-level representations.
        3. Ensure the final representation is a single fixed-length
           vector, compatible with other covariates.
        4. (Optional) Apply PCA on the training embeddings and retain
           enough components to explain 95% or 99% of the variance.
        
        The code below attempted to load a precomputed portrait for each
        subject, falling back to the population mean embedding if unavailable.
        '''
        # try:
        #     with open("data/portraits/{}".format(sub_id), 'rb') as f:
        #         portrait = pickle.load(f)
        # except:
        #     print(sub_id, "portrait not found.")
        #     with open("data/portraits/mean", 'rb') as f:
        #         portrait = pickle.load(f)



        # Extract covariates
        cov = list()
        cov.append(int(self.covs["S{}".format(sub_id)]["Covs"]["Age"]) / 100)
        cov.append(int(self.covs["S{}".format(sub_id)]["Covs"]["Height"]) / 200)
        cov.append(int(self.covs["S{}".format(sub_id)]["Covs"]["Weight"]) / 100)
        cov.append(1 if self.covs["S{}".format(sub_id)]["Covs"]["Gender"] == "male" else 0)
        cov.append(1 if self.covs["S{}".format(sub_id)]["Covs"]["Hand"] == "right" else 0)
        cov.append(0 if self.covs["S{}".format(sub_id)]["Covs"]["CoffeeToday"] == "No" else 0)
        cov.append(0 if self.covs["S{}".format(sub_id)]["Covs"]["Sports"] == "No" else 0)
        cov.append(0 if self.covs["S{}".format(sub_id)]["Covs"]["Smoker"] == "No" else 0)
        cov.append(0 if self.covs["S{}".format(sub_id)]["Covs"]["ill"] == "No" else 0)

        # packing
        return {
            'signals': torch.tensor(sample["signal"]).float().to(DEVICE),
            'labels': torch.tensor(sample["label"]).long().to(DEVICE),
            'subjects': sample["subject"],
            # "portrait": torch.tensor(portrait).float().to(DEVICE),
            "covs": torch.tensor(cov).float().to(DEVICE)
        }

def load_model(model_name, subjects):
    # baseline setting
    num_branch = 1
    personalize = False

    # calm-net setting
    if model_name == "personalize":
        personalize = True
        print("Use Personalization")
    elif model_name == 'personalize_branch':
        print("Use Branching...")
        num_branch = 3
        personalize = True
    elif model_name == "baseline":
        pass
    else:
        print("Model name", model_name, "not supported.")
        exit()

    model = BranchNetMultiTask(
        in_channel=14, 
        seq_length=240,
        emb_size=16, 
        hidden_size=256,
        dropout=0.5,
        num_classes=3, 
        num_branch=num_branch, # 1 is CALM-Net, >1 is Branched CALM-Net
        subjects=subjects,
        personalize=personalize
    ).to(DEVICE)

    return model

def eval_res(y_preds, y_trues, labels=[0, 1, 2]):
    pred_class = np.argmax(y_preds, axis=1)
    y_preds = np.array(y_preds)
    return {
        "confmats": confusion_matrix(y_trues, pred_class, labels=labels),
        "f1": f1_score(y_trues, pred_class, average='micro', labels=labels),
        "roc_auc": roc_auc_score(y_trues, y_preds, average='weighted', labels=labels, multi_class='ovr'),
        "acc": np.mean(pred_class == y_trues)
    }

def eval_one_fold(i, trail_name="baseline", split_name="data/splits_5fold",
                  sample_root="data/samples",
                  covs_path='data/RegularizedData4Hz.pkl',
                  result_output_root='data/exp_res/baseline'):
    os.makedirs(os.path.join(result_output_root, trail_name), exist_ok=True)

    # get splits
    with open(split_name, "rb") as f:
        splits_f = pickle.load(f)
        splits = splits_f["splits"]
        subjects = splits_f["subjects"]

    split = splits[i]
    # load data
    train_dataset = WESAD(split["train_fnames"], sample_root=sample_root, covs_path=covs_path)
    eval_dataset = WESAD(split["test_fnames"], sample_root=sample_root, covs_path=covs_path)

    # construct model
    model = load_model(trail_name, subjects)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Num param:", pytorch_total_params)
    # exit()

    # fit model
    record = model.fit(
        train_dataset,
        eval_dataset,
        batch_size=64, # 64
        epochs=30, # 30
        lr=1e-3, # 1e-3, 1e-2
        weight_decay=1e-5, # 1e-5
        step_size=10 # 10
    )

    # plot the training losses
    fig_name = os.path.join(result_output_root, trail_name, "{}_{}.png".format(trail_name, i))
    plt.clf()
    plt.plot(record["train_i"], record["train_losses"], label="Train")
    plt.plot(record["eval_i"], record["eval_losses"], label="Eval")
    plt.legend()
    plt.savefig(fig_name)

    # final evaluation and save the record
    curr_scores = eval_res(record["last_pred"]["y_preds"], record["last_pred"]["y_trues"])
    for s in curr_scores:
        print(s, curr_scores[s])
    
    record_name = os.path.join(result_output_root, trail_name, "{}_record_{}".format(trail_name, i))
    with open(record_name, "wb") as f:
        records = {
            "y_preds": record["last_pred"]["y_preds"],
            "y_trues": record["last_pred"]["y_trues"],
        }
        pickle.dump(records, f)
    
    return fig_name, record_name

if __name__ == "__main__":
    # NOTE: example running command:
    # python3 -m main baseline cv

    # command line arguments
    trail_name = sys.argv[1]
    split_scheme = sys.argv[2]



    # config
    sample_root = "data/samples"
    covs_path = 'data/RegularizedData4Hz.pkl'
    result_output_root = 'data/exp_res'
    
    num_fold = 15 if split_scheme == 'loocv' else 5
    portion = 0.2
    print("Portion:", portion)



    # main workflow for cross validation
    for i in range(num_fold):
        split_name = "data/splits_subjects_loocv_{}".format(portion) if split_scheme == 'loocv' else "data/splits_5fold"

        # _, _ = eval_one_fold(i, trail_name=trail_name, split_name="data/time_split")
        _, _ = eval_one_fold(i, trail_name=trail_name, split_name=split_name,
                             sample_root=sample_root, covs_path=covs_path,
                             result_output_root=result_output_root)
