import shap
import torch
import numpy as np
from plot import plot_topomap
from torch.utils.data import DataLoader, RandomSampler
from dataset import get_channel_list, transform_back_to_origin


def create_dataloader(dataset, num_samples, batch_size, device):
    sampler = RandomSampler(dataset, num_samples=num_samples, replacement=False)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return torch.cat([batch[0].to(device) for batch in loader], dim=0)


def get_feature_attribution(
    model,
    train_dataset,
    test_dataset,
    architecture,
    leader_size=1000,
    explain_size=200,
    batch_size=32,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    leader_size = min(leader_size, len(train_dataset))
    explain_size = min(explain_size, len(test_dataset))

    leader_data = create_dataloader(train_dataset, leader_size, batch_size, device)
    explain_data = create_dataloader(test_dataset, explain_size, batch_size, device)

    explainer = shap.DeepExplainer(model, leader_data)
    shap_values = explainer.shap_values(explain_data, check_additivity=False)

    shap_values_abs_mean = np.mean(np.abs(shap_values), axis=(0, -1))
    shap_values_abs_mean = torch.tensor(shap_values_abs_mean, dtype=torch.float32)
    shap_values_abs_mean = transform_back_to_origin(shap_values_abs_mean, architecture)
    shap_values_abs_mean = shap_values_abs_mean.mean(dim=1, keepdim=True)

    shap_values_pos_mean = np.mean(
        np.where(shap_values > 0, shap_values, 0), axis=(0, -1)
    )
    shap_values_pos_mean = torch.tensor(shap_values_pos_mean, dtype=torch.float32)
    shap_values_pos_mean = transform_back_to_origin(shap_values_pos_mean, architecture)
    shap_values_pos_mean = shap_values_pos_mean.mean(dim=1, keepdim=True)

    shap_values_neg_mean = np.mean(
        np.where(shap_values < 0, shap_values, 0), axis=(0, -1)
    )
    shap_values_neg_mean = torch.tensor(shap_values_neg_mean, dtype=torch.float32)
    shap_values_neg_mean = transform_back_to_origin(shap_values_neg_mean, architecture)
    shap_values_neg_mean = shap_values_neg_mean.mean(dim=1, keepdim=True)

    plot_topomap(
        torch.cat(
            [shap_values_abs_mean, shap_values_pos_mean, shap_values_neg_mean], axis=1
        ),
        fig_label=f"{architecture} - Feature Attribution",
        channel_list=get_channel_list(architecture),
        labeled_plot_points={
            "Absolute Mean": 0,
            "Positive Mean": 1,
            "Negative Mean": 2,
        },
        show_names=True,
    )
