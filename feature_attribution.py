import shap
import torch
import numpy as np
from plot import plot_topomap
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, RandomSampler
from dataset import get_channel_list, transform_back_to_origin


def create_dataloader(dataset, num_samples, batch_size, device):
    sampler = RandomSampler(dataset, num_samples=num_samples, replacement=False)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return torch.cat([batch[0].to(device) for batch in loader], dim=0)


# Takes N x [input_shape] x C and returns 32 x 1
def transform(values, architecture, axis=(0, -1)):
    if axis is not None:
        values = np.mean(values, axis=axis)
    values = torch.tensor(values, dtype=torch.float32)
    values = transform_back_to_origin(values, architecture)
    return values.mean(dim=1, keepdim=True)


def get_feature_attribution(
    model,
    train_dataset,
    test_dataset,
    architecture,
    device,
    leader_size=500,
    explain_size=100,
    batch_size=32,
):
    model.to(device).eval()

    leader_size = min(leader_size, len(train_dataset))
    explain_size = min(explain_size, len(test_dataset))

    leader_data = create_dataloader(train_dataset, leader_size, batch_size, device)
    explain_data = create_dataloader(test_dataset, explain_size, batch_size, device)

    explainer = shap.DeepExplainer(model, leader_data)
    shap_values = explainer.shap_values(explain_data, check_additivity=True)

    shap_values_pos_mean = transform(
        np.where(shap_values > 0, shap_values, 0), architecture
    )
    shap_values_neg_mean = transform(
        np.where(shap_values < 0, shap_values, 0), architecture
    )
    shap_values_abs_mean = transform(np.abs(shap_values), architecture)

    fig_label = f"{architecture} - Feature Attribution"
    plot_topomap(
        torch.cat(
            [shap_values_abs_mean, shap_values_pos_mean, shap_values_neg_mean], axis=1
        ),
        fig_label=fig_label,
        channel_list=get_channel_list(architecture),
        labeled_plot_points={
            "Absolute Mean": 0,
            "Positive Mean": 1,
            "Negative Mean": 2,
        },
        show_names=True,
    )

    shap_values_summary = np.moveaxis(shap_values, -1, 1)
    shap_values_summary = shap_values_summary.reshape(-1, *shap_values.shape[1:-1])
    shap_values_summary = torch.stack(
        [
            transform(shap_value, architecture, axis=None).squeeze()
            for shap_value in shap_values_summary
        ]
    )

    shap.summary_plot(
        shap_values_summary.cpu().numpy(),
        feature_names=np.array(get_channel_list(architecture)),
        show=False,
    )
    plt.savefig(f"./results/{fig_label}", dpi=300, bbox_inches="tight")
