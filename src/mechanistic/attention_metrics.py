import torch
import numpy as np


def attention_metrics(model, input_ids):

    with torch.no_grad():

        outputs = model(
            input_ids,
            output_attentions=True,
            return_dict=True
        )

    if outputs.attentions is None:
        return np.nan, np.nan

    entropies = []
    self_ratios = []

    for layer_attn in outputs.attentions:

        attn = layer_attn[0]

        avg_attn = attn.mean(dim=0)

        entropy = -torch.sum(
            avg_attn * torch.log(avg_attn + 1e-9)
        ).item()

        entropies.append(entropy)

        self_ratio = avg_attn[-1, -1].item()

        self_ratios.append(self_ratio)

    return (
        np.mean(entropies),
        np.mean(self_ratios)
    )
