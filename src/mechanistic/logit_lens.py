import torch
import torch.nn.functional as F


def logit_lens(model, input_ids):

    with torch.no_grad():

        outputs = model(
            input_ids,
            output_hidden_states=True,
            return_dict=True
        )

    hidden_states = outputs.hidden_states

    lm_head = model.lm_head

    confidences = []

    for layer_hidden in hidden_states:

        logits = lm_head(layer_hidden)

        probs = F.softmax(
            logits[:, -1, :],
            dim=-1
        )

        confidences.append(
            torch.max(probs).item()
        )

    return confidences
