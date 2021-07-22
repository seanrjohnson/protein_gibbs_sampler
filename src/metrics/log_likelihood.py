import torch
import math

from pgen import models


def calculate_log_likelihood(model, seq):
    _init_model(model)
    sum = 0
    for idx, val in enumerate(seq):
        masked_seq = _mask_index(idx, seq)
        batch = [(str(0), masked_seq)]
        _, _, tokens = model.batch_converter(batch)
        logits = model.model(tokens)['logits']
        prob = _convert_logits_to_prob(logits)
        alphabet_idx = model.alphabet.get_idx(val)

        idx_prob = prob[0][idx][alphabet_idx]
        log_likelihood = math.log(idx_prob)
        print(idx, val, idx_prob, log_likelihood)

        sum += log_likelihood

    return sum / len(seq)


def _mask_index(idx, seq):
    if idx > 0:
        masked_seq = seq[0:idx] + "<mask>" + seq[idx + 1:]
    else:
        masked_seq = "<mask>" + seq[idx:]
    return masked_seq


def _init_model(model, device="cpu"):
    """
        model should be an object with parameters model, alphabet, and batch_converter
    """
    model.model = model.model.eval()
    if (device == "gpu"):
        if (torch.cuda.is_available()):
            device = 'cuda:0'
        else:
            raise (Exception("gpu requested, but No Cuda devices found"))
    else:
        device = "cpu"
    device = torch.device(device)
    model.model.to(device)


def _convert_logits_to_prob(logits):
    prob = torch.tensor(logits)
    for batch in range(len(prob)):
        for char in range(len(prob[batch])):
            for alpha in range(len(prob[batch][char])):
                x = float(prob[batch][char][alpha])
                odds = math.e ** x
                prob[batch][char][alpha] = odds / (odds + 1)
    print(max(prob.sum(dim=2)))
    prob = torch.nn.Softmax(dim=2)(prob)  # For some reason they don't sum to 1? Doing a softmax to normalize
    print(max(prob.sum(dim=2)))
    return prob


print(calculate_log_likelihood(models.ESM6(), "ACDEFGHIKL"))
