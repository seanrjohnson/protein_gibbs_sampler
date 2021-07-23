import torch
import math

from pgen import models


def calculate_log_likelihood(model, seq, verbose=False):
    _init_model(model)
    log_likelihood_sum = 0
    for idx, val in enumerate(seq):
        alphabet_idx = model.alphabet.get_idx(val)
        masked_seq = _mask_index(idx, seq)
        batch = [(str(0), masked_seq)]

        _, _, tokens = model.batch_converter(batch)

        logits = model.model(tokens)['logits']
        prob = _convert_logits_to_prob(logits)
        idx_prob = prob[0][idx][alphabet_idx]
        log_likelihood = math.log(idx_prob)

        if verbose:
            print(idx, val, idx_prob, log_likelihood)

        log_likelihood_sum += log_likelihood

    return log_likelihood_sum / len(seq)


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
    if device == "gpu":
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            raise (Exception("gpu requested, but No Cuda devices found"))
    else:
        device = "cpu"
    device = torch.device(device)
    model.model.to(device)


def _convert_logits_to_prob(logits):
    return torch.distributions.categorical.Categorical(logits=logits).probs


print(calculate_log_likelihood(models.ESM12(), "MRHGDISSSNDTVGVAVVNYKMPRLHTAAEVLDNAR"))
print(calculate_log_likelihood(models.ESM12(), "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"))
print(calculate_log_likelihood(models.ESM12(), "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRS"))
print(calculate_log_likelihood(models.ESM6(), "MRHGDISSSNDTVGVAVVNYKMPRLHTAAEVLDNAR"))
print(calculate_log_likelihood(models.ESM6(), "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"))
print(calculate_log_likelihood(models.ESM6(), "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRS"))
