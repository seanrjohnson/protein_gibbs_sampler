import logging
import random

logger = logging.getLogger(__name__)

def mask_sequence(tokenized_sequence, alphabet):
    """
    Adapted from hugging face
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokenized_sequence: list of str representing all tokens of a single tokenized sequence.
    :param alphabet: Alphabet, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    masked_sequence = [alphabet.get_idx(token) for token in tokenized_sequence] # TODO -- Deep copy
    output_label = []

    MASK_TOK = alphabet.get_tok(alphabet.mask_idx)
    UNK_TOK = "<unk>"
    UNK_TOK_IDX = alphabet.get_idx(UNK_TOK)
    ALPHABET_TOKS = alphabet.standard_toks # Does not include special tokens

    for i, token in enumerate(tokenized_sequence):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                masked_sequence[i] = alphabet.mask_idx

            # 10% randomly change token to random token
            elif prob < 0.9:
                # tokens[i] = random.choice(list(ALPHABET_TOKS.items()))[0]
                masked_sequence[i] = alphabet.get_idx(random.choice(ALPHABET_TOKS)[0])

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                #TODO -- alphabet.get_idx(masked_sequence[i])?
                output_label.append(alphabet.get_idx(token))
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(UNK_TOK_IDX)
                logger.warning("Cannot find token '{}' in vocab. Using {} instead".format(token, UNK_TOK))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return masked_sequence, output_label
