import esm

class ESM6():
    def __init__(self):
        self.model, self.alphabet = esm.pretrained.esm1_t6_43M_UR50S()
        self.batch_converter = self.alphabet.get_batch_converter()
