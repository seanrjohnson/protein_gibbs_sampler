import esm

class ESM12():
    def __init__(self):
        self.model, self.alphabet = esm.pretrained.esm1_t12_85M_UR50S()
        self.batch_converter = self.alphabet.get_batch_converter()
