import esm

class ESM1b():
    def __init__(self):
        self.model, self.alphabet = esm.pretrained.esm1_t34_670M_UR50S()
        self.batch_converter = self.alphabet.get_batch_converter()
