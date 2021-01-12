from esm.pretrained import load_model_and_alphabet

class FineTuned:
    def __init__(self, checkpoint_path):
        self.model, self.alphabet = load_model_and_alphabet(checkpoint_path)
        self.batch_converter = self.alphabet.get_batch_converter()

class FineTunedESM12(FineTuned):
    def __init__(self, models_path, checkpoint_name):
        super(FineTunedESM12, self).__init__(f"{models_path}/esm1_t12_85M_UR50S/{checkpoint_name}")