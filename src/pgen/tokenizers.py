class Tokenizer:
    pass

class CharacterTokenizer(Tokenizer):
    @staticmethod
    def tokenize(sequence):
        return [c for c in sequence]
