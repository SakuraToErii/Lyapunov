from collections import OrderedDict
import sympy as sp
from transformers import PreTrainedTokenizer
import json
import os

SPECIAL_WORDS = ["<s>", "</s>", "<pad>", "(", ")"]
SPECIAL_WORDS = SPECIAL_WORDS + [f"<SPECIAL_{i}>" for i in range(10)]

class LyapunovTokenizer(PreTrainedTokenizer):
    def __init__(self):
        self.SYMPY_OPERATORS = {
            sp.Add: "+",
            sp.Mul: "*",
            sp.Pow: "^",
            sp.exp: "exp",
            sp.log: "ln",
            sp.Abs: "Abs",
            sp.sin: "sin",
            sp.cos: "cos",
            sp.tan: "tan",
            sp.asin: "asin",
            sp.acos: "acos",
            sp.atan: "atan",
            sp.DiracDelta: "delta0",
        }

        self.trig_ops = ["sin", "cos", "tan"]
        self.arctrig_ops = ["asin", "acos", "atan"]
        self.exp_ops = ["exp", "ln"]
        self.other_ops = ["sqrt"]
        self.pad_token = "<pad>"
        self.mask_token = "<mask>"

        op_set = {
            "+": 2,
            "-": 2,
            "*": 2,
            "/": 2,
            "^": 2,
            "sqrt": 1,
            "exp": 1,
            "ln": 1,
            "sin": 1,
            "cos": 1,
            "tan": 1,
            "asin": 1,
            "acos": 1,
            "atan": 1,
            "Abs": 1,
        }

        self.int_base = 1000
        self.max_degree = 6

        self.operators_lyap = op_set
        self.operators = self.operators_lyap

        self.variables = OrderedDict({f"x{i}": sp.Symbol(f"x{i}") for i in range(2 * self.max_degree)})
        self.constants = ["pi", "E"]
        self.symbols = ["I", "INT+", "INT-", "FLOAT+", "FLOAT-", ".", "10^"]
        self.elements = [str(i) for i in range(max(10, self.int_base))]
        # self.mask_symbol = []

        self.words = SPECIAL_WORDS + self.constants + list(self.variables.keys()) + list(self.operators.keys()) + self.symbols + self.elements + [self.mask_token]

        self.vocab = {s: i for i, s in enumerate(self.words)}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        super().__init__(
            model_max_length=4096, bos_token="<s>", eos_token="</s>"#, unk_token="<unk>", mask_token="<mask>"
        )

    def _tokenize(self, text):
        return text.split()

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index):
        return self.inv_vocab.get(index, self.unk_token)

    def get_vocab(self):
        return self.vocab

    @property
    def vocab_size(self):
        return len(self.vocab)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, "w") as f:
            json.dump(self.vocab, f)
        return (vocab_file,)
