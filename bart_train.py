import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset

# =============================================
# 1. Load Dataset
# =============================================
ds = load_dataset("martymukherjee/Lyapunov")

# =============================================
# 2. Masking Functions
# =============================================
def mask_random_substring(s):
    s = s.split()
    if len(s) <= 2:
        return '<mask>'
    start = np.random.randint(1, len(s) - 1)
    # Choose to mask forward or backward from the chosen index.
    if np.random.randint(2) == 0:
        end = np.random.randint(start + 1, len(s))
        s[start:end] = ['<mask>']
    else:
        end = np.random.randint(0, start)
        s[end:start] = ['<mask>']
    return " ".join(s)

def mask_vector_field(vf):
    vf_list = vf.split('<SPECIAL_3>')
    n = len(vf_list) - 1
    if n <= 3:
        num_masks = 1
    else:
        num_masks = np.random.randint(1, n // 2)
    mask_indices = np.random.choice(n, num_masks, replace=False) + 1
    for idx in mask_indices:
        vf_list[idx] = mask_random_substring(vf_list[idx])
    # Clean up double spaces if any
    return (' <SPECIAL_3> '.join(vf_list)).replace("  ", " ")

# =============================================
# 3. Custom Tokenizer
# =============================================
from collections import OrderedDict
import sympy as sp
from transformers import PreTrainedTokenizer

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
        
        # Build vocabulary with special words, constants, variables, operators, symbols, elements, and the mask token.
        self.words = SPECIAL_WORDS + self.constants + list(self.variables.keys()) + list(self.operators.keys()) + self.symbols + self.elements + [self.mask_token]
        self.vocab = {s: i for i, s in enumerate(self.words)}
        self.inv_vocab = {i: s for s, i in self.vocab.items()}
        
        super().__init__(
            model_max_length=4096, 
            bos_token="<s>", 
            eos_token="</s>"
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

# Instantiate the custom tokenizer.
tokenizer = LyapunovTokenizer()
# (Note: Your tokenizer’s vocab size is 1052 as expected.)

# =============================================
# 4. Custom Dataset Class
# =============================================
class LyapunovDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        example = self.dataset[idx]
        vector_field = example["vector_field"]
        lyap_fn = example["lyap_fn"]
        # Create a masked version of the vector_field for the autoencoding task.
        masked_vector_field = mask_vector_field(vector_field)
        return {
            "vector_field": vector_field,
            "masked_vector_field": masked_vector_field,
            "lyap_fn": lyap_fn
        }

train_dataset = LyapunovDataset(ds["train"])

# =============================================
# 5. Load BART Base Model & Resize Embeddings
# =============================================
from transformers import BartForConditionalGeneration
# Load the pre-trained BART model.
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
# Resize token embeddings to match your tokenizer’s vocabulary size.
model.resize_token_embeddings(tokenizer.vocab_size)

# =============================================
# 6. Data Collator for Dual Tasks
# =============================================
def collate_fn(batch):
    # Task 1: Reconstruction – input: masked_vector_field, target: original vector_field.
    masked_texts = [item["masked_vector_field"] for item in batch]
    original_texts = [item["vector_field"] for item in batch]
    # Task 2: Generation – input: unmasked vector_field, target: lyap_fn.
    vector_texts = [item["vector_field"] for item in batch]
    lyap_texts = [item["lyap_fn"] for item in batch]
    
    inputs_rec = tokenizer(masked_texts, padding=True, truncation=True, return_tensors="pt")
    labels_rec = tokenizer(original_texts, padding=True, truncation=True, return_tensors="pt").input_ids
    inputs_gen = tokenizer(vector_texts, padding=True, truncation=True, return_tensors="pt")
    labels_gen = tokenizer(lyap_texts, padding=True, truncation=True, return_tensors="pt").input_ids

    # Remove token_type_ids if present, as BART does not use them.
    inputs_rec.pop("token_type_ids", None)
    inputs_gen.pop("token_type_ids", None)
        
    # Replace padding token id's in labels with -100 to ignore them in loss computation.
    labels_rec[labels_rec == tokenizer.vocab["<pad>"]] = -100
    labels_gen[labels_gen == tokenizer.vocab["<pad>"]] = -100
    
    return {
        "inputs_rec": inputs_rec,
        "labels_rec": labels_rec,
        "inputs_gen": inputs_gen,
        "labels_gen": labels_gen
    }

# =============================================
# 7. Training Setup
# =============================================
batch_size = 8  # Adjust batch size as needed.
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = model.to(device)

num_epochs = 300000  # Adjust number of epochs as needed.
optimizer = AdamW(model.parameters(), lr=5e-5)
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

# =============================================
# 8. Training Loop (Dual-Task per Epoch)
# =============================================
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch in train_loader:
        # Move all tensors to device.
        inputs_rec = {k: v.to(device) for k, v in batch["inputs_rec"].items()}
        labels_rec = batch["labels_rec"].to(device)
        inputs_gen = {k: v.to(device) for k, v in batch["inputs_gen"].items()}
        labels_gen = batch["labels_gen"].to(device)
        
        optimizer.zero_grad()
        
        # --- Step 1: Autoencoding Reconstruction ---
        outputs_rec = model(**inputs_rec, labels=labels_rec)
        loss_rec = outputs_rec.loss
        
        # --- Step 2: Lyapunov Function Generation ---
        outputs_gen = model(**inputs_gen, labels=labels_gen)
        loss_gen = outputs_gen.loss
        
        # Total loss is the sum of both objectives.
        loss = loss_rec + loss_gen
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} — Loss: {avg_loss:.4f}")
    
    # =============================================
    # 9. Push to Hugging Face Hub
    # =============================================
    # Ensure you are logged in (e.g., via `huggingface-cli login`).
    model.push_to_hub("martymukherjee/lyapunov-bart", commit_message=f"Epoch {epoch+1} update")
    
    # Optionally save a local checkpoint.
    output_dir = f"./checkpoint-epoch-{epoch+1}"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_vocabulary(output_dir)

print("Training complete.")
# =============================================
