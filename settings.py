TOKENIZER_NAME = "unsloth/mistral-7b-v0.3-bnb-4bit"
TRAINING_FILE = "data/train.txt"
EVAL_FILE = "data/eval.txt"
# Configuration parameters

n_embd = 4096
n_head = 32
n_layer = 32
head_size = 128
dropout = 0.1
block_size = 32768
num_experts = 8
top_k = 2
