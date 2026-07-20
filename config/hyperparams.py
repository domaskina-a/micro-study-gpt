SEED = 1337

DATASET_PATH = "dataset.txt"

# Model
BLOCK_SIZE = 8
D_MODEL = 32
NUM_HEADS = 4  # d_model is split across the heads: head_dim = 32 / 4 = 8
FFN_MULTIPLIER = 4  # hidden width of the feed-forward layer: d_model -> d_model * 4 -> d_model

# Training
BATCH_SIZE = 4
LEARNING_RATE = 3e-3  # the model is tiny, so it tolerates a much larger step than a real GPT
MAX_STEPS = 500
LOG_INTERVAL = 50

# Generation
PROMPT = "the sun"
MAX_NEW_TOKENS = 20
