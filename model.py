from mingpt.model import GPT
from mingpt.utils import CfgNode


def get_config() -> CfgNode:
    # config: {vocab_size, block_size, model_type, n_layer, n_head, n_embd, embd_pdrop, attn_pdrop, resid_pdrop}
    cfg = GPT.get_default_config()
    cfg.model_type = "gpt-nano"
    # cfg.system.work_dir = "out"
    cfg.vocab_size = 50257  # openai's model vocabulary
    cfg.block_size = 1024  # openai's model block_size (i.e. input context length)
    return cfg


model_config = get_config()
model = GPT(model_config)
