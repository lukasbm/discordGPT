import sys
from dataset import MessagesDataset, load_text, load_channels
from mingpt.trainer import Trainer
from mingpt.model import GPT
from mingpt.utils import CfgNode, set_seed, setup_logging
import torch
import os


def get_config() -> CfgNode:
    cfg = CfgNode()

    # system
    cfg.system = CfgNode()
    cfg.system.seed = 3407
    cfg.system.work_dir = "out"

    # data
    cfg.data = MessagesDataset.get_default_config()

    # model
    cfg.model = GPT.get_default_config()
    cfg.model.model_type = 'gpt-mini'

    # trainer
    cfg.trainer = Trainer.get_default_config()
    cfg.trainer.learning_rate = 5e-4  # the model we're using is so small that we can go a bit faster

    return cfg


def batch_end_callback(t):
    """
    iteration callback

    :param t: the trainer instance itself
    """
    if t.iter_num % 10 == 0:
        print(
            f"iter_dt {t.iter_dt * 1000:.2f}ms; iter {t.iter_num}: train loss {t.loss.item():.5f}")

    if t.iter_num % 500 == 0:
        # evaluate both the train and test score
        model.eval()
        with torch.no_grad():
            # sample from the model...
            context = "ey ich hab"
            x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None, ...].to(
                t.device)
            y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
            completion = ''.join([train_dataset.itos[int(i)] for i in y])
            print(completion)
        # save the latest model
        print("saving model in:", config.system.work_dir)
        ckpt_path = os.path.join(config.system.work_dir, "model.pt")
        torch.save(model.state_dict(), ckpt_path)
        # revert model to training mode
        model.train()


# get default config and overrides from the command line, if any
config = get_config()
config.merge_from_args(sys.argv[1:])
print(config)

# setup
setup_logging(config)
set_seed(config.system.seed)

# data
full_text = load_text(load_channels())
train_dataset = MessagesDataset(config.data, full_text)

# construct the model
config.model.vocab_size = train_dataset.vocab_size
config.model.block_size = train_dataset.block_size
model = GPT(config.model)

# construct the trainer object
trainer = Trainer(config.trainer, model, train_dataset)
trainer.set_callback('on_batch_end', batch_end_callback)

# run the optimization
trainer.run()
