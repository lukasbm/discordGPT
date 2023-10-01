from dataset import MessagesDataset
from model import model
from mingpt.trainer import Trainer
from mingpt.utils import set_seed

set_seed(3407)

# your subclass of torch.utils.data.Dataset that emits example
# torch LongTensor of lengths up to 1024, with integers from [0,50257)
train_dataset = MessagesDataset()

train_config = Trainer.get_default_config()
train_config.learning_rate = 5e-4  # many possible options, see the file
train_config.max_iters = 1000
train_config.batch_size = 32
trainer = Trainer(train_config, model, train_dataset)
trainer.run()
