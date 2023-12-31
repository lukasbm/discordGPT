# discordGPT

## Setup

1. Get you discord data package [here](https://support.discord.com/hc/en-us/articles/360004027692)
2. Clone this repo: `git clone git@github.com:lukasbm/discordGPT.git`
3. place your extracted package in the root of the repo: `unzip -d package package.zip`
4. create a python env and install the dependencies: `python3 -m venv venv && pip install -U -r requirements.txt`
5. create a file called `channels.txt` in the root of the repository.
   Place the IDs of the channels you want to include in the training dataset in there.
   One ID per line. **TIP:**
   use [developer mode](https://www.howtogeek.com/714348/how-to-enable-or-disable-developer-mode-on-discord/) to quickly
   get the channel ids

## Train

Just call `python train.py`
Every 500 iterations the model checkpoint will be saved in the output directory (default: `/out`)

## Disclaimer

All code is run locally and none of your data is uploaded anywhere.
See [pytorch privacy guidelines](https://discuss.pytorch.org/privacy) for more information.

## References

- https://github.com/karpathy/minGPT
- https://bootcamp.uxdesign.cc/a-step-by-step-guide-to-building-a-chatbot-based-on-your-own-documents-with-gpt-2d550534eea5

