import json
import argparse

from src.Trainer import Trainer


def parse_args(args):
    parser = argparse.ArgumentParser(description='Choose config: ')
    parser.add_argument('--config', type=str, default="configs/configs")
    return parser.parse_args(args)


def main(args=None):

    args = parse_args(args)
    config = json.load(open(args.config))
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
