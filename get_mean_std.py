
import argparse
from configs.default import update_config
from configs import config
from datasets.rssrai import RssraiDataset
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        '--cfg',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')
    return parser.parse_args()

def main():
    global config
    args = parse_args()
    update_config(config, args)

    dataset = RssraiDataset('train', config.DATASET, mean=[0,0,0], std=[1,1,1])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=10,
        num_workers=4,
        shuffle=False
    )

    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, label in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    print("mean is: %f, std is: %f", mean, std)

if __name__ == "__main__":
    main()