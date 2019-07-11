
import logging
import os

import cv2
import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)

class RssraiDataset(Dataset):

    def __init__(self, 
                dataset, 
                config,
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]):
        
        self.num_classes = config.NUM_CLASSES
        if dataset == 'train':
            self.data_path = os.path.join(config.ROOT, config.TRAIN_SET)
        elif dataset == 'test':
            self.data_path = os.path.join(config.ROOT, config.TEST_SET)

        self.lbl2pixel, self.pixel2lbl = self.generate_label_mapping()

        self.mean = mean
        self.std = std

        self._db = self.__get_db__()

    def generate_label_mapping(self):
        lbls = [[0,0,0],
                [0,200,0],
                [150,250,0],
                [150,200,150],
                [200,0,200],
                [150,0,250],
                [150,150,250],
                [250,200,0],
                [200,200,0],
                [200,0,0],
                [250,0,150],
                [200,150,150],
                [250,150,150],
                [0,0,200],
                [0,150,200],
                [0,200,250]]
        lbls = [tuple(l) for l in lbls]
        
        assert len(lbls) == self.num_classes
        label2pixel = {}
        pixel2label = {}
        for i, lbl in enumerate(lbls):
            label2pixel[i] = lbl
            pixel2label[lbl] = i
        return label2pixel, pixel2label
    
    def __get_db__(self):
        files = []
        for f in os.listdir(os.path.join(self.data_path, 'src')):
            image_path = os.path.join(self.data_path, 'src', f)
            label_path = os.path.join(self.data_path, 'label', f)
            files.append({
                "image": image_path,
                "label": label_path,
                "name": f
            })
        logger.info("=> Loading %d files" % len(files))
        return files

    def __len__(self):
        return len(self._db)

    def __input_transform__(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def __generate_target__(self, label):
        height, width, _ = label.shape
        target = np.zeros((height, width), dtype=np.int32)

        for k, v in self.lbl2pixel.items():
            target[(label==v).all(axis=2)] = k
            # print(np.sum((label==v).all(axis=2)))
            # print(np.sum(target==k))
        return target



    def __getitem__(self, index):
        item = self._db[index]

        image = cv2.imread(item["image"],cv2.IMREAD_COLOR)
        label = cv2.imread(item["label"],cv2.IMREAD_COLOR)[:,:,::-1]
        image = self.__input_transform__(image)
        label = self.__generate_target__(label)
        image = image.transpose(2,0,1)
        return image.copy(), label.copy()


if __name__ == "__main__":
    import sys
    sys.path.append('E:\\Programmer\\RSSAI')

    from configs import cfg
    from torch.utils.data import  DataLoader
    import matplotlib.pyplot as plt
    
    dataset = RssraiDataset('train', cfg.DATASET)
    dataloader = DataLoader(dataset, batch_size=4,
                        shuffle=True, num_workers=1)
    for src, label in dataloader:
        fig = plt.figure()
        print(src.size())
        for i in range(4):
            ax1 = fig.add_subplot(4,2,2*i+1)
            ax1.imshow(src[i])
            ax2 = fig.add_subplot(4,2,2*(1+i))
            ax2.imshow(label[i], cmap='gray', vmin=0, vmax=16)
        plt.show()
        break