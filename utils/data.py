from torchvision import datasets, transforms
from utils.toolkit import read_annotations
from imgaug import augmenters as iaa


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class OSMA(iData):
    use_path = True
    train_trsf = []
    test_trsf = []
    common_trsf = [
        # transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    class_order = [0, 1, 2, 3, 4, 5, 6, 7]


    def process_data(self, train_data_path, val_data_path, test_data_path, out_data_path, debug):
        self.train_data, self.train_targets = read_annotations(train_data_path, debug)
        self.val_data, self.val_targets = read_annotations(val_data_path, debug)
        self.test_data, self.test_targets = read_annotations(test_data_path, debug)
        self.out_data, self.out_targets = read_annotations(out_data_path, debug)
