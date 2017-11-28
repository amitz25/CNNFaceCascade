import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.serialization import load_lua
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from PIL import Image
from random import randint
from glob import glob
from matplotlib import pyplot as plt
from skimage.transform import pyramid_gaussian
import matplotlib.patches as patches
import os

learn_rate = 0.0001
num_epochs = 10
batch_size = 32
iou_threshold = 0.5
pyramid_downscale = 1.16
pyramid_len = 15
min_face_size = 54

#TODO: Replace
#work_dir = os.path.dirname(__file__)
work_dir = r'c:\Study\Courses\dl\ex2'

def get_image_pixels(image):
    pixels = torch.from_numpy(np.asarray(image))

    # Reorder from H W C to C W H
    pixels = pixels.permute(2, 1, 0)
    # Convert to Float to save up space
    return pixels.float() / 255


def get_image_random_region(image_path, crop_size, scale_size):
    img = Image.open(image_path)
    max_x = img.size[0] - crop_size - 1
    max_y = img.size[1] - crop_size - 1

    x = randint(0, max_x)
    y = randint(0, max_y)

    img = img.crop((x, y, x + crop_size, y + crop_size))
    #img = img.resize((scale_size, scale_size))
    return get_image_pixels(img)


def show_tensor_as_image(pixels):
    plt.imshow((pixels.permute(2, 1, 0) * 255).byte().numpy())
    plt.show()


def generate_pascal_dataset(image_dir, crop_size, sample_size, num_samples, output_path):
    image_list = glob(os.path.join(image_dir, "*"))
    samples = []
    while len(samples) < num_samples:
        for image_path in image_list:
            print('\rGenerating images (%d\\%d)' % (len(samples), num_samples), end="")
            if len(samples) >= num_samples:
                break
            sample = get_image_random_region(image_path, crop_size, sample_size)

            # Convert to long to save space
            samples.append((sample * 255).long())

    # Convert to long to save space
    samples = [(x * 255).long() for x in samples]
    torch.save(samples, output_path)


def load_dataset(aflw_path, pascal_path):
    aflw_dataset = load_lua(aflw_path)
    pascal_dataset = torch.load(pascal_path)
    pascal_dataset = [x.float() / 255 for x in pascal_dataset]
    dataset = [(aflw_dataset[k], 1) for k in aflw_dataset]
    dataset += [(k, 0) for k in pascal_dataset]
    return dataset


class Dataset12(Dataset):
    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return {'data': self._dataset[idx][0], 'label': self._dataset[idx][1]}


class Net12(nn.Module):
    def __init__(self):
        super(Net12, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.linear1 = nn.Linear(in_features=256, out_features=16)
        self.linear2 = nn.Linear(in_features=16, out_features=2)
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x = F.relu(self.pool(self.conv(x)))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return self.softmax(x)

class Net12FCN(nn.Module):
    def __init__(self):
        super(Net12FCN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, stride=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, stride=1)
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return self.softmax(x)


def train_net12(net):
    dataset = load_dataset(os.path.join(work_dir, r'EX2_data\aflw\aflw_12.t7'), os.path.join(work_dir, 'pascal.t7'))
    random.shuffle(dataset)
    train_size = int(len(dataset) * 0.9)

    train_dataset = Dataset12(dataset[:train_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = dataset[train_size:]
    test_x = Variable(torch.stack([x[0] for x in test_dataset]))
    test_y = Variable(torch.LongTensor([x[1] for x in test_dataset]))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learn_rate)
    for epoch in range(num_epochs):
        print("Epoch %d" % epoch)
        losses = []
        for i_batch, batch in enumerate(train_loader):
            x = Variable(batch['data'])
            y = Variable(batch['label'])
            optimizer.zero_grad()
            output = net(x)
            output = output.view(output.size()[0], output.size()[1])
            loss = criterion(output, y)
            losses.append(loss.data[0])
            loss.backward()
            optimizer.step()

        test_predict = net(test_x)
        test_predict = test_predict.view(test_predict.size()[0], test_predict.size()[1])
        test_loss = criterion(test_predict, test_y)
        avg_train_loss = float(sum(losses)) / len(losses)
        print("Train loss %f, validation loss %f" % (avg_train_loss, test_loss.data[0]))

    net.eval()
    predict = net(test_x).max(1)[1]
    mistakes = sum(predict != test_y)
    mistakes = mistakes.data.view(1)[0]
    print("Error rate: %f" % (float(mistakes / len(test_y))))

    torch.save(net, os.path.join(work_dir, 'net12'))


class Rectangle(object):
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def intersection(self, other):
        dx = min(self.x + self.width, other.x + other.width) - max(self.x, other.x)
        dy = min(self.y + self.height, other.y + other.height) - max(self.y, other.y)
        if (dx >= 0) and (dy >= 0):
            return dx * dy
        else:
            return 0

    def iou(self, other):
        int_area = self.intersection(other)
        union = self.area() + other.area() - int_area
        return float(int_area) / union


class RegionProposal(Rectangle):
    def __init__(self, x, y, size, confidence):
        super(RegionProposal, self).__init__(x, y, size, size)
        self.confidence = confidence


def nms(regions):
    if len(regions) == 0:
        return []

    regions = sorted(regions, key=lambda x: x.confidence, reverse=True)

    x_min = np.array([regions[i].x for i in range(len(regions))], np.float32)
    y_min = np.array([regions[i].y for i in range(len(regions))], np.float32)
    x_max = np.array([regions[i].x + regions[i].width for i in range(len(regions))], np.float32)
    y_max = np.array([regions[i].y + regions[i].height for i in range(len(regions))], np.float32)
    sizes = (x_max-x_min) * (y_max-y_min)

    ids = np.array(range(len(regions)))
    res = []
    while len(ids) > 0:
        i = ids[0]
        res.append(regions[i])

        rightmost_min_x = np.maximum(x_min[i], x_min[ids[1:]])
        lowest_min_y = np.maximum(y_min[i], y_min[ids[1:]])
        leftmost_max_x = np.minimum(x_max[i], x_max[ids[1:]])
        highest_max_y = np.minimum(y_max[i], y_max[ids[1:]])

        w = np.maximum(leftmost_max_x - rightmost_min_x, 0)
        h = np.maximum(highest_max_y - lowest_min_y, 0)

        overlap = (w * h) / (sizes[ids[1:]] + sizes[i] - w * h)
        ids = np.delete(ids, np.concatenate(([0], np.where(((overlap >= 0.5) & (overlap <= 1)))[0] + 1)))

    return res


class EllipseFoldReader(object):
    def __init__(self, path, img_dir):
        self._data = open(path, "r").read().splitlines()
        self._img_dir = img_dir
        self._index = 0

    def is_end(self):
        return self._index >= len(self._data)

    def read_fold(self, folds_dict):
        path = self._translate_ellipse_path(str(self._data[self._index]), self._img_dir)
        num_regions = int(self._data[self._index + 1])
        rectangles = []
        for i in range(num_regions):
            major_radius, minor_radius, _, center_x, center_y, _ = [float(x) for x in self._data[self._index + 2 + i].split()]
            rectangles.append(Rectangle(center_x - minor_radius, center_y - major_radius, minor_radius*2, major_radius*2))
        folds_dict[path] = rectangles
        self._index += 2 + num_regions

    def _translate_ellipse_path(self, ellipse_path, img_dir):
        return os.path.join(img_dir, ellipse_path.replace('/', os.sep)) + '.jpg'

class ImagePyramid(object):
    def __init__(self, img, num_images, downscale):
        self._imgs = [img]
        for i in range(num_images - 1):
            size = [int(x / downscale) for x in img.size]
            if 0 in size:
                break
            img = img.resize(tuple(size))
            self._imgs.append(img)

    def __len__(self):
        return len(self._imgs)

    def __getitem__(self, i):
        return self._imgs[i]

class FaceDetector(object):
    def __init__(self, net, ground_truth_path, img_dir):
        self._net = net
        self._ground_truth = self.read_ground_truth(ground_truth_path, img_dir)

    def read_ground_truth(self, ground_truth_path, img_dir):
        fold_reader = EllipseFoldReader(ground_truth_path, img_dir)
        ground_truth = {}
        while not fold_reader.is_end():
            fold_reader.read_fold(ground_truth)
        return ground_truth

    def visualize_ground_truth(self):
        keys = list(self._ground_truth.keys())
        ind = randint(0, len(keys))
        key = keys[ind]

        img = Image.open(key)
        fig, ax = plt.subplots(1)

        ax.imshow(img)
        for r in self._ground_truth[key]:
            ax.add_patch(patches.Rectangle((r.x, r.y), r.width, r.height, linewidth=1, edgecolor='r', facecolor='none'))
        plt.show()

    def calculate_recall(self):
        mistakes = 0
        num_truths = 0
        num_regions = 0

        res = self.detect_all_images()

        for k in res:
            truths = self._ground_truth[k]
            for truth in truths:
                num_truths += 1
                agreeing_regions = [r for r in res[k] if truth.iou(r) >= iou_threshold]
                if len(agreeing_regions) == 0:
                    mistakes += 1

        print("Mistakes: %d" % mistakes)
        print("Num of truths: %d" % num_truths)
        print("Num of images: %d" % len(self._ground_truth))
        print("Num of region proposals: %d" % num_regions)
        print("Recall: %f" % ((num_truths - mistakes) / num_truths))


    def detect_image(self, image_path, debug=True):
        image = Image.open(image_path)

        res = []
        pyramid = ImagePyramid(image, pyramid_len, pyramid_downscale)
        scale_resize_factor = 1.0
        for img in pyramid:
            resized_img = img.resize((int(x * 12.0 / min_face_size) for x in img.size))
            if min(resized_img.size) < 12:
                break

            pixels = get_image_pixels(resized_img)
            predict = self._net(Variable(pixels.unsqueeze(0)))
            predict = predict[0].data

            regions = []
            for i in range(predict.size()[1]):
                for j in range(predict.size()[2]):
                    max_val, max_index = predict[:, i, j].max(0)
                    if max_index[0] == 1:
                        regions.append(RegionProposal(scale_resize_factor * 2 * i * min_face_size / 12.0,
                                                      scale_resize_factor * 2 * j * min_face_size / 12.0,
                                                      scale_resize_factor * min_face_size, max_val[0]))
            regions = nms(regions)
            res += regions
            scale_resize_factor *= pyramid_downscale

        if debug:
            fig, ax = plt.subplots(1)
            ax.imshow(image)
            for r in res:
                ax.add_patch(
                    patches.Rectangle((r.x, r.y), r.width, r.height, linewidth=1, edgecolor='r', facecolor='none'))
            plt.show()
        return res

    def detect_all_images(self):
        res = {}
        for k in self._ground_truth:
            try:
                regions = self.detect_image(k, debug=False)
                res[k] = regions
            except:
                # TODO: Ask if this is ok
                continue

        return res