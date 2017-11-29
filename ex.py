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

learn_rate = 0.001
num_epochs = 2
batch_size = 32
iou_threshold = 0.5
pyramid_downscale = 1.16
pyramid_len = 15
min_face_size = 30

#TODO: Replace
#work_dir = os.path.dirname(__file__)
work_dir = r'c:\Study\Courses\dl\ex2'

def get_image_pixels(image):
    pixels = torch.from_numpy(np.asarray(image))

    # Reorder from H W C to C W H
    pixels = pixels.permute(2, 0, 1)

    # Convert to float to maintain consistency
    return pixels.float() / 255


def get_image_random_region(image_path, crop_size):
    img = Image.open(image_path)
    max_x = img.size[0] - crop_size - 1
    max_y = img.size[1] - crop_size - 1

    x = randint(0, max_x)
    y = randint(0, max_y)

    img = img.crop((x, y, x + crop_size, y + crop_size))
    return get_image_pixels(img)


def show_tensor_as_image(pixels):
    plt.imshow((pixels.permute(1, 2, 0) * 255).byte().numpy())
    plt.show()


def generate_pascal_dataset(image_dir, crop_size, num_samples, output_path):
    image_list = glob(os.path.join(image_dir, "*"))
    samples = []
    while len(samples) < num_samples:
        for image_path in image_list:
            print('\rGenerating images... (%d\\%d)' % (len(samples), num_samples), end="")
            if len(samples) >= num_samples:
                break
            sample = get_image_random_region(image_path, crop_size)

            # Convert to long to save space
            samples.append((sample * 255).long())

    torch.save(samples, output_path)

def mine_negative_samples(img_dir, face_detector, output_path, sample_size=200000):
    img_list = glob(os.path.join(img_dir, "*"))
    samples = []

    for img_path in img_list:
        if len(samples) >= sample_size:
            break

        print('\rGenerating samples from background images... (%d\\%d)' % (len(samples), sample_size), end="")
        img = Image.open(img_path)

        res = face_detector.detect_image_12(img_path, debug=False)
        for r in res:
            import pdb
            pdb.set_trace()
            crop = img.crop((r.x, r.y, r.x + r.width, r.y + r.height))
            resized_crop = crop.resize((24, 24))
            # Convert to long to save space
            pixels = (get_image_pixels(resized_crop) * 255).long()
            samples.append(pixels)

    torch.save(samples, output_path)


def load_dataset(positive_path, negative_path):
    positive_dataset = load_lua(positive_path)
    negative_dataset = torch.load(negative_path)
    negative_dataset = [x.float() / 255 for x in negative_dataset]
    dataset = [(positive_dataset[k], 1) for k in positive_dataset]
    dataset += [(k, 0) for k in negative_dataset]
    return dataset


class FaceDataset(Dataset):
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


class Net24(nn.Module):
    def __init__(self):
        super(Net24, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.linear1 = nn.Linear(in_features=5184, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=2)
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return self.softmax(x)

    def load_dataset(self, aflw_path, mine_path):
        aflw_dataset = load_lua(aflw_path)
        mine_dataset = torch.load(mine_path)
        dataset = [(aflw_dataset[k], 1) for k in aflw_dataset]
        dataset += [(k, 0) for k in mine_dataset]
        return dataset

def train_net24_temp():
    net = Net24()
    dataset = load_dataset(r'c:\Study\Courses\dl\ex2\EX2_data\aflw\aflw_24.t7', r'c:\Study\Courses\dl\ex2\negative_mine.t7')
    random.shuffle(dataset)
    dataset = dataset[:int(0.1 * len(dataset))]
    train_size = int(len(dataset) * 0.9)

    train_dataset = FaceDataset(dataset[:train_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = dataset[train_size:]
    test_x = Variable(torch.stack([x[0] for x in test_dataset]))
    test_y = Variable(torch.LongTensor([x[1] for x in test_dataset]))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learn_rate)
    for epoch in range(3):
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

    predict = net(test_x).max(1)[1]
    mistakes = sum(predict != test_y)
    mistakes = mistakes.data.view(1)[0]
    print("Error rate: %f" % (float(mistakes / len(test_y))))

    torch.save(net, r'c:\Study\Courses\dl\ex2\net24')

def train_net(net, positive_dataset_path, negative_dataset_path, output_path):
    dataset = load_dataset(positive_dataset_path, negative_dataset_path)
    random.shuffle(dataset)
    train_size = int(len(dataset) * 0.9)

    train_dataset = FaceDataset(dataset[:train_size])
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

    predict = net(test_x).max(1)[1]
    mistakes = sum(predict != test_y)
    mistakes = mistakes.data.view(1)[0]
    print("Error rate: %f" % (float(mistakes / len(test_y))))

    torch.save(net, output_path)


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
    def __init__(self, net12, net24, ground_truth_path, img_dir):
        self._net12 = net12
        self._net24 = net24
        self._ground_truth = self.read_ground_truth(ground_truth_path, img_dir)

    def get_ground_truth(self):
        return self._ground_truth

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

    def calculate_recall(self, full_stack, debug=True):
        mistakes = 0
        num_truths = 0
        num_regions = 0

        res = self.detect_all_images(full_stack)

        for k in res:
            truths = self._ground_truth[k]
            num_regions += len(res[k])
            for truth in truths:
                num_truths += 1
                agreeing_regions = [r for r in res[k] if truth.iou(r) >= iou_threshold]
                if len(agreeing_regions) == 0:
                    mistakes += 1

        if debug:
            print("Mistakes: %d" % mistakes)
            print("Num of truths: %d" % num_truths)
            print("Num of images: %d" % len(self._ground_truth))
            print("Num of region proposals: %d" % num_regions)
            print("Recall: %f" % ((num_truths - mistakes) / num_truths))
        else:
            return (num_truths - mistakes) / num_truths


    def detect_image_12(self, image_path, debug=True):
        image = Image.open(image_path)

        res = []
        #pyramid = ImagePyramid(image, pyramid_len, pyramid_downscale)
        pyramid = tuple(pyramid_gaussian(image, downscale=pyramid_downscale))
        scale_resize_factor = 1.0
        for i in range(pyramid_len):
            img = Image.fromarray(np.uint8(pyramid[i] * 255))
            resized_img = img.resize((int(x * 12.0 / min_face_size) for x in img.size))
            if min(resized_img.size) < 12:
                break

            pixels = get_image_pixels(resized_img)
            predict = self._net12(Variable(pixels.unsqueeze(0)))
            predict = predict[0].data

            regions = []
            approved_idxs = torch.nonzero(predict.max(0)[1][0])
            if len(approved_idxs):
                for idxs in approved_idxs:
                    regions.append(RegionProposal(scale_resize_factor * 2 * idxs[1] * min_face_size / 12.0,
                                                  scale_resize_factor * 2 * idxs[0] * min_face_size / 12.0,
                                                  scale_resize_factor * min_face_size,
                                                  predict[0][idxs[0]][idxs[1]]))
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

    def detect_all_images(self, full_stack):
        res = {}
        num_images = 0
        num_errors = 0
        for k in self._ground_truth:
            try:
                print('\rRunning detector on images... (%d)' % num_images, end="")
                if full_stack:
                    regions = self.detect_image(k, debug=False)
                else:
                    regions = self.detect_image_12(k, debug=False)
                res[k] = regions
                num_images += 1
            except:
                # TODO: Ask if this is ok
                num_errors += 1
                continue

        print('')
        print('Num of errors: %d' % num_errors)
        return res

    def detect_image(self, image_path, debug=True):
        img = Image.open(image_path)

        regions = self.detect_image_12(image_path, debug)
        pixel_regions = []
        for r in regions:
            crop = img.crop((r.x, r.y, r.x + r.width, r.y + r.height))
            resized_crop = crop.resize((24, 24))
            pixels = get_image_pixels(resized_crop)
            pixel_regions.append(pixels)

        predict = self._net24(Variable(torch.stack(pixel_regions)))
        predict = predict.data
        labels = predict.max(1)[1]
        labels = labels.view(-1)

        final_regions = [RegionProposal(regions[i].x, regions[i].y, regions[i].width, predict[i][1]) for i in torch.nonzero(labels).view(-1)]
        #final_regions = nms(final_regions)

        if debug:
            fig, ax = plt.subplots(1)
            ax.imshow(img)
            for r in final_regions:
                ax.add_patch(
                    patches.Rectangle((r.x, r.y), r.width, r.height, linewidth=1, edgecolor='r', facecolor='none'))
            plt.show()
        else:
            return final_regions

    #TODO: Delete
    def calc_ground_truth_recall_24net(self):
        mistakes = 0
        errors = 0
        for k in self._ground_truth:
            for r in self._ground_truth[k]:
                try:
                    img = Image.open(k)
                    crop = img.crop((r.x, r.y, r.x + r.width, r.y + r.height))
                    resized_crop = crop.resize((24, 24))
                    pixels = get_image_pixels(resized_crop)
                    predict = self._net24(Variable(pixels.unsqueeze(0)))
                    predict = predict.data
                    if predict[0][0] > predict[0][1]:
                        mistakes += 1
                except:
                    errors += 1
                    continue
        print("Mistakes: %d" % mistakes)
        print("Errors: %d" % errors)

def test_temp():
    net12 = torch.load(r'c:\Study\Courses\dl\ex2\net12')
    net24 = torch.load(r'c:\Study\Courses\dl\ex2\net24')
    f = FaceDetector(net12, net24, r'c:\Study\Courses\dl\ex2\EX2_data\fddb\FDDB-folds\FDDB-fold-01-ellipseList.txt',
                     r'c:\Study\Courses\dl\ex2\EX2_data\fddb\images')

    min_config = ()
    min_recall = 1
    for pyramid_downscale in [1.06, 1.07, 1.08, 1.09, 1.1, 1.11]:
        for min_face_size in range(25, 35, 1):
            recall = f.calculate_recall(full_stack=False, debug=False)
            if recall < min_recall:
                min_recall = recall
                min_config = (pyramid_downscale, min_face_size)

    print("BEST: %f %f %f" % (min_recall, min_config[0], min_config[1]))
    mine_negative_dataset(f, r'c:\Study\Courses\dl\ex2\negative_mine.t7', 30000)
    train_net24_temp()
    net24 = torch.load(r'c:\Study\Courses\dl\ex2\net24')
    f = FaceDetector(net12, net24, r'c:\Study\Courses\dl\ex2\EX2_data\fddb\FDDB-folds\FDDB-fold-01-ellipseList.txt',
                     r'c:\Study\Courses\dl\ex2\EX2_data\fddb\images')
    f.calculate_recall(full_stack=True)
