import os
import os.path as osp
import glob

import cv2
import numpy as np

import torch
from torch.utils import data
from torchvision import transforms as T
from tps_transformation import tps_transform
from PIL import Image
from utils import elastic_transform, plt
from albumentations import *
from imageio import imread

MEAN = np.array([0.67, 0.55, 0.74])
STD = np.array([0.16, 0.20, 0.17])


class DataSet(data.Dataset):

    def __init__(self, config, img_transform_gt, img_transform_sketch):
        self.img_transform_gt = img_transform_gt
        self.img_transform_sketch = img_transform_sketch
        self.img_dir = osp.join(config['TRAINING_CONFIG']['IMG_DIR'], config['TRAINING_CONFIG']['MODE'])
        self.img_size = (config['MODEL_CONFIG']['IMG_SIZE'], config['MODEL_CONFIG']['IMG_SIZE'], 3)

        self.data_list = glob.glob(os.path.join(self.img_dir, '*.png'))
        self.data_list = [x.split(os.sep)[-1].split('_')[0] for x in self.data_list]
        self.data_list = list(set(self.data_list))
        # random.seed(config['TRAINING_CONFIG']['CPU_SEED'])

        self.augment = config['TRAINING_CONFIG']['AUGMENT']

        self.dist = config['TRAINING_CONFIG']['DIST']
        if self.dist == 'uniform':
            self.a = config['TRAINING_CONFIG']['A']
            self.b = config['TRAINING_CONFIG']['B']
        else:
            self.mean = config['TRAINING_CONFIG']['MEAN']
            self.std = config['TRAINING_CONFIG']['STD']

    def __getitem__(self, index):
        fid = self.data_list[index]
        reference = Image.open(osp.join(self.img_dir, '{}_color.png'.format(fid))).convert('RGB')
        sketch = Image.open(osp.join(self.img_dir, '{}_sketch.png'.format(fid))).convert('L')

        if self.dist == 'uniform':
            noise = np.random.uniform(self.a, self.b, np.shape(reference))
        else:
            noise = np.random.normal(self.mean, self.std, np.shape(reference))

        reference = np.array(reference) + noise
        reference = Image.fromarray(reference.astype('uint8'))

        if self.augment == 'elastic':
            augmented_reference = elastic_transform(np.array(reference), 1000, 8, random_state=None)
            augmented_reference = Image.fromarray(augmented_reference)
        elif self.augment == 'tps':
            augmented_reference = tps_transform(np.array(reference))
            augmented_reference = Image.fromarray(augmented_reference)
        else:
            augmented_reference = reference

        return fid, self.img_transform_gt(augmented_reference), self.img_transform_gt(
            reference), self.img_transform_sketch(sketch)

    def __len__(self):
        """Return the number of images."""
        return len(self.data_list)


def find_bbox(binary_mask):
    maskx = np.any(binary_mask, axis=0)
    masky = np.any(binary_mask, axis=1)
    x1 = np.argmax(maskx)
    y1 = np.argmax(masky)
    x2 = len(maskx) - np.argmax(maskx[::-1])
    y2 = len(masky) - np.argmax(masky[::-1])
    sub_image = binary_mask[y1:y2, x1:x2]

    # h,w = y2-y1,x2-x1
    return sub_image


def generate_mask(Epithelial_number=2, Macrophage_number=2, Neutrophil_number=2, Lymphocyte_number=2,
                  height=512, width=512, easy=False, patient_id=None, patient_case_id=None):
    root = '/home/pzsuen/Code/I2I/CoCosNet/datasets/nuclear_all/cell_masks'
    classes = {'Epithelial': 1, 'Macrophage': 2, 'Neutrophil': 3, 'Lymphocyte': 4}

    # easy 就根据原细胞生成mask;否则，随机生成
    full_mask = np.zeros(shape=(height, width))
    cell_number = {'Epithelial': Epithelial_number, 'Macrophage': Macrophage_number,
                   'Neutrophil': Neutrophil_number, 'Lymphocyte': Lymphocyte_number}

    if easy:
        patient_case_root = os.path.join(root, patient_id, patient_case_id)

        # 记录所有细胞路径
        cell_paths = {}
        for key, value in classes.items():
            fname_list = glob.glob(os.path.join(patient_case_root, key, '*.png'))
            cell_paths[key] = fname_list

        for key, value in classes.items():
            cell_number_ = cell_number[key]
            fname_list = cell_paths[key]
            # print(cell_number_)

            for i in range(len(fname_list)):
                # print(i)
                cell_fname = fname_list[i]
                bmask = imread(cell_fname)

                cell = find_bbox(bmask) / 255

                cell_h, cell_w = cell.shape
                if cell_h < height and cell_w < width:
                    tl_y, tl_x = np.random.randint(0, height - cell_h), np.random.randint(0, width - cell_w)
                    select_region = full_mask[tl_y:tl_y + cell_h, tl_x:tl_x + cell_w]

                    # 判断该区域是否已有细胞
                    iter_times = 0
                    # 和大于零说明有细胞
                    while np.max(select_region) > 0:
                        # plt.subplot(121)
                        # plt.imshow(cell)
                        # plt.title(np.max(cell))
                        # plt.subplot(122)
                        # plt.imshow(select_region)
                        # plt.title(np.max(select_region))
                        # plt.show()

                        tl_y, tl_x = np.random.randint(0, height - cell_h), np.random.randint(0, width - cell_w)
                        select_region = full_mask[tl_y:tl_y + cell_h, tl_x:tl_x + cell_w]
                        iter_times += 1

                        if iter_times > 25:
                            break
                    if iter_times < 25:
                        full_mask[tl_y:tl_y + cell_h, tl_x:tl_x + cell_w] = cell * int(value)
                else:
                    continue

                # full_mask[tl_y:tl_y + cell_h, tl_x:tl_x + cell_w] = cell * int(value)
                # plt.imshow(full_mask)
                # plt.axis('off')
                # plt.show()


    else:
        # 记录所有细胞路径
        cell_paths = {}
        for key, value in classes.items():
            fname_list = glob.glob(os.path.join(root, '**/**', key, '*.png'))
            cell_paths[key] = fname_list

        # 对每个类别粘贴细胞
        for key, value in classes.items():
            cell_number_ = cell_number[key]
            fname_list = cell_paths[key]
            # print(cell_number_)
            for i in range(cell_number_):
                # print(i)
                # 随机选择一个细胞
                idx = np.random.randint(0, len(fname_list))
                bmask = imread(fname_list[idx])

                cell = find_bbox(bmask) / 255

                cell_h, cell_w = cell.shape
                if cell_h < height and cell_w < width:
                    tl_y, tl_x = np.random.randint(0, height - cell_h), np.random.randint(0, width - cell_w)
                    select_region = full_mask[tl_y:tl_y + cell_h, tl_x:tl_x + cell_w]

                    # 判断该区域是否已有细胞
                    iter_times = 0
                    while np.max(select_region) > 0:
                        tl_y, tl_x = np.random.randint(0, height - cell_h), np.random.randint(0, width - cell_w)
                        select_region = full_mask[tl_y:tl_y + cell_h, tl_x:tl_x + cell_w]
                        iter_times += 1

                        if iter_times > 25:
                            break
                    if iter_times < 25:
                        full_mask[tl_y:tl_y + cell_h, tl_x:tl_x + cell_w] = cell * int(value)

                else:
                    continue

    return full_mask


class NuclearData(data.Dataset):
    def __init__(self, config, root, HW):
        super(NuclearData, self).__init__()
        self.root = root
        self.image_root = os.path.join(root, 'image_normalized')
        self.real_mask_root = os.path.join(root, 'mask')
        self.fake_mask_root = os.path.join(root, 'mask_generated')

        image_list = glob.glob(os.path.join(self.image_root, '*.png'))

        self.images = image_list

        self.hw = HW

        self.hard_ratio = 0.5
        self.gen_num = 2

        self.augment = config['TRAINING_CONFIG']['AUGMENT']

        self.dist = config['TRAINING_CONFIG']['DIST']
        if self.dist == 'uniform':
            self.a = config['TRAINING_CONFIG']['A']
            self.b = config['TRAINING_CONFIG']['B']
        else:
            self.mean = config['TRAINING_CONFIG']['MEAN']
            self.std = config['TRAINING_CONFIG']['STD']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_fpath = self.images[idx]

        basename = os.path.basename(image_fpath)
        # 有的case名称用的不是下划线，而是-，如 TCGA-KK-A59X-01Z-00-DX1-3
        patient_case_id = basename.split('.')[0]
        patient_id = patient_case_id.split('_')[0]
        patient_root = os.path.join(self.root, 'cell_masks', patient_id)
        if not os.path.exists(patient_root):
            patient_id = '-'.join(patient_case_id.split('-')[:-1])

            patient_root = os.path.join(self.root, 'cell_masks', patient_id)

        image = np.array(imread(image_fpath)) / 255.
        mask = np.array(imread(os.path.join(self.real_mask_root, basename)))

        classes = {'Epithelial': 1, 'Macrophage': 2, 'Neutrophil': 3, 'Lymphocyte': 4}
        cell_number = {}
        for key, value in classes.items():
            cell_number[key] = len(glob.glob(os.path.join(patient_root, patient_case_id, key, "*.png")))

        # ori_image = imread(image_fpath)
        # mask = imread(os.path.join(self.real_mask_root, basename))
        # ori_image = image
        # print(image.shape, mask.shape)

        # 随机选择fake mask
        # easy = True if np.random.random(1) < 0.7 else False
        # fake_mask_path = 'easy' if easy else 'hard'
        # fake_mask_list = glob.glob(os.path.join(self.fake_mask_root, patient_case_id, fake_mask_path, '*.png'))
        # idx = np.random.randint(0, fake_mask_list.__len__())
        # fake_mask = np.array(imread(fake_mask_list[idx]))

        # if self.augment == 'elastic':
        #     augmented_mask = elastic_transform(mask, 1000, 8, random_state=None)
        #     augmented_image = elastic_transform(image, 1000, 8, random_state=None)
        #
        #     # augmented_mask = Image.fromarray(augmented_mask)
        # elif self.augment == 'tps':
        #     augmented_mask = tps_transform(mask)
        #     augmented_image = tps_transform(image)
        #     # augmented_mask = Image.fromarray(augmented_mask)
        # else:
        #     augmented_mask = mask
        #     augmented_image = image

        # 数据扩增

        h, w = mask.shape
        # hard
        is_easy = np.random.randn() < 0.9

        # print(is_hard)
        # for i in range(round(self.gen_num * self.hard_ratio)):
        if is_easy:
            gen_mask = generate_mask(cell_number['Epithelial'], cell_number['Macrophage'], cell_number['Neutrophil'],
                                     cell_number['Lymphocyte'], height=h, width=w, easy=True,
                                     patient_id=patient_id, patient_case_id=patient_case_id)
            # save_dist = os.path.join(self.root, 'mask_generated', patient_case_id, 'hard')
            # if not os.path.exists(save_dist):
            #     os.makedirs(save_dist)
            #
            # cv2.imwrite(os.path.join(save_dist, str(i).zfill(3) + '.png'), gen_mask)
            #
            # plt.imshow(gen_mask)
            # plt.title('hard')
            # plt.show()

        # sample
        else:
            # for i in range(round(self.gen_num - self.gen_num * self.hard_ratio)):
            gen_mask = generate_mask(cell_number['Epithelial'], cell_number['Macrophage'], cell_number['Neutrophil'],
                                     cell_number['Lymphocyte'], height=h, width=w, easy=False,
                                     patient_id=patient_id, patient_case_id=patient_case_id)

            # save_dist = os.path.join(self.root, 'mask_generated', patient_case_id, 'easy')
            # if not os.path.exists(save_dist):
            #     os.makedirs(save_dist)

            # cv2.imwrite(os.path.join(save_dist, str(i).zfill(3) + '.png'), gen_mask)

            # plt.imshow(gen_mask)
            # plt.title('sample')
            # plt.show()

        transformer = self.get_aug()
        ori_augmented = transformer(image=image, mask=mask, mask2=gen_mask)
        image = ori_augmented['image']
        mask = ori_augmented['mask']
        gen_mask = ori_augmented['mask2']
        # plt.subplot(121)
        # plt.imshow(image)
        # plt.subplot(122)
        # plt.imshow(mask)
        # plt.show()
        # fake_mask = ori_augmented['mask2']

        # aug_augmented = transformer(image=augmented_image, mask=augmented_mask)
        # aug_image = aug_augmented['image']
        # aug_mask = aug_augmented['mask']

        image = np.transpose(image, (2, 0, 1))  # / 255.
        mask = mask[np.newaxis, :]
        gen_mask = gen_mask[np.newaxis, :]

        # aug_image = np.transpose(aug_image, (2, 0, 1))  # / 255.
        # aug_mask = aug_mask[np.newaxis, :]

        # gen_mask = generate_mask(cell_number['Epithelial'], cell_number['Macrophage'], cell_number['Neutrophil'],
        #                          cell_number['Lymphocyte'], height=h, width=w, easy=False,
        #                          patient_id=patient_id, patient_case_id=patient_case_id)
        # print(np.max(mask))
        image, mask, gen_mask = torch.from_numpy(image), preprocess_label(torch.tensor(mask, dtype=torch.int64)), \
                                preprocess_label(torch.tensor(gen_mask, dtype=torch.int64))
        # aug_image, aug_mask = torch.from_numpy(aug_image), preprocess_label(torch.tensor(aug_mask, dtype=torch.int64))

        # plt.subplot(141)
        # plt.imshow(image)
        # plt.subplot(142)
        # plt.imshow(mask)
        # plt.subplot(143)
        # plt.imshow(aug_image)
        # plt.subplot(144)
        # plt.imshow(aug_mask)
        # plt.show()
        if np.random.randn() < 0.5:
            return basename, is_easy, mask, gen_mask, image
        else:
            return basename, is_easy, mask, mask, image

    def get_aug(self, p=1.0):
        # return Compose([Resize(self.hw, self.hw)], p=p)
        return Compose([
            OneOf([
                HorizontalFlip(),
                VerticalFlip(),
                RandomRotate90(),
            ]),
            OneOf([
                OpticalDistortion(p=0.3),
                GridDistortion(p=.1),
                IAAPiecewiseAffine(p=0.3),
                ElasticTransform(),
            ], p=0.5),
            # OneOf([
            #     # HueSaturationValue(10, 15, 10),
            #     # CLAHE(clip_limit=2),
            #     # RandomBrightnessContrast(),
            # ], p=0.3),
            Resize(self.hw, self.hw),
            # RandomCrop(self.hw, self.hw),
            Normalize(mean=MEAN, std=STD, max_pixel_value=1.0)
        ],
            additional_targets={'mask2': 'mask'},
            p=p)


class RandomData(data.Dataset):
    def __init__(self, image_size):
        super(RandomData, self).__init__()
        self.lable_nc = 5  # label_nc是包含背景的
        self.image_size = image_size

    def __getitem__(self, item):
        real_image = torch.randn(size=(3, self.image_size, self.image_size))
        real_mask = preprocess_label(torch.randint(0, 4, size=(self.image_size, self.image_size)).unsqueeze(0))
        fake_mask = preprocess_label(torch.randint(0, 4, size=(self.image_size, self.image_size)).unsqueeze(0))
        fid = str(item).zfill(8)
        # return {"basename": str(item).zfill(8), "image": real_image, "mask": real_mask, "gen_mask": fake_mask}
        return fid, real_image, real_mask, fake_mask

    def __len__(self):
        return 10000


def preprocess_label(label_map, lable_nc=5):
    """create one-hot label map
      将不同的类别映射到不同的channel上
     读取的label map是单通道的，背景是0，类别一是1，类别二是2……
    """
    _, h, w = label_map.size()
    input_label = torch.zeros(size=(lable_nc, h, w))
    # 对于该图像不存在的类别的那一个channel会是全0
    input_semantics = input_label.scatter_(0, label_map, 1.0)  # (dim,index,src)

    return input_semantics


def postprocess_label(onehot_content):
    c, w, h = onehot_content.shape
    result = torch.zeros(size=(w, h))

    for i in range(c):
        result += onehot_content[i] * i

    return result


def get_loader(config):
    root = config['TRAINING_CONFIG']['IMG_DIR']

    # transform_img = list()
    # transform_mask = list()
    img_size = config['MODEL_CONFIG']['IMG_SIZE']

    # transform_img.append(T.Resize((img_size, img_size)))
    # transform_img.append(T.ToTensor())
    # transform_img.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    # transform_img = T.Compose(transform_img)

    # transform_mask.append(T.Resize((img_size, img_size)))
    # transform_mask.append(T.ToTensor())
    # transform_mask.append(T.Normalize(mean=(0.5), std=(0.5)))
    # transform_mask = T.Compose(transform_mask)

    # dataset = RandomData(img_size)
    dataset = NuclearData(config, root, HW=img_size)

    # for d in dataset:
    #     print(d)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config['TRAINING_CONFIG']['BATCH_SIZE'],
                                  shuffle=(config['TRAINING_CONFIG']['MODE'] == 'train'),
                                  num_workers=config['TRAINING_CONFIG']['NUM_WORKER'],
                                  drop_last=True)
    return data_loader


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_m2n.yml', help='specifies config yaml file')

    params = parser.parse_args()

    # if os.path.exists(params.config):
    config = yaml.load(open(params.config, 'r'), Loader=yaml.FullLoader)

    print(config)
    dl = get_loader(config)
