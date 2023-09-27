import os
import torch
import time
import json
import pynvml
from backbone import vgg
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import random
import numpy as np
from torchvision import transforms
from network_files import FasterRCNN, AnchorsGenerator
from utils.draw_box_utils import draw_box
import warnings
import cv2
from skimage import io
from CTFNet import *

warnings.filterwarnings("ignore")


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def generate_aps_train():
    model = get_vgg(cfg.device)
    model.to(cfg.device)
    # read class_indict
    for name in os.listdir(cfg.raw_json_root):
        if 'jhu' not in name:
            continue
        print(name)
        img_path = os.path.join(cfg.raw_img_root, name.split('.')[0] + '.jpeg')
        json_path = os.path.join(cfg.raw_json_root, name)
        with open(json_path, 'r')as js:
            js = json.load(js)
            gt_count = js['count']
        original_img = Image.open(img_path)
        # from pil image to tensor, do not normalize image
        data_transform = transforms.Compose([transforms.ToTensor()])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        cfg.train_info['count'] = gt_count
        model.eval()  # 进入验证模式
        with torch.no_grad():
            predictions = model(img.to(cfg.device))[0]
            aps_json_path = os.path.join(cfg.ctf_save_root, name)
            save_dict = {}
            save_dict.update(cfg.aps_dict)
            save_dict.update(cfg.ctf_dict)
            save_dict['count'] = gt_count
            save_dict['name'] = name
            with open(aps_json_path, 'w')as json_file:
                json.dump(save_dict, json_file)


def map_points_to_grid(points, image_shape):
    x_unit = image_shape[0] / 8
    y_unit = image_shape[1] / 8
    points = np.array(points)
    x_ctr = points[:, 0]
    y_ctr = points[:, 1]
    pos = y_ctr // y_unit * 8 + x_ctr // x_unit
    pos = pos.astype('int32')
    box_count = len(points)
    grid_map = np.zeros(64)
    for i in range(box_count):
        grid_map[pos[i]] += 1
    assert np.sum(grid_map) == len(points)
    return grid_map.astype('int32').tolist()


def generate_point_train():
    cfg.use_ctf = False
    model = get_vgg(cfg.device)
    model.to(cfg.device)
    # read class_indict
    json_list = [name for name in os.listdir(cfg.point_json_root)]
    json_list = sorted(json_list)
    fail_list = []
    for name in json_list:
        print(name, fail_list)
        img_path = os.path.join(cfg.raw_img_root, name.split('.')[0] + '.jpeg')
        json_path = os.path.join(cfg.point_json_root, name)
        with open(json_path, 'r')as js:
            js = json.load(js)
            gt_count = js['count']
            gt_points = js['points']
        original_img = Image.open(img_path)
        image_shape = original_img.size
        # from pil image to tensor, do not normalize image
        data_transform = transforms.Compose([transforms.ToTensor()])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        cfg.train_info['count'] = gt_count
        model.eval()  # 进入验证模式
        with torch.no_grad():
            try:
                predictions = model(img.to(cfg.device))[0]
                aps_json_path = os.path.join(cfg.point_save_root, name)
                save_dict = {}
                save_dict.update(cfg.aps_dict)
                save_dict.update(cfg.ctf_dict)
                save_dict['count'] = gt_count
                save_dict['name'] = name
                save_dict['count64'] = js['count64']
                with open(aps_json_path, 'w')as json_file:
                    json.dump(save_dict, json_file)
            except:
                fail_list.append(name)


def create_model(num_classes):
    vgg_feature = vgg(model_name="vgg16").features
    backbone = torch.nn.Sequential(*list(vgg_feature._modules.values())[:-1])  # 删除features中最后一个Maxpool层
    backbone.out_channels = 512
    anchor_generator = AnchorsGenerator(sizes=((4, 8, 16, 32, 64),),
                                        aspect_ratios=((1.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],  # 在哪些特征层上进行roi pooling
                                                    output_size=[7, 7],  # roi_pooling输出特征矩阵尺寸
                                                    sampling_ratio=2)  # 采样率
    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    return model


def get_vgg(device):
    model = create_model(2)
    checkpoint = r'./backbone/vgg-model-NJ-21.pth'
    checkpoint = torch.load(checkpoint, map_location=device)["model"]
    model.load_state_dict(checkpoint, strict=False)
    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def vis_feature(img_path):
    # img = cv2.imread(img_path)
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('-2.png',gray)
    # return
    threshes = [0.0, 0.1, 0.2, 0.3, 0.4]
    device = torch.device("cuda:0")
    model = get_vgg(device)
    model.to(device).eval()
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(img).unsqueeze(dim=0)

    with torch.no_grad():
        predictions = model(img.to(device))[0]
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()
        predict_boxes = predictions["boxes"].to("cpu").numpy()
        areas = (predict_boxes[:, 2] - predict_boxes[:, 0]) * (predict_boxes[:, 3] - predict_boxes[:, 1])
        sizes = np.sqrt(areas)
        predict_num = len(predict_boxes)
        print(predict_num)
        for t in threshes:
            show_img = np.ones((h, w)) * 0
            for i in range(len(predict_boxes)):
                score = predict_scores[i]
                if score < t:
                    continue
                xmin, ymin, xmax, ymax = predict_boxes[i]
                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                show_img[ymin:ymax, xmin:xmax] = 255
            show_img = np.uint8(show_img)
            cv2.imwrite(str(t) + '.png', show_img)
        layers = cfg.aps_dict['pyramid']
        factor = 20
        for i in range(5):
            layer = layers[i]
            vis_img = np.zeros((cfg.y_grid * factor, cfg.x_grid * factor))
            for j in range(cfg.y_grid):
                for k in range(cfg.x_grid):
                    vis_img[j * factor:(j + 1) * factor, k * factor:(k + 1) * factor] = layer[j][k]
            vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
            vis_img = (vis_img * 255).astype(np.uint8)
            # vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
            cv2.imwrite(str(i) + '.png', vis_img)
        box_score = cfg.aps_dict['box_score']
        box_score = np.array(box_score)
        box_score = (box_score - box_score.min()) / (box_score.max() - box_score.min() + 1e-5)
        box_score = np.expand_dims(box_score,0)
        box_score = np.repeat(box_score,50,axis=0)
        box_score = (box_score * 255).astype(np.uint8)
        cv2.imwrite('-1.png', box_score)

        # for i in range(len(predict_boxes)):
        #     score = predict_scores[i]
        #     xmin, ymin, xmax, ymax = predict_boxes[i]
        #     size = int(round(sizes[i] / 50))
        #     cv2.putText(show_img, format(score, '.2f'), (int(round(xmin)), int(round((ymin + ymax) / 2))),
        #                 cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 2)
        # cv2.imwrite('3.png', show_img)
        # show_img = Image.fromarray(np.uint8(show_img))
        # show_img.save('2.png')
        # io.imsave(os.path.join('1.png'), show_img.astype(np.uint8))


def detect_one_image(img_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    model = get_vgg(device)
    model.to(device)
    original_img = Image.open(img_path).convert('RGB')
    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    h, w = img.shape[-2:]
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    plt.axis("off")
    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        # init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        # model(init_img)
        predictions = model(img.to(device))[0]
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()
        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_num = len(predict_boxes)
        count = cfg.test_info['count']
        dpi = round(100 / (img_width / 1000))
        plt.figure(figsize=(img_width / 200, img_height / 200), dpi=dpi)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        w, h = original_img.size
        img = np.zeros((h, w))
        size = 4
        areas = (predict_boxes[:, 2] - predict_boxes[:, 0]) * (predict_boxes[:, 3] - predict_boxes[:, 1]) / (w * h)
        x_ctr = (predict_boxes[:, 0] + predict_boxes[:, 2]) / 2
        y_ctr = (predict_boxes[:, 1] + predict_boxes[:, 3]) / 2
        w_unit = w / size
        h_unit = h / size
        pos = (y_ctr // h_unit) * size + x_ctr // w_unit
        pos = pos.astype('int32').tolist()
        area_list = [0 for i in range(size ** 2)]
        for i in range(len(predict_boxes)):
            xmin, ymin, xmax, ymax = predict_boxes[i]
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            img[ymin:ymax, xmin:xmax] = 1
            area_list[pos[i]] += areas[i]
        io.imsave(os.path.join('mask.png'), (img * 255).astype(np.uint8))
        print(area_list)

        draw_box(original_img,
                 predict_boxes,
                 predict_classes,
                 predict_scores,
                 {'person': 1},
                 thresh=0,
                 line_thickness=3)
        font = {
            'family': 'Segoe UI',
            'style': 'normal',
            'weight': 'bold',
            'color': 'white',
            'size': img_width * 0.055 * 0.5
        }
        w, h = original_img.size
        original_img = original_img.resize((int(w / 2), int(h / 2))).convert('RGB')
        original_img.save('result.jpg')
        txt_len = len(str(count))
        offset = 0.98 - txt_len * 0.02
        # plt.text(img_width * offset, img_height * 0.98, count, horizontalalignment='center', fontdict=font)
        plt.imshow(original_img)
        # plt.savefig('raw.png')
        plt.show()


def evaluation(img_root, json_root):
    if cfg.mode is not "test":
        print("only in test mode can this function work properly!")
        return
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    base_consume = meminfo.used
    print(base_consume)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    # create model
    model = get_vgg(device)
    model.to(device)
    model.eval()
    img_count = 0
    total_mae = 0
    total_mse = 0
    save_info = []
    json_list = [name for name in os.listdir(json_root)]
    json_list = sorted(json_list)
    for name in json_list:
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        img_count += 1
        name = name.split('.')[0]
        name = name.strip()
        img_path = os.path.join(img_root, name + '.jpeg')
        json_path = os.path.join(json_root, name + '.json')
        with open(json_path, 'r')as js:
            js = json.load(js)
            gt_count = js['count']
        original_img = Image.open(img_path)
        # from pil image to tensor, do not normalize image
        data_transform = transforms.Compose([transforms.ToTensor()])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            # init
            img_height, img_width = img.shape[-2:]
            # init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            # model(init_img)
            st = time.time()
            predictions = model(img.to(device))[0]
            et = time.time()
            detect_count = cfg.test_info['count']
            mae = abs(detect_count - gt_count)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem = meminfo.used
            total_mae += mae
            total_mse += mae ** 2
            dic = {'id': name, 'mem': mem - base_consume, 'time': et - st, 'mae': mae, 'gt_count': gt_count,
                   'detect_count': detect_count}
            print(name, total_mae / img_count, (total_mse / img_count) ** 0.5)
            save_info.append(dic)
            if cfg.visualize is True:
                predict_classes = predictions["labels"].to("cpu").numpy()
                predict_scores = predictions["scores"].to("cpu").numpy()
                predict_boxes = predictions["boxes"].to("cpu").numpy()
                dpi = round(100 / (img_width / 1000))
                plt.figure(figsize=(img_width / 200, img_height / 200), dpi=dpi)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)
                draw_box(original_img,
                         predict_boxes,
                         predict_classes,
                         predict_scores,
                         {'person': 1},
                         thresh=0,
                         line_thickness=round(img_width / 250))
                font = {
                    'family': 'Segoe UI',
                    'style': 'normal',
                    'weight': 'bold',
                    'color': 'white',
                    'size': img_width * 0.055 * 0.5
                }
                txt_len = len(str(detect_count))
                offset = 0.98 - txt_len * 0.02
                plt.text(img_width * offset, img_height * 0.98, detect_count, horizontalalignment='center',
                         fontdict=font)
                plt.imshow(original_img)
                plt.show()
    with open('log.json', 'w')as new_json_file:
        json.dump(save_info, new_json_file)


if __name__ == '__main__':
    # detect_one_image(img_path=os.path.join(cfg.raw_img_root, 'jhu_1151.jpeg'))
    evaluation(img_root=cfg.qnrf_img, json_root=cfg.qnrf_json)
