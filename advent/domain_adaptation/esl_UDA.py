import os
import os.path as osp
import time
from PIL import Image

import multiprocessing

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from advent.utils.func import per_class_iu, fast_hist
from advent.utils.serialization import pickle_dump, pickle_load

def extract_pseudo_labels(model, test_loader, cfg,
                                verbose=True):
    if cfg.PSEUDO_LABEL_DIRECTORY_TARGET is '':
        raise NotImplementedError(f"No pseudo-label directory provided.")

    if not os.path.exists(cfg.PSEUDO_LABEL_DIRECTORY_TARGET):
        os.makedirs(cfg.PSEUDO_LABEL_DIRECTORY_TARGET)

    device = cfg.GPU_ID
    interp = nn.Upsample(size=(cfg.ESL.OUTPUT_SIZE_TARGET[1], cfg.ESL.OUTPUT_SIZE_TARGET[0]),
                         mode='bilinear', align_corners=True)
    # eval
    extract_labels(cfg, model,
                device, test_loader, interp,
                verbose)

def compute_entropy(arr, num_classes):
    tensor = torch.from_numpy(arr)
    predicted_entropy = torch.sum(torch.mul(tensor,torch.log(tensor)),dim=2) * (-1/np.log(num_classes))
    return predicted_entropy.numpy()

def extract_labels(cfg, model,
                device, test_loader, interp,
                verbose):

    max_length = min(cfg.ESL.MAX_MEDIAN_BATCH, len(test_loader))

    predicted_label = np.zeros((min(len(test_loader),max_length), cfg.ESL.OUTPUT_SIZE_TARGET[1], cfg.ESL.OUTPUT_SIZE_TARGET[0]))
    predicted_entropy = np.zeros((min(len(test_loader),max_length), cfg.ESL.OUTPUT_SIZE_TARGET[1], cfg.ESL.OUTPUT_SIZE_TARGET[0]))

    image_name = []

    load_checkpoint_for_evaluation(model, cfg.ESL.RESTORE_FROM, device)

    for index, batch in tqdm(enumerate(test_loader)):
        image, _, _, name = batch
        with torch.no_grad():
            _, pred_main = model(image.cuda(device))
            output = torch.nn.functional.softmax(pred_main, dim=1)
            output = interp(output).cpu().data[0].numpy()
            output = output.transpose(1, 2, 0)
            label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
            predicted_label[index % max_length] = label.copy()
            predicted_entropy[index % max_length] = compute_entropy(output,cfg.NUM_CLASSES)
            image_name.append(name[0])

            if (index + 1) % max_length == 0:
                thres = []
                for i in range(cfg.NUM_CLASSES):
                    x = predicted_entropy[predicted_label==i]
                    if len(x) == 0:
                        thres.append(0)
                        continue
                    x = np.sort(x)
                    thres.append(x[np.int(np.round(len(x)*0.5))])
                thres = np.array(thres)
                thres[thres<cfg.ESL.THRESHOLD]=cfg.ESL.THRESHOLD
                for ind in range(len(image_name)):
                    name = image_name[ind]
                    label = predicted_label[ind]
                    entropy = predicted_entropy[ind]
                    for i in range(cfg.NUM_CLASSES):
                        label[(entropy>thres[i])*(label==i)] = 255
                    output = np.asarray(label, dtype=np.uint8)
                    output = Image.fromarray(output)
                    name = name.split('/')[-1]
                    output.save('%s/%s' % (cfg.ESL.OUTPUT_DIR, osp.splitext(name)[0] + ".png"), "PNG")
                thres = []
                image_name = []
                predicted_label = np.zeros((min(len(test_loader),max_length), cfg.ESL.OUTPUT_SIZE_TARGET[1], cfg.ESL.OUTPUT_SIZE_TARGET[0]))
                predicted_entropy = np.zeros((min(len(test_loader),max_length), cfg.ESL.OUTPUT_SIZE_TARGET[1], cfg.ESL.OUTPUT_SIZE_TARGET[0]))
    thres = []
    for i in range(cfg.NUM_CLASSES):
        x = predicted_entropy[predicted_label==i]
        if len(x) == 0:
            thres.append(0)
            continue
        x = np.sort(x)
        thres.append(x[np.int(np.round(len(x)*0.5))])
    thres = np.array(thres)
    thres[thres<cfg.ESL.THRESHOLD]=cfg.ESL.THRESHOLD
    for index in range(len(image_name)):
        name = image_name[index]
        label = predicted_label[index]
        entropy = predicted_entropy[index]
        for i in range(cfg.NUM_CLASSES):
            label[(entropy>thres[i])*(label==i)] = 255
        output = np.asarray(label, dtype=np.uint8)
        output = Image.fromarray(output)
        name = name.split('/')[-1]
        output.save('%s/%s' % (cfg.ESL.OUTPUT_DIR, osp.splitext(name)[0]) + ".png", "PNG")


def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda(device)
