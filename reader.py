import numpy as np
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import torch
import json
import random
import copy


# randomly move the bounding box around
def aug_line(line, width, height):
    bbox = np.array(line[2:5])
    bias = round(30 * random.uniform(-1, 1))
    bias = max(np.max(-bbox[0, [0, 2]]), bias)
    bias = max(np.max(-2 * bbox[1:, [0, 2]] + 0.5), bias)

    line[2][0] += int(round(bias))
    line[2][1] += int(round(bias))
    line[2][2] += int(round(bias))
    line[2][3] += int(round(bias))

    line[3][0] += int(round(0.5 * bias))
    line[3][1] += int(round(0.5 * bias))
    line[3][2] += int(round(0.5 * bias))
    line[3][3] += int(round(0.5 * bias))

    line[4][0] += int(round(0.5 * bias))
    line[4][1] += int(round(0.5 * bias))
    line[4][2] += int(round(0.5 * bias))
    line[4][3] += int(round(0.5 * bias))

    line[5][2] = line[2][2] / width
    line[5][3] = line[2][0] / height

    line[5][6] = line[3][2] / width
    line[5][7] = line[3][0] / height

    line[5][10] = line[4][2] / width
    line[5][11] = line[4][0] / height
    return line


class loader(Dataset):

    def __init__(self, data_path, data_type):
        self.lines = []
        self.labels = {}
        self.data_path = data_path
        self.data_type = data_type
        subjects = os.listdir(data_path)
        subjects.sort()
        for subject in subjects:
            subject_path = os.path.join(data_path, subject)
            if ((not os.path.isdir(subject_path)) or subject == '01185' or subject == '01730' or subject == '02065'):
                continue
            info = open(os.path.join("Data/label/", subject + ".label"), "r")
            # current_data_type = info_json["Dataset"]
            # device_name = info_json["DeviceName"]
            self.labels[subject] = open(os.path.join("Data/label/", subject + ".label"), "r")
            # if not current_data_type == data_type:
            #      continue
            #
            # # test on iPhone datas only
            # if data_type=="test" and device_name[:6]!="iPhone":
            #      continue
            # name_file = open(os.path.join(subject_path, "newFaceLdmk.json"), "r")
            # temp = json.load(name_file)

            self.lines = self.lines + info.readlines()
            if (len(self.lines) >= 150000):
                break

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        # this is no longer accurate
        #   0     1     2      3         4       5      
        # Face  Left  Right  Rects     2DGaze  Label

        line = self.lines[idx + 1]

        line = line.split(" ")

        face_path = os.path.join(self.data_path, line[0])
        left_path = os.path.join(self.data_path, line[1])
        right_path = os.path.join(self.data_path, line[2])

        #frame = line[1]
        #item_index = int(frame[:-4])

        # face
        face_img = cv2.imread(face_path)
        face_img = np.array(face_img)
        height = face_img.shape[0]
        width = face_img.shape[1]

        # left eye
        leftEye_img= cv2.imread(left_path)
        leftEye_img = np.array(leftEye_img)
        leftEye_height = leftEye_img.shape[0]
        leftEye_width = leftEye_img.shape[1]

        # right eye
        rightEye_img = cv2.imread(right_path)
        rightEye_img = np.array(rightEye_img)
        rightEye_height = rightEye_img.shape[0]
        rightEye_width = rightEye_img.shape[1]


        #origin = copy.deepcopy(line)
        # print(line[2])
        # if not (self.data_type == 'test'):
        #     line = aug_line(copy.deepcopy(line), width, height)

        # face_img = img[line[2][0]:line[2][1], line[2][2]:line[2][3]]
        # leftEye_img = img[line[3][0]:line[3][1], line[3][2]:line[3][3]]
        # rightEye_img = img[line[4][0]:line[4][1], line[4][2]:line[4][3]]

        face_img = cv2.resize(face_img, (224, 224))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = face_img / 255
        face_img = face_img.transpose(2, 0, 1)

        leftEye_img = cv2.resize(leftEye_img, (112, 112))
        leftEye_img = cv2.cvtColor(leftEye_img, cv2.COLOR_BGR2RGB)
        leftEye_img = leftEye_img / 255
        leftEye_img = leftEye_img.transpose(2, 0, 1)

        rightEye_img = cv2.resize(rightEye_img, (112, 112))
        rightEye_img = cv2.cvtColor(rightEye_img, cv2.COLOR_BGR2RGB)
        rightEye_img = cv2.flip(rightEye_img, 1)
        rightEye_img = rightEye_img / 255
        rightEye_img = rightEye_img.transpose(2, 0, 1)

        rects = line[4]
        label = line[5]
        #TODO change this
        exlabel = line[5]

        print(line)

        # what is exlabel
        # how to get rects?

        return {"faceImg": torch.from_numpy(face_img).type(torch.FloatTensor),
                "leftEyeImg": torch.from_numpy(leftEye_img).type(torch.FloatTensor),
                "rightEyeImg": torch.from_numpy(rightEye_img).type(torch.FloatTensor),
                "rects": torch.from_numpy(np.array(rects)).type(torch.FloatTensor),
                "label": torch.from_numpy(label).type(torch.FloatTensor),
                "exlabel": torch.from_numpy(np.array(exlabel)).type(torch.FloatTensor), "frame": line}


def txtload(path, type, batch_size, shuffle=False, num_workers=0):
    dataset = loader(path, type)
    print("[Read Data]: MPIIFaceGaze Dataset")
    print("[Read Data]: Total num: {:d}".format(len(dataset)))
    print("[Read Data]: Dataset type: {:s}".format(type))
    load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return load


if __name__ == "__main__":
    path = "Data/out/Image"
    type = "train"
    loader = txtload(path, type, batch_size=2)
    for i, (data) in enumerate(loader):
        # print(data['frame'][0][0] + ' ' + data['frame'][1][0])
        '''print(data['faceImg'][0].shape)
                                print(torch.mean(data['faceImg'][0]))
                                print(torch.mean(data['leftEyeImg'][0]))
                                print(torch.mean(data['rightEyeImg'][0]))
                                print(data['rects'][0])
                                print(data['exlabel'][0])'''
        break
