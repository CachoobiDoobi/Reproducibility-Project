import numpy as np
import scipy.io as sio
import cv2
import os
import sys
# sys.path.append("../core/")
import data_processing_core as dpc

# root = "/mnt/d/Master/Q3/Deep\ Learning/AFF-Net/AFF-Net/data/MPIIFaceGaze"
root = "MPIIFaceGaze"
sample_root = "MPIIFaceGaze"
out_root = "out"
scale = False
# loading the haar case algorithm for face detection
alg = "haarcascade_frontalface_default.xml"
# passing the algorithm to OpenCV
haar_cascade = cv2.CascadeClassifier(alg)

def ImageProcessing_MPII():
    persons = os.listdir(sample_root)[:-1]
    persons.sort()
  
    for person in persons:
        sample_list = os.path.join(sample_root, person)

        person = person.split(".")[0]
        im_root = os.path.join(root, person)
        anno_path = os.path.join(root, person, f"{person}.txt")

        im_outpath = os.path.join(out_root, "Image", person)
        label_outpath = os.path.join(out_root, "../label", f"{person}.label")

        if not os.path.exists(im_outpath):
            os.makedirs(im_outpath)
        if not os.path.exists(os.path.join(out_root, "../label")):
            os.makedirs(os.path.join(out_root, "../label"))

        print(f"Start Processing {person}")
        # print(str(im_root) +" " +  str(anno_path) + " " + str(sample_list) + " " + str(im_outpath) + " " + str(label_outpath) + " " + str(person))
        ImageProcessing_Person(im_root, anno_path, sample_list, im_outpath, label_outpath, person)


def ImageProcessing_Person(im_root, anno_path, sample_list, im_outpath, label_outpath, person):
    # Read camera matrix
    camera = sio.loadmat(os.path.join(f"{im_root}", "Calibration", "Camera.mat"))
    camera = camera["cameraMatrix"]

    # Read gaze annotation
    annotation = os.path.join(anno_path)
    with open(annotation) as infile:
        anno_info = infile.readlines()
    anno_dict = {line.split(" ")[0]: line.strip().split(" ")[1:-1] for line in anno_info}

    # Create the handle of label 
    outfile = open(label_outpath, 'w')
    outfile.write("Face Left Right Rects 2DGaze Label\n")
    if not os.path.exists(os.path.join(im_outpath, "face")):
        os.makedirs(os.path.join(im_outpath, "face"))
    if not os.path.exists(os.path.join(im_outpath, "left")):
        os.makedirs(os.path.join(im_outpath, "left"))
    if not os.path.exists(os.path.join(im_outpath, "right")):
        os.makedirs(os.path.join(im_outpath, "right"))
    if not os.path.exists(os.path.join(im_outpath, "rects")):
        os.makedirs(os.path.join(im_outpath, "rects"))
    # caceu code
    # TODO delete this line
    days = os.listdir(sample_list)[1:-1]


    # Image Processing 
    with open(sample_list + "/" + person + ".txt") as infile:
        im_list = infile.readlines()
        total = len(im_list)

    for count, info in enumerate(im_list):

        progressbar = "".join(["\033[41m%s\033[0m" % '   '] * int(count / total * 20))
        progressbar = "\r" + progressbar + f" {count}|{total}"
        print(progressbar, end="", flush=True)

        # Read image info
        im_info = info.strip().split(" ")[0]
        which_eye = info.strip().split(" ")[-1]
        day, im_name = im_info.split("/")
        im_number = int(im_name.split(".")[0])

        # Read image annotation and image
        im_path = os.path.join(im_root, day, im_name)
        im = cv2.imread(im_path)
        # cv2.imshow('image',im)
        # cv2.waitKey(0)
        annotation = anno_dict[im_info]
        annotation = AnnoDecode(annotation)
        rects = []
        
        # Crop face images
        im_face = im.copy()
        c1_face, c2_face = CropFace(im)
        rects.append([c1_face[0], c1_face[1], c2_face[0], c2_face[1]])

        # Crop left eye images
        llc = annotation["left_left_corner"]
        lrc = annotation["left_right_corner"]
        im_left,(c1_left,c2_left) = CropEye(im,llc, lrc)
        rects.append([c1_left[0], c1_left[1], c2_left[0], c2_left[1]])

        # Crop Right eye images
        rlc = annotation["right_left_corner"]
        rrc = annotation["right_right_corner"]
        im_right,(c1_right,c2_right) = CropEye(im,rlc, rrc)
        rects.append([c1_right[0], c1_right[1], c2_right[0], c2_right[1]])
        #Flip right eye image
        im_right = cv2.flip(im_right, 1)

        # Save rects for debugging
        cv2.rectangle(im, (rects[0][0],rects[0][1]), (rects[0][2],rects[0][3]), (0, 255, 0), 2)
        cv2.rectangle(im, (rects[1][0],rects[1][1]), (rects[1][2],rects[1][3]), (0, 0, 255), 2)
        cv2.rectangle(im, (rects[2][0],rects[2][1]), (rects[2][2],rects[2][3]), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(im_outpath, "rects", str(count + 1) + ".jpg"), im)
        # Save the acquired info

        cv2.imwrite(os.path.join(im_outpath, "face", str(count + 1) + ".jpg"), im_face)
        cv2.imwrite(os.path.join(im_outpath, "left", str(count + 1) + ".jpg"), im_left)
        cv2.imwrite(os.path.join(im_outpath, "right", str(count + 1) + ".jpg"), im_right)

        label = os.path.join(person, str(count + 1) )
        save_name_face = os.path.join(person, "face", str(count + 1) + ".jpg")
        save_name_left = os.path.join(person, "left", str(count + 1) + ".jpg")
        save_name_right = os.path.join(person, "right", str(count + 1) + ".jpg")
        rects = ",".join(str(item) for sublist in rects for item in sublist)
        save_gaze2d = ",".join(annotation["2d_gaze"].astype("str"))
        save_str = " ".join(
            [save_name_face, save_name_left, save_name_right, rects, save_gaze2d, label])

        outfile.write(save_str + "\n")
    # print("")
    outfile.close()


def AnnoDecode(anno_info):
    annotation = np.array(anno_info).astype("float32")
    out = {}
    out["2d_gaze"] = annotation[0:2]
    out["left_left_corner"] = annotation[2:4]
    out["left_right_corner"] = annotation[4:6]
    out["right_left_corner"] = annotation[6:8]
    out["right_right_corner"] = annotation[8:10]
    out["headrotvectors"] = annotation[14:17]
    out["headtransvectors"] = annotation[17:20]
    out["facecenter"] = annotation[20:23]
    out["3d_gaze"] = annotation[23:26]
    return out

def CropFace(im):
    grayImg = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    face = haar_cascade.detectMultiScale(grayImg, 1.3, 4)
    for (x, y, w, h) in face:
        return [x, y], [x + w, y + h]
    print("No face found! Bounding box equal with size of the image\n")
    return [0, 0], [im.shape[0], im.shape[1]]

def CropEye(im, lcorner, rcorner):
        imsize=(im.shape[1], im.shape[0])
        x, y = list(zip(lcorner, rcorner))
        
        center_x = np.mean(x)
        center_y = np.mean(y)

        width = np.abs(x[0] - x[1])*1.5
        times = width/60
        height = 36 * times

        x1 = [max(center_x - width/2, 0), max(center_y - height/2, 0)]
        x2 = [min(x1[0] + width, imsize[0]), min(x1[1] + height, imsize[1])]
        result = im[int(x1[1]):int(x2[1]), int(x1[0]):int(x2[0])]
        result = cv2.resize(result, (60, 36))
        #we also need to return the rects
        rects = [int(x1[0]),int(x1[1])] , [int(x2[0]),int(x2[1])]
        return result, rects

if __name__ == "__main__":
    ImageProcessing_MPII()
