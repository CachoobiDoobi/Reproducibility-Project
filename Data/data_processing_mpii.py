import numpy as np
import scipy.io as sio
import cv2
import os

# root = "/mnt/d/Master/Q3/Deep\ Learning/AFF-Net/AFF-Net/data/MPIIFaceGaze"
root = "MPIIFaceGaze"
sample_root = "MPIIFaceGaze"
out_root = "out"
scale = False
# loading the haar case algorithm for face detection
alg = "haarcascade_frontalface_default.xml"
# passing the algorithm to OpenCV
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + alg)


def ImageProcessing_MPII():
    persons = os.listdir(sample_root)
    persons.sort()

    for person in persons:

        sample_list = os.path.join(sample_root, person)

        person = person.split(".")[0]
        if person == "readme":
            continue
        im_root = os.path.join(root, person)
        anno_path = os.path.join(root, person, f"{person}.txt")

        im_outpath = os.path.join(out_root, "Image", person)
        label_outpath = os.path.join(out_root, "../label", f"{person}.label")

        if not os.path.exists(im_outpath):
            os.makedirs(im_outpath)
        if not os.path.exists(os.path.join(out_root, "../label")):
            os.makedirs(os.path.join(out_root, "../label"))

        screen = os.path.join(root, person, "Calibration", "screenSize.mat")

        print(f"Start Processing {person}")
        ImageProcessing_Person(im_root, anno_path, screen, sample_list, im_outpath, label_outpath, person)


def ImageProcessing_Person(im_root, anno_path, screen_path, sample_list, im_outpath, label_outpath, person):
    # Read camera matrix
    camera = sio.loadmat(os.path.join(f"{im_root}", "Calibration", "Camera.mat"))
    camera = camera["cameraMatrix"]

    infile = sio.loadmat(screen_path)
    screen_width_p = infile['width_pixel'].flatten()[0]
    screen_height_p = infile['height_pixel'].flatten()[0]
    screen_width_m = infile['width_mm'].flatten()[0]
    screen_height_m = infile['height_mm'].flatten()[0]
    screen_width_r = screen_width_m/screen_width_p
    screen_height_r = screen_height_m/screen_height_p
    # Read gaze annotation
    annotation = os.path.join(anno_path)
    with open(annotation) as infile:
        anno_info = infile.readlines()
    anno_dict = {line.split(" ")[0]: line.strip().split(" ")[1:-1] for line in anno_info}

    # Create the handle of label 
    outfile = open(label_outpath, 'w')
    outfile.write("Face Left Right Rects 2DGazeNorm Label 2DGazeScreen\n")
    if not os.path.exists(os.path.join(im_outpath, "face")):
        os.makedirs(os.path.join(im_outpath, "face"))
    if not os.path.exists(os.path.join(im_outpath, "left")):
        os.makedirs(os.path.join(im_outpath, "left"))
    if not os.path.exists(os.path.join(im_outpath, "right")):
        os.makedirs(os.path.join(im_outpath, "right"))
    if not os.path.exists(os.path.join(im_outpath, "rects")):
        os.makedirs(os.path.join(im_outpath, "rects"))

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

        # Read image
        im_path = os.path.join(im_root, day, im_name)
        im = cv2.imread(im_path)
        # Get image size
        im_height, im_width, _ = im.shape

        annotation = anno_dict[im_info]
        annotation = AnnoDecode(annotation)
        rects = []
        # Extract information
        gaze = annotation["2d_gaze"].copy()
        rlc = annotation["right_left_corner"]
        rrc = annotation["right_right_corner"]
        llc = annotation["left_left_corner"]
        lrc = annotation["left_right_corner"]
        l_mouth = annotation["left_mouth_corner"]
        r_mouth = annotation["right_mouth_corner"]
        face_center = (llc + lrc + rlc + rrc) / 8.0 + (l_mouth + r_mouth) / 4.0

        # Normalise gaze to image coordinates from screen coordinates
        gaze[0] = int((gaze[0]/screen_width_p)*im_width)
        gaze[1] = int((gaze[1]/screen_height_p)*im_height)

        # Crop left eye images
        im_left, (c1_left, c2_left), sizeL = CropEye(im, llc, lrc)
        
        # Crop Right eye images
        im_right, (c1_right, c2_right), sizeR = CropEye(im, rlc, rrc)
       
        # Flip right eye image
        im_right = cv2.flip(im_right, 1)

        # Crop face images
        im_face = im.copy()
        im_face, (c1_face, c2_face), = CropFace(im, face_center, sizeL)

        # Append rects in right order
        rects.append([c1_face[0], c1_face[1], c2_face[0], c2_face[1]])
        rects.append([c1_left[0], c1_left[1], c2_left[0], c2_left[1]])
        rects.append([c1_right[0], c1_right[1], c2_right[0], c2_right[1]])

        # Draw rects for debugging
        cv2.rectangle(im, (rects[0][0], rects[0][1]), (rects[0][2], rects[0][3]), (0, 255, 0), 2)
        cv2.rectangle(im, (rects[1][0], rects[1][1]), (rects[1][2], rects[1][3]), (255, 0, 0), 2)
        cv2.rectangle(im, (rects[2][0], rects[2][1]), (rects[2][2], rects[2][3]), (255, 0, 0), 2)

        # Draw gaze location for debugging
        cv2.circle(im, (im_width - int((gaze[0])),int((gaze[1]))), radius=5, color=(0, 0, 255), thickness=-1)
        cv2.imwrite(os.path.join(im_outpath, "rects", str(count + 1) + ".jpg"), im)

        # Save the acquired info
        cv2.imwrite(os.path.join(im_outpath, "face", str(count + 1) + ".jpg"), im_face)
        cv2.imwrite(os.path.join(im_outpath, "left", str(count + 1) + ".jpg"), im_left)
        cv2.imwrite(os.path.join(im_outpath, "right", str(count + 1) + ".jpg"), im_right)
        # Normalize Rects and Gaze
        for rect in rects:
            rect[0] = rect[0] / im_width
            rect[1] = rect[1] / im_height
            rect[2] = rect[2] / im_width
            rect[3] = rect[3] / im_height
        gaze[0] = gaze[0] / im_width
        gaze[1] = gaze[1] / im_height
        # Save the acquired info
        label = os.path.join(person, str(count + 1))
        save_name_face = os.path.join(person, "face", str(count + 1) + ".jpg")
        save_name_left = os.path.join(person, "left", str(count + 1) + ".jpg")
        save_name_right = os.path.join(person, "right", str(count + 1) + ".jpg")
        rects = ",".join(str(item) for sublist in rects for item in sublist)
        save_gaze2d_norm = ",".join(gaze.astype("str"))
        save_gaze2d_screen = ",".join(annotation["2d_gaze"].astype("int").astype("str"))
        save_str = " ".join(
            [save_name_face, save_name_left, save_name_right, rects, save_gaze2d_norm, label, save_gaze2d_screen])

        outfile.write(save_str + "\n")
    outfile.close()


def AnnoDecode(anno_info):
    annotation = np.array(anno_info).astype("float32")
    out = {}
    out["2d_gaze"] = annotation[0:2]
    out["left_left_corner"] = annotation[2:4]
    out["left_right_corner"] = annotation[4:6]
    out["right_left_corner"] = annotation[6:8]
    out["right_right_corner"] = annotation[8:10]
    out["left_mouth_corner"] = annotation[10:12]
    out["right_mouth_corner"] = annotation[12:14]
    out["headrotvectors"] = annotation[14:17]
    out["headtransvectors"] = annotation[17:20]
    out["facecenter"] = annotation[20:23]
    out["3d_gaze"] = annotation[23:26]
    return out


def CropFace(im, face_center, size):
    imsize = (im.shape[1], im.shape[0])

    size = size * (1.0 / 0.3)
    x1 = [max(face_center[0] - size / 2, 0), max(face_center[1] - size / 2, 0)]
    x2 = [min(x1[0] + size, imsize[0]), min(x1[1] + size, imsize[1])]

    result = im[int(x1[1]):int(x2[1]), int(x1[0]):int(x2[0])]
    result = cv2.resize(result, (224, 224))

    rects = [int(x1[0]), int(x1[1])], [int(x2[0]), int(x2[1])]

    # grayImg = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # face = haar_cascade.detectMultiScale(grayImg, 1.3, 4)
    # for (x, y, w, h) in face:
    #     return [x, y], [x + w, y + h]
    # print("No face found! Bounding box equal with size of the image\n")
    return result, rects


def CropEye(im, lcorner, rcorner):
    imsize = (im.shape[1], im.shape[0])
    x, y = list(zip(lcorner, rcorner))

    center_x = np.mean(x)
    center_y = np.mean(y)

    size = np.abs(x[0] - x[1]) * 1.7
    # times = width/60
    # height = 36 * times

    x1 = [max(center_x - size / 2, 0), max(center_y - size / 2, 0)]
    x2 = [min(x1[0] + size, imsize[0]), min(x1[1] + size, imsize[1])]

    result = im[int(x1[1]):int(x2[1]), int(x1[0]):int(x2[0])]
    result = cv2.resize(result, (112, 112))

    # we also need to return the rects
    rects = [int(x1[0]), int(x1[1])], [int(x2[0]), int(x2[1])]

    return result, rects, size


if __name__ == "__main__":
    ImageProcessing_MPII()
