import sys
import cv2
import torch
import numpy as np
import requests
sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device
from yolov5.utils.centroidtracker import CentroidTracker

useIPCam = True

CAMERA_ID = 'name'
#CAMERA_SRC = ''
CAMERA_SRC = 'cam_source'

BACKEND_ADDR = "http://localhost:8001/save_vehicles"


def generateCentroid(rects):
    inputCentroids = np.zeros((len(rects), 2), dtype="int")
    for (i, (startX, startY, endX, endY)) in enumerate(rects):
        cX = int((startX + endX) / 2.0)
        cY = int((startY + endY) / 2.0)
        inputCentroids[i] = (cX, cY)
    return inputCentroids


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def frames():
    global useIPCam
    total_car_count = 0
    total_accident_count = 0
    weights, imgsz = 'yolov5/weights/best.pt', 640
    
    source = CAMERA_SRC
    
    device = select_device()
    model = attempt_load(weights, map_location=device)  # load FP32 model
    model.to(device).eval()

    # Half precision
    half = False and device.type != 'cpu'
    print('half = ' + str(half))

    if half:
        model.half()

    # Set Dataloader
    if (useIPCam):
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    names = model.names if hasattr(model, 'names') else model.modules.names
    # Run inference
    ct = CentroidTracker(maxDisappeared=1)
    crash_ct = CentroidTracker(maxDisappeared=30)
    listDet = ['car', 'crash']

    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)
        rects = []
        crash_rects = []
        labelObj = []
        for i, det in enumerate(pred):  # detections per image

            if (useIPCam):
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()  # if rtsp/camera
            else:
                p, s, im0 = path, '', im0s
            height, width, channels = im0.shape
            print(height, width)

            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %s, ' % (n, names[int(c)])  # add to string
                for *xyxy, conf, cls in det:
                    label = '%s %.2f' % (names[int(cls)], conf)

                    x = xyxy
                    tl = None or round(0.002 * (im0.shape[0] + im0.shape[1]) / 2) + 1  # line/font thickness
                    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

                    label1 = label.split(' ')
                    if label1[0] in listDet:
                        box = (int(x[0]), int(x[1]), int(x[2]), int(x[3]))
                        rects.append(box)
                        if label1[0] == 'crash':
                            crash_rects.append(box)
                        labelObj.append(label1[0])
                        cv2.rectangle(im0, c1, c2, (0, 0, 0), thickness=tl, lineType=cv2.LINE_AA)
                        tf = max(tl - 1, 1)
                        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                        cv2.rectangle(im0, c1, c2, (0, 100, 0), -1, cv2.LINE_AA)

                        cv2.putText(im0, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                                    lineType=cv2.LINE_AA)

            ct.update(rects)
            crash_ct.update(crash_rects)

            total_car_count += ct.total_count
            total_accident_count += crash_ct.total_count


            cv2.putText(im0, 'Total car count : ' + str(total_car_count), (int(width * 0.02), int(height * 0.15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(im0, 'Total accident : ' + str(total_accident_count), (int(width * 0.02), int(height * 0.25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.imshow("name", im0)
            try:
                request_param = {'camera_id': CAMERA_ID, 'accident_count': crash_ct.total_count,
                                 'pass_count': ct.total_count}
                response = requests.get(BACKEND_ADDR, params=request_param, timeout=5)
                response.raise_for_status()
                print(f'uploaded :{request_param}')
                print(f'uploaded :{response}')
            except requests.exceptions.HTTPError as errh:
                print(errh)
            except requests.exceptions.ConnectionError as errc:
                print(errc)
            except requests.exceptions.Timeout as errt:
                print(errt)
            except requests.exceptions.RequestException as err:
                print(err)

            ct.total_count = 0
            crash_ct.total_count = 0


if __name__ == '__main__':
    with torch.no_grad():
        frames()
