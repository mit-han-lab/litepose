import struct
import time

import cv2
import numpy as np

import openncc as ncc
from core import get_model_executor, process, get_cfg

media_head = 'iII13I'


def main():
    res = ncc.load_fw("./moviUsbBoot", "fw/flicRefApp.mvcmd")
    if res < 0:
        raise RuntimeError('load firmware error!')
    print("get usb %d sdk version %s" % (ncc.get_usb_version(), ncc.get_sdk_version()))
    print("get fw version: %s and ncc id %s" % (ncc.camera_get_fw_version(),
                                                ncc.camera_get_ncc_id()))

    sensors = ncc.CameraSensor()
    sensor1 = ncc.SensorModesConfig()
    if sensors.GetFirstSensor(sensor1) == 0:
        print("camera: %s, %dX%d@%dfps, AFmode:%d, maxEXP:%dus,gain[%d, %d]\n" % (
            sensor1.moduleName, sensor1.camWidth, sensor1.camHeight, sensor1.camFps,
            sensor1.AFmode, sensor1.maxEXP, sensor1.minGain, sensor1.maxGain))

    sensor2 = ncc.SensorModesConfig()
    while sensors.GetNextSensor(sensor2) == 0:
        print("camera: %s, %dX%d@%dfps, AFmode:%d, maxEXP:%dus,gain[%d, %d]\n" % (
            sensor2.moduleName, sensor2.camWidth, sensor2.camHeight, sensor2.camFps,
            sensor2.AFmode, sensor2.maxEXP, sensor2.minGain, sensor2.maxGain))

    ncc.camera_select_sensor(0)  # 0 1080p 1 4k
    cameraCfg = sensor1

    cam_info = ncc.CameraInfo()
    cam_info.inputFormat = ncc.IMG_FORMAT_BGR_PLANAR
    # cam_info.meanValue = [float(0.0)]*3
    cam_info.stdValue = 1

    cam_info.isOutputYUV = 1
    cam_info.isOutputH26X = 1
    cam_info.isOutputJPEG = 1

    cam_info.imageWidth = cameraCfg.camWidth
    cam_info.imageHeight = cameraCfg.camHeight
    cam_info.meanValue = [0.0, 0.0, 0.0]
    cam_info.mode = ncc.ENCODE_H264_MODE

    ret = ncc.sdk_init(None, None, None,
                       cam_info, struct.calcsize("13I4f"))  # struct CameraInfo

    metasize = ncc.get_meta_size()
    print("xlink_init ret=%d  %d" % (ret, metasize))
    if (ret < 0):
        return

    offset = struct.calcsize(media_head)
    size = cameraCfg.camWidth * cameraCfg.camHeight * 2
    yuvbuf = bytearray(size + offset)

    ncc.camera_video_out(ncc.YUV420p, ncc.VIDEO_OUT_CONTINUOUS)

    cfg = get_cfg()
    executor, gmod, device = get_model_executor()

    while True:
        size = ncc.GetYuvData(yuvbuf)
        if size <= 0:
            time.sleep(0.1)
            continue
        st = time.time()
        numarray = np.array(yuvbuf[offset:size])
        yuv = numarray.reshape((int(cameraCfg.camHeight * 3 / 2), cameraCfg.camWidth))
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420, 3)  # in bgr format
        output_frame = process(cfg, frame, executor)
        # output_frame = cv2.resize(output_frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        cv2.namedWindow('lite_pose', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('lite_pose', output_frame)
        ed = time.time()
        print('Used %.2f:' % (1000 * (ed - st)))
        if cv2.waitKey(20) == 27:
            break
    ncc.sdk_uninit()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
