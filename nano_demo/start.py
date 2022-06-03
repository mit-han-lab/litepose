from threading import Thread

import cv2

from core import get_model_executor, process, get_cfg

resolution = 448

visualize_queue = []


class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        # rounded up to 16:9
        h = 720  # int(math.ceil(resolution / 9) * 9)
        w = 1280  # int(h / 9 * 16)
        print('using an input resolution of {}x{}'.format(h, w))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.stopped = False
        self.get_once()

    def start(self):
        self.thread = Thread(target=self.get, args=())
        self.thread.start()
        return self

    def get_once(self):
        (self.grabbed, frame) = self.stream.read()
        h, w, _ = frame.shape
        # self.frame = self.frame[:resolution, (w - h) // 2: (w - h) // 2 + resolution, :]
        # 1. to square
        frame = frame[:, (w - h) // 2: (w - h) // 2 + h, :]
        # 2. resize
        self.frame = cv2.resize(frame, (resolution, resolution))

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                self.get_once()

    def stop(self):
        self.stopped = True
        self.thread.join()


class VideoShow:
    def __init__(self, frame=None):
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            if len(visualize_queue) > 0:
                cv2.imshow(WINDOW_NAME, visualize_queue.pop(0))
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        self.stopped = True


if __name__ == '__main__':
    cfg = get_cfg()
    executor, gmod, device = get_model_executor()

    WINDOW_NAME = 'lite_pose'
    print("Open camera...")

    # env variables
    full_screen = False
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, resolution, resolution)  # Please adjust to appropriate size
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)

    i_frame = -1

    video_getter = VideoGet(0).start()
    video_show = VideoShow().start()

    print("Ready!")
    while True:
        i_frame += 1
        frame = video_getter.frame
        if frame is None:
            continue
        output_frame = process(cfg, frame, executor)[:, ::-1]
        # cv2.imshow(WINDOW_NAME, output_frame)
        visualize_queue.append(output_frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:  # exit
            break
        elif key == ord('F') or key == ord('f'):  # full screen
            print('Changing full screen option!')
            full_screen = not full_screen
            if full_screen:
                print('Setting FS!!!')
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL)

    video_getter.stop()
    cv2.destroyAllWindows()
