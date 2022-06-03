import cv2

from core import get_model_executor, process, get_cfg

if __name__ == '__main__':
    cfg = get_cfg()
    executor, gmod, device = get_model_executor()

    WINDOW_NAME = 'lite_pose'
    print("Open camera...")
    cap = cv2.VideoCapture(0)

    print(cap)

    # set a lower resolution for speed up
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 448)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 448)

    # env variables
    full_screen = False
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 448, 448)  # Please adjust to appropriate size
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)

    i_frame = -1

    print("Ready!")
    while True:
        i_frame += 1
        _, frame = cap.read()
        if frame is None:
            continue
        output_frame = process(cfg, frame, executor)
        cv2.imshow(WINDOW_NAME, output_frame)

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

    cap.release()
    cv2.destroyAllWindows()
