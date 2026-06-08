# camera_utils.py
import cv2, threading, time

class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.capture.read()
        self.stopped = False
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while not self.stopped:
            if self.capture.isOpened(): self.ret, self.frame = self.capture.read()
            time.sleep(0.01)

    def read(self): return self.ret, self.frame
    def isOpened(self): return self.capture.isOpened()
    def release(self):
        self.stopped = True
        self.thread.join()
        self.capture.release()