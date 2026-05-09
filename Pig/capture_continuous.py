#!/usr/bin/env python3

import os
import cv2
import sys
import time
import random
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


# =========================================================
# Cấu hình ROS2 camera topic
# =========================================================
IMAGE_TOPIC = "/camera_front/image_raw"

# =========================================================
# Cấu hình dataset YOLO26
# =========================================================
DATASET_DIR = "/home/pihuy/tracktor-beam/Pig/yolo26_dataset"

TRAIN_RATIO = 0.8
CLASS_NAMES = ["target"]

IMAGE_TRAIN_DIR = os.path.join(DATASET_DIR, "images", "train")
IMAGE_VAL_DIR = os.path.join(DATASET_DIR, "images", "val")

LABEL_TRAIN_DIR = os.path.join(DATASET_DIR, "labels", "train")
LABEL_VAL_DIR = os.path.join(DATASET_DIR, "labels", "val")

# =========================================================
# Cấu hình chụp ảnh
# =========================================================
TARGET_IMAGE_COUNT = 500     # Muốn chụp bao nhiêu ảnh thì sửa ở đây
CAPTURE_INTERVAL = 0.5       # 0.5s = 2 ảnh/giây
JPEG_QUALITY = 95

# Không dùng GUI
SHOW_PREVIEW = False

# =========================================================
# Cấu hình lọc ảnh
# =========================================================
ENABLE_BLUR_FILTER = True
BLUR_THRESHOLD = 80.0

ENABLE_BRIGHTNESS_FILTER = True
MIN_BRIGHTNESS = 40
MAX_BRIGHTNESS = 230


class Yolo26DatasetCaptureNode(Node):
    def __init__(self):
        super().__init__("yolo26_dataset_capture_node")

        self.bridge = CvBridge()

        self.lastCaptureTime = 0.0
        self.imageCount = 0
        self.skipBlurCount = 0
        self.skipBrightnessCount = 0
        self.failedSaveCount = 0
        self.receivedFrameCount = 0

        self.startTime = time.time()
        self.lastProgressPrintTime = 0.0
        self.isFinished = False

        self.createDatasetFolders()
        self.createDataYaml()

        self.imageSub = self.create_subscription(
            Image,
            IMAGE_TOPIC,
            self.imageCallback,
            qos_profile_sensor_data
        )

        self.waitTimer = self.create_timer(2.0, self.waitingCameraCheck)

        print("==============================================")
        print("YOLO26 Dataset Auto Capture Started")
        print(f"Input topic      : {IMAGE_TOPIC}")
        print(f"Dataset dir      : {DATASET_DIR}")
        print(f"Target images    : {TARGET_IMAGE_COUNT}")
        print(f"Capture interval : {CAPTURE_INTERVAL} s")
        print("Mode             : Auto capture, no GUI")
        print("==============================================")

    def createDatasetFolders(self):
        os.makedirs(IMAGE_TRAIN_DIR, exist_ok=True)
        os.makedirs(IMAGE_VAL_DIR, exist_ok=True)
        os.makedirs(LABEL_TRAIN_DIR, exist_ok=True)
        os.makedirs(LABEL_VAL_DIR, exist_ok=True)

    def createDataYaml(self):
        yamlPath = os.path.join(DATASET_DIR, "data.yaml")

        namesText = ""
        for classId, className in enumerate(CLASS_NAMES):
            namesText += f"  {classId}: {className}\n"

        content = f"""path: {DATASET_DIR}

train: images/train
val: images/val

names:
{namesText}
"""

        with open(yamlPath, "w", encoding="utf-8") as file:
            file.write(content)

        print(f"[INFO] Created data.yaml: {yamlPath}")

    def waitingCameraCheck(self):
        if self.isFinished:
            return

        if self.receivedFrameCount == 0:
            print(f"[WAIT] Chưa nhận được ảnh từ topic: {IMAGE_TOPIC}")

    def imageCallback(self, msg):
        if self.isFinished:
            return

        self.receivedFrameCount += 1

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as error:
            print(f"\n[ERROR] Cannot convert ROS Image to OpenCV: {error}")
            return

        currentTime = time.time()

        if currentTime - self.lastCaptureTime < CAPTURE_INTERVAL:
            return

        self.lastCaptureTime = currentTime

        blurry, blurScore = self.checkBlur(frame)
        brightnessBad, brightness = self.checkBrightness(frame)

        if ENABLE_BLUR_FILTER and blurry:
            self.skipBlurCount += 1
            self.printProgress(blurScore, brightness)
            return

        if ENABLE_BRIGHTNESS_FILTER and brightnessBad:
            self.skipBrightnessCount += 1
            self.printProgress(blurScore, brightness)
            return

        success = self.saveFrame(frame)

        if success:
            self.imageCount += 1
        else:
            self.failedSaveCount += 1

        self.printProgress(blurScore, brightness)

        if self.imageCount >= TARGET_IMAGE_COUNT:
            self.finishCapture()

    def saveFrame(self, frame):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        imageDir, splitName = self.chooseSaveDir()

        filename = f"drone_belly_{timestamp}_{self.imageCount + 1:06d}.jpg"
        imagePath = os.path.join(imageDir, filename)

        encodeParams = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        success = cv2.imwrite(imagePath, frame, encodeParams)

        return success

    def chooseSaveDir(self):
        if random.random() < TRAIN_RATIO:
            return IMAGE_TRAIN_DIR, "train"

        return IMAGE_VAL_DIR, "val"

    def checkBlur(self, frame):
        if not ENABLE_BLUR_FILTER:
            return False, 999.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurScore = cv2.Laplacian(gray, cv2.CV_64F).var()

        blurry = blurScore < BLUR_THRESHOLD

        return blurry, blurScore

    def checkBrightness(self, frame):
        if not ENABLE_BRIGHTNESS_FILTER:
            return False, 128.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()

        brightnessBad = brightness < MIN_BRIGHTNESS or brightness > MAX_BRIGHTNESS

        return brightnessBad, brightness

    def printProgress(self, blurScore, brightness):
        currentTime = time.time()

        # Giới hạn tần suất in để terminal không bị spam quá nhiều
        if currentTime - self.lastProgressPrintTime < 0.2 and self.imageCount < TARGET_IMAGE_COUNT:
            return

        self.lastProgressPrintTime = currentTime

        percent = (self.imageCount / TARGET_IMAGE_COUNT) * 100.0

        elapsed = currentTime - self.startTime

        if self.imageCount > 0:
            speed = self.imageCount / elapsed
            remainingImages = TARGET_IMAGE_COUNT - self.imageCount
            etaSeconds = remainingImages / speed if speed > 0 else 0
        else:
            speed = 0.0
            etaSeconds = 0.0

        barLength = 30
        filledLength = int(barLength * self.imageCount / TARGET_IMAGE_COUNT)
        bar = "#" * filledLength + "-" * (barLength - filledLength)

        etaText = self.formatTime(etaSeconds)

        text = (
            f"\r[{bar}] "
            f"{percent:6.2f}% | "
            f"saved {self.imageCount}/{TARGET_IMAGE_COUNT} | "
            f"speed {speed:.2f} img/s | "
            f"ETA {etaText} | "
            f"skip_blur {self.skipBlurCount} | "
            f"skip_light {self.skipBrightnessCount} | "
            f"fail {self.failedSaveCount} | "
            f"blur {blurScore:.1f} | "
            f"brightness {brightness:.1f}"
        )

        sys.stdout.write(text)
        sys.stdout.flush()

    def formatTime(self, seconds):
        seconds = int(seconds)

        minutes = seconds // 60
        seconds = seconds % 60

        return f"{minutes:02d}:{seconds:02d}"

    def finishCapture(self):
        if self.isFinished:
            return

        self.isFinished = True

        print("\n==============================================")
        print("[DONE] Đã chụp đủ dataset")
        print(f"Saved images      : {self.imageCount}")
        print(f"Skipped blur      : {self.skipBlurCount}")
        print(f"Skipped brightness: {self.skipBrightnessCount}")
        print(f"Failed save       : {self.failedSaveCount}")
        print(f"Dataset dir       : {DATASET_DIR}")
        print("==============================================")

        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    node = Yolo26DatasetCaptureNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[INFO] Người dùng dừng bằng Ctrl+C")

    if rclpy.ok():
        rclpy.shutdown()

    node.destroy_node()


if __name__ == "__main__":
    main()