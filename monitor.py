# -*- coding: utf-8 -*-
"""
YOLOv8 多源视频监控安全检测系统
支持 USB 摄像头、RTSP 监控流、本地视频同时运行。
"""

import os
import sys
import time
import platform
import subprocess
import threading
from datetime import datetime
from pathlib import Path

# 强制控制台输出使用 UTF-8 编码
if platform.system() == "Windows":
    os.system("chcp 65001 >nul 2>&1")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cv2
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# 路径处理
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
RECORDS_DIR = os.path.join(BASE_DIR, "records")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CONFIG_PATH = os.path.join(BASE_DIR, "config.txt")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RECORDS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 读取配置 - 支持多视频源
# ---------------------------------------------------------------------------
def read_config(config_path):
    """
    读取配置文件，支持多视频源。
    """
    sources = ["0"]
    show_window = True
    resolution = None  # 如 "1920x1080"
    detect_interval = 1  # 每 N 帧检测一次，1=每帧都检测
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "=" not in line or line.startswith("#"):
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "source":
                    sources = [s.strip() for s in value.split(",") if s.strip()]
                elif key == "show_window":
                    show_window = value.lower() in ("true", "1", "yes")
                elif key == "resolution":
                    resolution = value
                elif key == "detect_interval":
                    detect_interval = max(1, int(value))
    except Exception as e:
        print(f"[警告] 读取配置失败: {e}，使用默认值")
    return sources, show_window, resolution, detect_interval

# ---------------------------------------------------------------------------
# GPU 检测与模型加载（共享）
# ---------------------------------------------------------------------------
_model = None
_device = None
_model_lock = threading.Lock()

def detect_gpu():
    """检测可用 GPU。"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"[GPU] 检测到显卡: {gpu_name} ({gpu_mem:.1f} GB)")
            return "cuda:0"
    except Exception:
        pass
    print("[GPU] 未检测到可用 GPU，使用 CPU")
    return "cpu"

def load_shared_model():
    """加载共享模型（线程安全）。"""
    global _model, _device
    with _model_lock:
        if _model is not None:
            return _model, _device
        _device = detect_gpu()
        from ultralytics import YOLO
        model_path = os.path.join(MODELS_DIR, "yolov8m.pt")
        if not os.path.exists(model_path):
            print(f"[错误] 模型文件不存在: {model_path}")
            sys.exit(1)
        _model = YOLO(model_path)
        print(f"[模型] 已加载 yolov8m，设备: {_device}")
        return _model, _device

# ---------------------------------------------------------------------------
# 视频源类型判断
# ---------------------------------------------------------------------------
def classify_source(source):
    """判断视频源类型：usb / rtsp / local"""
    s = source.strip()
    if s.isdigit():
        return "usb", int(s)
    if s.lower().startswith("rtsp://"):
        return "rtsp", s
    return "local", s

# ---------------------------------------------------------------------------
# 打开视频源
# ---------------------------------------------------------------------------
def open_video_source(source_type, source_value, resolution=None):
    """
    打开不同类型的视频源。
    resolution: USB 摄像头分辨率，如 "1920x1080"
    """
    if source_type == "usb":
        # USB 摄像头
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(source_value, backend)
            if cap.isOpened():
                # 设置分辨率
                if resolution:
                    try:
                        w, h = resolution.lower().split("x")
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
                        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print(f"[摄像头] 请求分辨率: {resolution}，实际: {actual_w}x{actual_h}")
                    except Exception as e:
                        print(f"[警告] 设置分辨率失败: {e}")
                return cap
        print(f"[错误] 无法打开 USB 摄像头 (设备 {source_value})")
        return None

    elif source_type == "rtsp":
        # RTSP 流
        cap = cv2.VideoCapture(source_value, cv2.CAP_FFMPEG)
        if cap.isOpened():
            return cap
        # 重试
        for i in range(3):
            time.sleep(1)
            cap = cv2.VideoCapture(source_value, cv2.CAP_FFMPEG)
            if cap.isOpened():
                return cap
        print(f"[错误] 无法连接 RTSP: {source_value}")
        return None

    else:
        # 本地视频文件
        if os.path.exists(source_value):
            cap = cv2.VideoCapture(source_value)
            if cap.isOpened():
                return cap
            # 中文路径兼容
            try:
                data = np.fromfile(source_value, dtype=np.uint8)
                cap = cv2.VideoCapture(cv2.imdecode(data, cv2.IMREAD_COLOR))
                if cap.isOpened():
                    return cap
            except Exception:
                pass
        print(f"[错误] 视频源不存在: {source_value}")
        return None

# ---------------------------------------------------------------------------
# RTSP 断线重连
# ---------------------------------------------------------------------------
def reconnect_rtsp(source_value, max_attempts=0):
    """RTSP 断线重连，max_attempts=0 无限重试。"""
    attempt = 0
    while True:
        attempt += 1
        if max_attempts > 0 and attempt > max_attempts:
            return None
        print(f"[重连] 第 {attempt} 次尝试: {source_value}")
        cap = cv2.VideoCapture(source_value, cv2.CAP_FFMPEG)
        if cap.isOpened():
            print("[重连] 连接成功")
            return cap
        time.sleep(5)

# ---------------------------------------------------------------------------
# 检测类别
# ---------------------------------------------------------------------------
DETECT_CLASSES = [0, 2, 3, 5, 7, 15, 16]  # person, car, motorcycle, bus, truck, cat, dog
CLASS_NAMES_CN = {
    0: "人", 2: "汽车", 3: "摩托车",
    5: "公交车", 7: "卡车", 15: "猫", 16: "狗"
}
CLASS_COLORS = {
    0: (0, 255, 0), 2: (255, 128, 0), 3: (255, 0, 255),
    5: (0, 128, 255), 7: (128, 0, 255), 15: (255, 255, 0), 16: (0, 255, 255)
}

# ---------------------------------------------------------------------------
# 日志记录器（线程安全）
# ---------------------------------------------------------------------------
class DetectionLogger:
    def __init__(self):
        self.lock = threading.Lock()
        log_name = datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
        log_path = os.path.join(LOGS_DIR, log_name)
        self.log_file = open(log_path, "a", encoding="utf-8")
        self.log_file.write(f"=== YOLOv8 监控检测日志 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        self.log_file.flush()

    def log_detection(self, source_name, class_name, track_id, confidence):
        with self.lock:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            line = f"[{timestamp}] [{source_name}] 发现目标: {class_name} | 追踪ID: {track_id} | 置信度: {confidence:.2f}\n"
            self.log_file.write(line)
            self.log_file.flush()

    def close(self):
        with self.lock:
            if self.log_file:
                self.log_file.close()

# ---------------------------------------------------------------------------
# 自动录像器
# ---------------------------------------------------------------------------
class AutoRecorder:
    def __init__(self, source_name):
        self.source_name = source_name
        self.writer = None
        self.recording = False
        self.last_target_time = 0
        self.lock = threading.Lock()
        self.codec = cv2.VideoWriter_fourcc(*"mp4v")

    def start(self, fps, frame_size):
        with self.lock:
            if self.recording:
                return
            record_name = f"record_{self.source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            record_path = os.path.abspath(os.path.join(RECORDS_DIR, record_name))
            self.writer = cv2.VideoWriter(record_path, self.codec, fps, frame_size)
            self.recording = True
            self.last_target_time = time.time()
            print(f"[录像][{self.source_name}] 开始: {record_name}")

    def write(self, frame):
        with self.lock:
            if self.writer and self.recording:
                self.writer.write(frame)
                self.last_target_time = time.time()

    def check_timeout(self, timeout=5):
        with self.lock:
            if self.recording and time.time() - self.last_target_time > timeout:
                self.stop()
                return True
            return False

    def stop(self):
        with self.lock:
            if self.writer:
                self.writer.release()
                self.writer = None
                self.recording = False
                print(f"[录像][{self.source_name}] 已保存")

# ---------------------------------------------------------------------------
# 单路视频源处理线程
# ---------------------------------------------------------------------------
class VideoStreamThread(threading.Thread):
    def __init__(self, source_str, show_window, logger, shared_model, resolution=None, detect_interval=1):
        super().__init__(daemon=True)
        self.source_str = source_str
        self.source_type, self.source_value = classify_source(source_str)
        self.source_name = source_str.replace(":", "_").replace("/", "_").replace("\\", "_")[:20]
        self.show_window = show_window
        self.resolution = resolution
        self.detect_interval = detect_interval
        self.logger = logger
        self.model, self.device = shared_model
        self.running = True
        self.recorder = AutoRecorder(self.source_name)

    def run(self):
        tag = f"[{self.source_str}]"
        print(f"{tag} 启动监控线程...")

        try:
            self._run_inner(tag)
        except Exception as e:
            print(f"{tag} 线程异常: {e}")
            import traceback
            traceback.print_exc()

    def _run_inner(self, tag):
        cap = open_video_source(self.source_type, self.source_value, self.resolution)
        if cap is None:
            print(f"{tag} 无法打开视频源，线程退出")
            return

        # 获取视频参数
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps is None:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"{tag} 已连接 {width}x{height} @ {fps:.1f}fps")
        print(f"{tag} 开始检测循环...")

        consecutive_failures = 0
        frame_count = 0
        last_results = None  # 缓存上次检测结果
        window_name = f"YOLOv8 - {self.source_str}"

        while self.running:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if frame_count == 0:
                    print(f"{tag} 首帧读取失败，ret={ret}")
                if self.source_type == "rtsp" and consecutive_failures >= 10:
                    print(f"{tag} 断线，进入重连...")
                    cap = reconnect_rtsp(self.source_value)
                    if cap is None:
                        break
                    consecutive_failures = 0
                elif self.source_type == "local":
                    print(f"{tag} 视频播放完毕")
                    break
                else:
                    time.sleep(0.1)
                continue

            consecutive_failures = 0
            frame_count += 1
            if frame_count == 1:
                print(f"{tag} 首帧读取成功，开始检测")

            # 每 5000 帧重置追踪器，防止积累误差
            if frame_count % 5000 == 0:
                old_stderr, devnull_fd = _suppress_output()
                try:
                    self.model = YOLO(os.path.join(MODELS_DIR, "yolov8m.pt"))
                finally:
                    _restore_output(old_stderr, devnull_fd)
                print(f"{tag} 帧#{frame_count} 追踪器已重置")

            # 按间隔检测，降低 GPU 负载
            need_detect = (frame_count % self.detect_interval == 0)

            if need_detect:
                # 推理（OS 级别屏蔽 C++ 追踪器警告）
                old_stderr, devnull_fd = _suppress_output()
                try:
                    results = self.model.track(
                        frame, persist=True, classes=DETECT_CLASSES,
                        conf=0.4, iou=0.5, device=self.device, verbose=False
                    )
                    last_results = results
                except Exception:
                    last_results = None
                finally:
                    _restore_output(old_stderr, devnull_fd)

                if last_results is None and results is None:
                    # 推理失败，重置模型
                    self.model = YOLO(os.path.join(MODELS_DIR, "yolov8m.pt"))
                    print(f"{tag} 追踪器异常已自动恢复")
                    continue
                results = last_results
            else:
                results = last_results

            has_target = False
            annotated = frame.copy()

            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    track_id = int(box.id[0]) if box.id is not None else -1
                    cn_name = CLASS_NAMES_CN.get(cls_id, "未知")
                    color = CLASS_COLORS.get(cls_id, (255, 255, 255))

                    self.logger.log_detection(self.source_str, cn_name, track_id, conf)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"ID:{track_id} {cn_name} {conf:.2f}"
                    cv2.putText(annotated, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    has_target = True

            # 录像
            if has_target:
                if not self.recorder.recording:
                    self.recorder.start(fps, (width, height))
                self.recorder.write(annotated)
            else:
                self.recorder.check_timeout(timeout=5)

            # 显示
            if self.show_window:
                try:
                    cv2.imshow(window_name, annotated)
                except cv2.error:
                    self.show_window = False

            # 退出键
            if self.show_window:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    self.running = False

        # 清理
        cap.release()
        self.recorder.stop()
        print(f"{tag} 线程退出")

    def stop(self):
        self.running = False

# ---------------------------------------------------------------------------
# 扫描可用摄像头
# ---------------------------------------------------------------------------
def _suppress_output():
    """OS 级别屏蔽 stderr（C++ 警告）。返回 (old_stderr, devnull_fd)。"""
    old_stderr = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    return old_stderr, devnull_fd

def _restore_output(old_stderr, devnull_fd):
    """恢复 stderr。"""
    try:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
        os.close(devnull_fd)
    except Exception:
        pass

def scan_cameras(max_scan=10):
    """扫描系统可用的 USB 摄像头（静默扫描）。"""
    available = []
    old_stderr, devnull_fd = _suppress_output()
    try:
        for i in range(max_scan):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available.append((i, w, h))
                cap.release()
    finally:
        _restore_output(old_stderr, devnull_fd)
    return available

# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------
def main():
    print("=" * 50)
    print("  YOLOv8 多源视频监控安全检测系统")
    print("  支持 USB / RTSP / 本地视频 同时运行")
    print("=" * 50)
    print()

    # 扫描摄像头
    cameras = scan_cameras()
    if cameras:
        print("[扫描] 发现 USB 摄像头:")
        for idx, w, h in cameras:
            print(f"         设备 {idx}: {w}x{h}")
    else:
        print("[扫描] 未发现 USB 摄像头")
    print()

    # 读取配置
    sources, show_window, resolution, detect_interval = read_config(CONFIG_PATH)
    print(f"[配置] 视频源: {sources}")
    print(f"[配置] 显示窗口: {show_window}")
    if resolution:
        print(f"[配置] USB 摄像头分辨率: {resolution}")
    if detect_interval > 1:
        print(f"[配置] 检测间隔: 每 {detect_interval} 帧检测一次")
    print()

    # 加载模型
    shared_model = load_shared_model()

    # 创建日志
    logger = DetectionLogger()

    # 启动各路视频线程
    threads = []
    for src in sources:
        t = VideoStreamThread(src, show_window, logger, shared_model, resolution, detect_interval)
        threads.append(t)
        t.start()
        time.sleep(0.5)  # 错开启动

    print(f"[系统] 已启动 {len(threads)} 路监控，按 Ctrl+C 停止")
    print()

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[系统] 用户中断，正在停止...")
        for t in threads:
            t.stop()

    # 等待线程结束
    for t in threads:
        t.join(timeout=5)

    logger.close()
    cv2.destroyAllWindows()
    print("[系统] 已安全退出")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[致命错误] {e}")
        import traceback
        traceback.print_exc()
        input("按回车键退出...")
