# YOLOv8 多源视频监控系统

基于 YOLOv8 的智能视频监控系统，支持 USB 摄像头、RTSP 监控流、本地视频多源同时运行，支持 Nvidia GPU 加速。

## 功能特性

- **多源同时监控** - USB 摄像头 + RTSP + 本地视频可同时运行
- **GPU 加速推理** - 自动检测 Nvidia GPU（P104/V100），FP16 半精度加速
- **目标检测与追踪** - 支持人、车、宠物检测，ByteTrack 追踪分配唯一 ID
- **自动录像** - 检测到目标自动录制，目标消失 5 秒后停止
- **检测日志** - 实时记录检测数据，包含时间戳、类别、追踪 ID、置信度
- **断线重连** - RTSP 流自动重连，防止程序卡死
- **中文路径兼容** - 完美支持中文目录和文件名
- **无显示器兼容** - 无 GUI 环境下自动后台运行

## 系统要求

- Windows 10/11
- Python 3.10
- Nvidia GPU（可选，支持 CUDA 11.8）

## 安装

### 方式一：使用绿色版 Python（推荐）

1. 下载 [Python 3.10 Embeddable](https://www.python.org/downloads/release/python-31011/) 解压到 `python_env/`

2. 安装 pip：
```bash
cd python_env
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```

3. 编辑 `python_env/python310._pth`，取消 `#import site` 的注释

4. 安装依赖：
```bash
python -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --extra-index-url https://download.pytorch.org/whl/cu118
```

### 方式二：使用系统 Python

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --extra-index-url https://download.pytorch.org/whl/cu118
```

## 模型下载

下载 [yolov8m.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) 放入 `models/` 目录

## 使用方法

### 1. 配置视频源

编辑 `config.txt`：

```ini
# 视频源（支持多源，逗号分隔）
# USB 摄像头: 0, 1, 2...
# RTSP 摄像头: rtsp://用户名:密码@IP:端口/路径
# 本地视频: D:\video.mp4
source=0,rtsp://admin:123456@192.168.1.100:554/stream1

# USB 摄像头分辨率（可选）
resolution=1920x1080

# 是否显示检测窗口
show_window=True
```

### 2. 启动

双击 `双击运行.bat` 或运行：

```bash
python monitor.py
```

### 3. 配置示例

| 场景 | source 设置 |
|------|------------|
| 单路 USB | `source=0` |
| 单路 RTSP | `source=rtsp://admin:123456@192.168.1.100:554/stream1` |
| USB + RTSP | `source=0,rtsp://admin:123456@192.168.1.100:554/stream1` |
| 多路监控 | `source=0,1,rtsp://cam1:554/stream1` |

## 目录结构

```
yolov8/
├── python_env/        # Python 绿色环境
├── models/
│   └── yolov8m.pt    # 模型权重
├── logs/              # 检测日志
├── records/           # 自动录像
├── config.txt         # 配置文件
├── monitor.py         # 核心脚本
├── 双击运行.bat       # 启动脚本
├── requirements.txt   # 依赖列表
└── README.md
```

## 检测类别

| 类别 | COCO ID |
|------|---------|
| 人 (person) | 0 |
| 汽车 (car) | 2 |
| 摩托车 (motorcycle) | 3 |
| 公交车 (bus) | 5 |
| 卡车 (truck) | 7 |
| 猫 (cat) | 15 |
| 狗 (dog) | 16 |

## 输出说明

### 日志格式

```
[2026-07-15 16:30:45.123] [0] 发现目标: 人 | 追踪ID: 1 | 置信度: 0.87
[2026-07-15 16:30:45.456] [rtsp://...] 发现目标: 汽车 | 追踪ID: 2 | 置信度: 0.92
```

### 录像文件

- 文件名格式：`record_{源标识}_{时间戳}.mp4`
- 例：`record_0_20260715_163045.mp4`

## GPU 加速

系统自动检测 Nvidia GPU：

- **显卡** - 自动启用 FP16 半精度推理
- **无 GPU** - 自动使用 CPU（速度较慢）

## 常见问题

### 摄像头打不开

1. 确认摄像头未被其他软件占用
2. 尝试更换 `source` 值（0, 1, 2...）
3. 检查设备管理器中摄像头状态

### RTSP 连接失败

1. 确认摄像头支持 RTSP 协议
2. 检查网络连接和防火墙设置
3. 确认用户名密码正确

### 中文乱码

确保使用 `双击运行.bat` 启动，已内置 UTF-8 编码设置。

## License

GNU Affero General Public License v3.0


🛠️ 获取免配置一键整合包
本项目代码开源，欢迎开发者自行配置环境运行。
如果您是零基础用户、企业客户，或者不想折腾复杂的 PyTorch、CUDA 显卡驱动配置，我们提供 开箱即用、双击即运行的 GPU 加速绿色整合包（约 5GB），并提供以下支持：

⚙️ 完整的 Windows 10 一键免安装绿色运行环境

⚡ 针对 Nvidia 10系老旧显卡的 CUDA/cuDNN 极速推理优化

💬 专属的技术支持与定制功能开发（如特定报警对接、特定物体训练等）

欢迎联系咨询： [andy615.white@gmail.com]
