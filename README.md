# 视频转文档工具 (Video to Document)

**Language:** [English](README_EN.md) | **中文**

这是一个将教学视频自动转换为图文笔记的Python工具。

## 功能特性

- 🎥 从视频中提取音频
- 🗣️ 语音转文字（支持中文）
- 📸 智能提取关键帧（去重）
- 📝 自动生成Word文档
- ⏱️ 时间戳同步

## 快速开始

### 1. 激活虚拟环境
```bash
source bin/activate
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 准备视频文件
将你的教学视频文件重命名为 `input_video.mp4` 并放在项目根目录。

### 4. 运行程序

#### 使用本地Whisper模型（推荐）
```bash
python v2d.py
```

#### 使用阿里云语音识别API
```bash
# 首先设置环境变量
source .env.local

# 运行阿里云版本
python v2d_ali.py
```

## 程序流程

1. **提取音频** - 从视频中提取音频文件
2. **语音转文字** - 使用Whisper模型识别语音并生成带时间戳的文字
3. **提取关键帧** - 智能提取视频关键帧，自动去重
4. **图文匹配** - 将文字片段与对应时间的关键帧匹配
5. **生成文档** - 创建包含时间戳、图片和文字的Word文档

## 输出结果

程序会生成一个Word文档，包含：
- 每个时间段的时间戳标题
- 对应的关键帧图片
- 语音转文字的内容

## 项目结构

```
v2d/
├── v2d.py              # 主程序：视频转Word文档（本地Whisper模型）
├── v2d_ali.py          # 阿里云语音识别版本（自动OSS上传）
├── requirements.txt    # Python依赖包
├── .env.local         # 阿里云环境变量配置
├── README.md         # 项目使用说明
├── input_video.mp4   # 示例输入视频文件
├── output.docx       # 示例输出文档
└── temp_files/       # 临时处理文件目录
    ├── temp_audio.wav     # 提取的音频文件
    ├── key_frame_*.png    # 提取的关键帧图片
    └── ali_api_result.json # 阿里云API缓存结果
```

## 自定义配置

在 `v2d.py` 中修改以下参数：

```python
# 输入视频路径
VIDEO_PATH = "input_video.mp4"

# 输出Word文档名称
OUTPUT_WORD = "教学视频图文笔记.docx"

# 语音识别模型（模型越大精度越高，但速度越慢）
WHISPER_MODEL = "small"  # 可选：tiny/base/small/medium/large

# 抽帧间隔（秒）
FRAME_INTERVAL = 1

# 关键帧相似度阈值（0-1之间，值越大去重越严格）
SSIM_THRESHOLD = 0.93

# 临时文件目录
TEMP_DIR = "temp_files"

# 相邻文字片段合并阈值（秒）
MERGE_SEC = 2
```

## 依赖包

- `moviepy`: 视频处理
- `whisper`: 语音识别
- `opencv-python`: 图像处理
- `scikit-image`: 图像相似度计算
- `python-docx`: Word文档生成
- `numpy`: 数值计算

## 阿里云语音识别版本 (v2d_ali.py)

### 功能特点
- 使用阿里云在线语音识别API，无需本地模型
- 支持中文语音识别，准确率高
- 自动处理音频文件上传和识别结果获取
- 与v2d.py相同的输出格式和功能

### 环境配置
在 `.env.local` 文件中设置阿里云凭证：
```bash
export ALIYUN_AK_ID=你的AccessKey ID
export ALIYUN_AK_SECRET=你的AccessKey Secret
export NLS_APP_KEY=你的AppKey
export OSS_ACCESS_KEY_ID=你的AccessKey ID
export OSS_ACCESS_KEY_SECRET=你的AccessKey Secret
export OSS_REGION=你的 OSS REGION，例如 cn-beijing
export OSS_BUCKET=你的 OSS BUCKET，例如 v2d
export OSS_ENDPOINT=你的 OSS ENDPOINT，例如 oss-cn-beijing.aliyuncs.com
export OSS_ENDPOINT_INTERNAL=你的 OSS 内网 ENDPOINT，例如 oss-cn-beijing-internal.aliyuncs.com
```

**OSS_ENDPOINT 与 OSS_ENDPOINT_INTERNAL 的区别：**
- `OSS_ENDPOINT`: 用于外部网络访问，适用于文件上传和下载
- `OSS_ENDPOINT_INTERNAL`: 用于阿里云内部网络访问，语音识别时使用内网可节省带宽和降低延迟

### 使用说明
1. 设置环境变量：`source .env.local`
2. 运行程序：`python v2d_ali.py`
3. 程序会自动：
   - 从视频提取音频
   - 上传音频到阿里云OSS
   - 调用语音识别API
   - 生成图文笔记文档

### 命令行选项
```bash
# 使用缓存数据，跳过API调用
python v2d_ali.py --use_cache

# 指定输入视频和输出文档路径
python v2d_ali.py --video my_video.mp4 --output my_notes.docx

# 显示帮助信息
python v2d_ali.py --help
```

### 缓存功能
- 阿里云API结果会自动保存到 `temp_files/ali_api_result.json`
- 使用 `--use_cache` 开关可以直接使用缓存数据，跳过API调用
- 便于测试和开发，避免重复调用API产生费用

### 注意事项
- 需要有效的阿里云账户和语音识别服务权限
- 使用阿里云服务会产生费用
- 需要网络连接访问阿里云API
- 需要配置OSS存储桶用于音频文件上传
- 程序会自动清理OSS上的临时音频文件

## 注意事项

- 需要Python 3.7+
- 首次运行会自动下载Whisper模型，请确保网络连接
- 处理长视频需要较长时间，请耐心等待
- 确保有足够的磁盘空间存储临时文件
- 如果遇到内存不足，可以尝试使用更小的Whisper模型
- 程序会在项目根目录生成临时文件夹 `temp_files`，处理完成后可选择删除