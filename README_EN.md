# Video to Document Tool

**Language:** **English** | [中文](README.md)

A Python tool that automatically converts educational videos into illustrated text notes by extracting audio, transcribing speech, identifying keyframes, and generating Word documents with synchronized timestamps.

## Features

- 🎥 Extract audio from video
- 🗣️ Speech to text (supports Chinese)
- 📸 Intelligent keyframe extraction (with deduplication)
- 📝 Automatic Word document generation
- ⏱️ Timestamp synchronization

## Quick Start

### 1. Activate Virtual Environment
```bash
source bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Video File
Rename your educational video file to `input_video.mp4` and place it in the project root directory.

### 4. Run the Program

#### Using Local Whisper Model (Recommended)
```bash
python v2d.py
```

#### Using Alibaba Cloud Speech Recognition API
```bash
# First set environment variables
source .env.local

# Run Alibaba Cloud version
python v2d_ali.py
```

## Program Flow

1. **Extract Audio** - Extract audio file from video
2. **Speech to Text** - Use Whisper model to recognize speech and generate timestamped text
3. **Extract Keyframes** - Intelligently extract video keyframes with automatic deduplication
4. **Content Matching** - Match text segments with corresponding keyframes
5. **Generate Document** - Create Word document with timestamps, images, and text

## Output Results

The program generates a Word document containing:
- Timestamp headings for each time period
- Corresponding keyframe images
- Speech-to-text content

## Project Structure

```
v2d/
├── v2d.py              # Main program: video to Word document (local Whisper model)
├── v2d_ali.py          # Alibaba Cloud speech recognition version (automatic OSS upload)
├── requirements.txt    # Python dependencies
├── .env.local         # Alibaba Cloud environment variable configuration
├── README.md         # Project usage instructions (Chinese)
├── README_EN.md      # Project usage instructions (English)
├── input_video.mp4   # Example input video file
├── output.docx       # Example output document
└── temp_files/       # Temporary processing files directory
    ├── temp_audio.wav     # Extracted audio file
    ├── key_frame_*.png    # Extracted keyframe images
    └── ali_api_result.json # Alibaba Cloud API cache results
```

## Custom Configuration

Modify the following parameters in `v2d.py`:

```python
# Input video path
VIDEO_PATH = "input_video.mp4"

# Output Word document name
OUTPUT_WORD = "Educational Video Notes.docx"

# Speech recognition model (larger models have higher accuracy but slower speed)
WHISPER_MODEL = "small"  # Options: tiny/base/small/medium/large

# Frame extraction interval (seconds)
FRAME_INTERVAL = 1

# Keyframe similarity threshold (0-1, higher values mean stricter deduplication)
SSIM_THRESHOLD = 0.93

# Temporary files directory
TEMP_DIR = "temp_files"

# Adjacent text segment merge threshold (seconds)
MERGE_SEC = 2
```

## Dependencies

- `moviepy`: Video processing
- `whisper`: Speech recognition
- `opencv-python`: Image processing
- `scikit-image`: Image similarity calculation
- `python-docx`: Word document generation
- `numpy`: Numerical computation

## Alibaba Cloud Speech Recognition Version (v2d_ali.py)

### Features
- Uses Alibaba Cloud online speech recognition API, no local model required
- Supports Chinese speech recognition with high accuracy
- Automatically handles audio file upload and recognition result retrieval
- Same output format and functionality as v2d.py

### Environment Configuration
Set Alibaba Cloud credentials in `.env.local` file:
```bash
export ALIYUN_AK_ID=Your_AccessKey_ID
export ALIYUN_AK_SECRET=Your_AccessKey_Secret
export NLS_APP_KEY=Your_AppKey
export OSS_ACCESS_KEY_ID=Your_AccessKey_ID
export OSS_ACCESS_KEY_SECRET=Your_AccessKey_Secret
export OSS_REGION=Your_OSS_REGION, e.g., cn-beijing
export OSS_BUCKET=Your_OSS_BUCKET, e.g., v2d
export OSS_ENDPOINT=Your_OSS_ENDPOINT, e.g., oss-cn-beijing.aliyuncs.com
export OSS_ENDPOINT_INTERNAL=Your_OSS_INTERNAL_ENDPOINT, e.g., oss-cn-beijing-internal.aliyuncs.com
```

**Difference between OSS_ENDPOINT and OSS_ENDPOINT_INTERNAL:**
- `OSS_ENDPOINT`: Used for external network access, suitable for file upload and download
- `OSS_ENDPOINT_INTERNAL`: Used for Alibaba Cloud internal network access, using internal network during speech recognition saves bandwidth and reduces latency

### Usage Instructions
1. Set environment variables: `source .env.local`
2. Run program: `python v2d_ali.py`
3. The program automatically:
   - Extracts audio from video
   - Uploads audio to Alibaba Cloud OSS
   - Calls speech recognition API
   - Generates illustrated notes document

### Command Line Options
```bash
# Use cached data, skip API calls
python v2d_ali.py --use_cache

# Specify input video and output document paths
python v2d_ali.py --video my_video.mp4 --output my_notes.docx

# Show help information
python v2d_ali.py --help
```

### Cache Functionality
- Alibaba Cloud API results are automatically saved to `temp_files/ali_api_result.json`
- Use `--use_cache` switch to directly use cached data, skipping API calls
- Convenient for testing and development, avoiding repeated API call costs

### Important Notes
- Requires valid Alibaba Cloud account and speech recognition service permissions
- Using Alibaba Cloud services incurs costs
- Requires network connection to access Alibaba Cloud APIs
- Requires configured OSS bucket for audio file upload
- Program automatically cleans up temporary audio files on OSS

## Important Notes

- Requires Python 3.7+
- First run automatically downloads Whisper model, ensure network connection
- Processing long videos takes significant time, please be patient
- Ensure sufficient disk space for temporary files
- If encountering memory issues, try using smaller Whisper models
- Program generates temporary folder `temp_files` in project root, can be deleted after processing