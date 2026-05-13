"""
glaskuser_init — one-command setup: install deps + download models.
Called by the /glaskuser_init slash command.

Steps:
  1. Check Python version
  2. Install pip dependencies from requirements.txt
  3. Check ffmpeg
  4. Download Whisper + embedding models

Set HF_ENDPOINT before running to use a Chinese mirror, e.g.:
    HF_ENDPOINT=https://hf-mirror.com python src/init.py
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Apply mirror before any HuggingFace import; user can override via env
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

_PROJECT_ROOT = Path(__file__).parent.parent


def check_python() -> bool:
    if sys.version_info >= (3, 10):
        print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")
        return True
    print(f"✗ Python {sys.version_info.major}.{sys.version_info.minor}，需要 3.10+")
    return False


def install_deps() -> bool:
    req_file = _PROJECT_ROOT / "requirements.txt"
    if not req_file.exists():
        print("✗ 未找到 requirements.txt")
        return False

    print("\n正在安装 pip 依赖（首次约 2–3 分钟）...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q"],
        )
        print("✓ 依赖安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 依赖安装失败：{e}")
        print("  可手动运行：pip install -r requirements.txt")
        return False


def create_data_dir() -> bool:
    data_dir = _PROJECT_ROOT / "data"
    if data_dir.exists():
        print("✓ data/ 目录已存在")
        return True
    data_dir.mkdir()
    (data_dir / ".gitkeep").touch()
    print("✓ 已创建 data/ 目录（请将用户研究文件放入此目录）")
    return True


def check_ffmpeg() -> bool:
    if shutil.which("ffmpeg"):
        print("✓ ffmpeg 已安装")
        return True

    print("\nffmpeg 未找到，尝试自动安装...")
    if sys.platform == "darwin":
        if shutil.which("brew"):
            try:
                subprocess.check_call(["brew", "install", "ffmpeg"])
                print("✓ ffmpeg 安装完成")
                return True
            except subprocess.CalledProcessError:
                print("✗ ffmpeg 安装失败，请手动运行：brew install ffmpeg")
                return False
        else:
            print("✗ 未找到 Homebrew，请先安装 Homebrew：")
            print('  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"')
            print("  然后运行：brew install ffmpeg")
            return False
    elif sys.platform.startswith("linux"):
        try:
            subprocess.check_call(["sudo", "apt-get", "install", "-y", "ffmpeg"])
            print("✓ ffmpeg 安装完成")
            return True
        except subprocess.CalledProcessError:
            print("✗ ffmpeg 安装失败，请手动运行：sudo apt install ffmpeg")
            return False
    elif sys.platform == "win32":
        # Try winget first (built into Windows 10+), then choco
        if shutil.which("winget"):
            try:
                subprocess.check_call(["winget", "install", "Gyan.FFmpeg", "--silent", "--accept-package-agreements", "--accept-source-agreements"])
                print("✓ ffmpeg 安装完成（via winget）")
                print("  提示：重新打开终端后 ffmpeg 命令才会生效")
                return True
            except subprocess.CalledProcessError:
                pass
        if shutil.which("choco"):
            try:
                subprocess.check_call(["choco", "install", "ffmpeg", "-y"])
                print("✓ ffmpeg 安装完成（via choco）")
                return True
            except subprocess.CalledProcessError:
                pass
        print("✗ 未找到 winget 或 choco，请手动安装 ffmpeg：")
        print("  方式 1（推荐）：在 PowerShell 中运行：winget install Gyan.FFmpeg")
        print("  方式 2：从 https://ffmpeg.org/download.html 下载，将 bin/ 目录加入 PATH")
        return False
    else:
        print(f"✗ 未知系统（{sys.platform}），请从 https://ffmpeg.org 下载安装包")
        return False


def download_whisper() -> bool:
    local = _PROJECT_ROOT / "models" / "small.pt"
    if local.exists():
        print("\n✓ Whisper small 模型已存在（本地文件，跳过下载）")
        return True
    print("\n正在下载 Whisper small 模型（约 461MB）...")
    print("  下载源：OpenAI Azure CDN（非 HuggingFace，中国大陆通常可访问）")
    print("  提示：也可将 small.pt 手动放入 models/ 目录以跳过此步骤")
    try:
        import whisper
        whisper.load_model("small")
        print("✓ Whisper small 模型就绪")
        return True
    except Exception as e:
        print(f"✗ Whisper 加载失败：{e}")
        return False


def download_embeddings() -> bool:
    local_path = _PROJECT_ROOT / "models" / "bge-small-zh-v1.5"
    if local_path.exists():
        print("\n✓ Embedding 模型已存在（本地文件，跳过下载）")
        return True
    print("\n正在下载 embedding 模型（约 92MB，从 HuggingFace 镜像）...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        local_path.mkdir(parents=True, exist_ok=True)
        model.save(str(local_path))
        print(f"✓ Embedding 模型已下载并保存至 models/bge-small-zh-v1.5/")
        return True
    except Exception as e:
        print(f"✗ Embedding 模型加载失败：{e}")
        return False


if __name__ == "__main__":
    mirror = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    print(f"=== GlaskUser 初始化 ===\n使用镜像：{mirror}\n")

    # Step 1: Python version
    py_ok = check_python()
    if not py_ok:
        sys.exit(1)

    # Step 2: Install dependencies
    deps_ok = install_deps()
    if not deps_ok:
        sys.exit(1)

    # Step 3: Create data directory
    create_data_dir()

    # Step 4: ffmpeg
    ffmpeg_ok = check_ffmpeg()

    # Step 5: Download models
    whisper_ok = download_whisper()
    embed_ok = download_embeddings()

    print()
    if deps_ok and whisper_ok and embed_ok:
        if not ffmpeg_ok:
            print("所有 AI 模型已就绪（ffmpeg 未安装，音频文件暂无法转录）。")
        else:
            print("所有组件已就绪。现在可以运行 /glaskuser_build")
    else:
        print("部分步骤未完成，请按上方提示处理后重试。")
        sys.exit(1)
