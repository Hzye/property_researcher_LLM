import os
from pathlib import Path
import yt_dlp
import whisper
import re
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

CHANNEL_NAMES = config["channels"]
K = config["num_videos"]
DOWNLOAD_DIR = Path(config["download_dir"])

def sanitize_filename(title):
    """Remove characters that are invalid for Windows/ffmpeg from a Youtube title."""
    return re.sub(r'[<>:"/\\|?*$]', '', title)

def download_recent_videos(channel_name, k=3):
    """Download the k most recent videos from a youtube channel.

    Args:
        channel_name (str): Youtube channel name.
        k (int, optional): Number of recent videos to download. Defaults to 3.

    Returns:
        downloaded_files (list[str]): Downloaded and sanitized audio file names.
    """
    channel_url = f"https://www.youtube.com/@{channel_name}/videos"
    downloaded_files = []
    ydl_opts = {
        "format": "bestaudio/best",
        "quiet": True,
        "noplaylist": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "playlistend": k
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
        videos = info["entries"] if "entries" in info else [info]

        for video in videos[:k]:
            safe_title = sanitize_filename(video['title'])
            filename = DOWNLOAD_DIR / f"{safe_title}.mp3"
            text_filename = DOWNLOAD_DIR / f"{safe_title}.txt"
            if text_filename.exists():
                print(f"Skipping already downloaded: {safe_title}")
            else:
                print(f"Downloading: {safe_title}")
                ydl_opts["outtmpl"] = str(DOWNLOAD_DIR / f"{safe_title}.%(ext)s")
                with yt_dlp.YoutubeDL(ydl_opts) as inner_ydl:
                    inner_ydl.download([video["webpage_url"]])
            downloaded_files.append(filename)

    return downloaded_files

def transcribe_audio_files(audio_files, model_name="small"):
    """Transcribe 

    Args:
        audio_files (list[str]): List of audio file names.
        model_name (str, optional): Whisper transcription model to use. Defaults to "small".

    Returns:
        transcripts (dict): Generated transcripts.
    """
    model = whisper.load_model(model_name)
    transcripts = {}

    for audio_file in audio_files:
        transcript_file = audio_file.with_suffix(".txt")
        if transcript_file.exists():
            print(f"Skipping already transcribed: {audio_file.name}")
            with open(transcript_file, "r", encoding="utf-8") as f:
                transcripts[audio_file.name] = f.read()
        else:
            print(f"Transcribing: {audio_file.name}")
            result = model.transcribe(str(audio_file))
            transcripts[audio_file.name] = result["text"]
            with open(transcript_file, "w", encoding="utf-8") as f:
                f.write(result["text"])

    return transcripts

def remove_mp3s(folder=DOWNLOAD_DIR):
    """Remove mp3s from download directory.

    Args:
        folder (Path, optional): Download folder name. Defaults to DOWNLOAD_DIR.
    """
    count = 0
    for file in folder.glob("*.mp3"):
        try:
            file.unlink()
            print(f"Deleted: {file.name}")
            count += 1
        except Exception as e:
            print(f"Error deleting {file.name}: {e}")
    print(f"\nDeleted {count} mp3 files.")

if __name__ == "__main__":
    for channel in CHANNEL_NAMES:
        # download and transcribe
        print(f"--- Transcribing {channel} videos ---")
        audio_files = download_recent_videos(channel, k=K)
        transcripts = transcribe_audio_files(audio_files)
    # remove audio files
    remove_mp3s(DOWNLOAD_DIR)