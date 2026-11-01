"""Download public sample traffic videos for testing (needs internet).

  python scripts/download_samples.py

Sources (check each repository's licence before reusing the videos):
  * https://github.com/andrewssobral/vehicle_detection_haarcascades   (highway CCTV clips)
  * https://github.com/intel-iot-devkit/sample-videos                 (street / drone clips)
"""
import urllib.request
from pathlib import Path

FILES = {
    "highway_video1.avi": "https://github.com/andrewssobral/vehicle_detection_haarcascades/raw/master/dataset/video1.avi",
    "highway_video2.avi": "https://github.com/andrewssobral/vehicle_detection_haarcascades/raw/master/dataset/video2.avi",
    "car-detection.mp4": "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4",
}

out = Path(__file__).resolve().parent.parent / "sample_videos"
out.mkdir(exist_ok=True)
for name, url in FILES.items():
    dst = out / name
    if dst.exists():
        print("exists:", dst)
        continue
    print("downloading", name, "...")
    urllib.request.urlretrieve(url, dst)
print("done ->", out)
