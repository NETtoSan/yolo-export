from ultralytics.utils.benchmarks import benchmark as Benchmark
from pathlib import Path


yolo_ver = "yolov8"
model = f"files/abu-newrtv8-150.pt"
custdir = "d:\\stuff\\opencv\\yolo-export\\yolov8\\opencv_abu_new-1"

if __name__ == "__main__":
    Benchmark(model=Path(model), data = f"{custdir}\\data.yaml", imgsz=640, half=False, device="cpu")
    #Benchmark(model=model, data = f"{custdir}\\data.yaml", imgsz=640, half=False, device="cuda")