from ultralytics import YOLO


model = YOLO("./files/abu-newrtv8-150.pt")


if __name__ == "__main__":
    # Validate the model
    results = model.val(data="D:\\stuff\\opencv\\yolo-export\\yolov8\\opencv_abu-32\\data.yaml", task="detect", save_json=True, plots=True)
    #results = model.val(data="files\\data.yaml", conf=0.25, iou=0.65, task="detect", save_json=True, plots=True)