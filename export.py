from ultralytics import YOLO


if __name__ == "__main__":
    custdir = "./yolov11/bottlesv11/data.yaml"
    #model = YOLO("files/best.pt")  # load a pretrained model (recommended for training)
    model = YOLO("./runs/detect/bottles11-11s-480/weights/best.pt")

    # Train the model
    export = model.export(format="openvino", dynamic= True, imgsz=480, int8=True, data=custdir)
