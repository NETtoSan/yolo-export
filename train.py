from ultralytics import YOLO

if __name__ == "__main__":
    custdir = "./yolov11/bottlesv10"
    model = YOLO("yolo11n.pt")#.load("./runs/detect/bottles8-v11_3gflops_results/weights/last.pt") #load a pretrained model (recommended for training)
    model.info()
    #info_list = model.info(detailed=True, verbose==True)

    
    # Train the model+
    results = model.train(
        data=f"{custdir}/data.yaml",
        epochs=50,                 # Reduce epochs
        imgsz=480,                # Smaller image size
        batch=16,                 # Increase batch size if GPU allows
        name="bottles10-11n-480-16",
        optimizer="AdamW",
        save_period=10,
        amp=True
    )
    

    #model2 = YOLO("yolo11n")
    #results = model2.train(data=f"{custdir}\\data.yaml", epochs=150, imgsz=640, optimizer="adam", name="abu-31v11n-150_results", dynamic=True, task="detect")