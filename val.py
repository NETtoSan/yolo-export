from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns


import pandas as pd


mod_name = "bottles8-v11_6gflops_640_results"

model = YOLO(f"./runs/detect/{mod_name}/weights/best.pt")

if __name__ == "__main__":
    results = model.val(data="./yolov11/bottles-v8/data.yaml", imgsz=640, batch=16, plots=True)

    names = list(model.names.values())

    print(names)
    print(f"map50: {results.box.map50:.3f} {mod_name}")

    '''
    cm = results.box.map50
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=names, yticklabels=names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion')
    plt.tight_layout()
    plt.show()
    '''