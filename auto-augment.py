from ultralytics.data.annotator import auto_annotate


auto_annotate(data="./yolov11/bottles/test/images", det_model="./files/bottles8-640.pt", sam_model="mobile_sam.pt")
auto_annotate(data="./yolov11/bottles/valid/images", det_model="./files/bottles8-640.pt", sam_model="mobile_sam.pt")