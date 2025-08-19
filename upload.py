from roboflow import Roboflow


vers = 4
rf = Roboflow(api_key="FNAxkOK3wxWDdNuLEzRA")
project = rf.workspace("yolotest-lafak").project("bottles-8bf70")
version = project.version(vers)

project.version(vers).deploy(model_type="yolov11", model_path=f"runs\\detect\\bottles1n_results")