%cd /content
!rm -rf yolov5
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
!pip install -r requirements.txt
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="r5QTR45EghQWBDqU9udL")
project = rf.workspace("archaeology").project("artifacts-z1lar")
version = project.version(3)
dataset = version.download("yolov5")
!ls Artifacts-3
!python train.py \
  --img 416 \
  --batch 8 \
  --epochs 50 \
  --data /content/yolov5/Artifacts-3/data.yaml \
  --weights yolov5s.pt \
  --name artifact_yolov5 \
  --project runs/train \
  --exist-ok
!python detect.py \
  --weights runs/train/artifact_yolov5/weights/best.pt \
  --img 416 \
  --conf 0.25 \
  --source /content/yolov5/Artifacts-3/valid/images \
  --project runs/detect \
  --name artifact_results \
  --exist-ok
import glob
from IPython.display import Image, display

for img in glob.glob('/content/yolov5/runs/detect/artifact_results/*.jpg')[:5]:
    display(Image(filename=img))

!python val.py \
  --weights runs/train/artifact_yolov5/weights/best.pt \
  --data /content/yolov5/Artifacts-3/data.yaml \
  --img 416 \
  --iou 0.5 \
  --task test

