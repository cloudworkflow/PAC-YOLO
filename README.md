# PAC-YOLOv5

This is the source code to reproduce *PAC-YOLOv5* proposed in the paper: 

# Cell Detection with Patch-wise Local Spatial Attention and Convolutional Enhancements.

### Deying Liu, Ming Gao*, Jixiang Yu, Wei Xu, Shancheng Jiang, Kai Leung Yung and Wai Huang Ip

If you are interested in this research and wish to use these datasets and codes, please kindly cite our paper or contact the corresponding author(*).

The datasets used in the paper are also accompanied:

[**Plasmodium**](https://aistudio.baidu.com/aistudio/datasetdetail/152739/0)

[**Tuberculosis**](https://www.heywhale.com/mw/dataset/5efc4de063975d002c9792de/content)

[**Complete Blood Count (CBC)**](https://github.com/MahmudulAlam/Complete-Blood-Cell-Count-Dataset)

## Quick Start Examples

### Install

Clone repo and install [requirements.txt](requirements.txt) in a Python>=3.7.0 environment, including PyTorch>=1.7.

```python
git clone https://github.com/cloudworkflow/PAC-YOLO
cd PAC-YOLO
pip install -r requirements.txt
```

### Training

Use the `--batch-size 32` on a GPU of NVIDIA RTX 3090 and the memory is 32GB.

```python
python train.py --data 'data/coco.yaml' --cfg 'models/yolov5-pac.yaml' --weights '' --batch-size 32
``` 

### Inference

Inference refers [YOLOv5](https://github.com/ultralytics/yolov5). 

weights of [PAC-YOLOv5s](pac-yolov5s.pt) download automatically from this repo trained from Plasmodium.

```python
python val.py --data 'data/coco.yaml' --cfg 'models/yolov5-pac.yaml' --weights 'pac-yolov5s.pt' --batch-size 32
```

### Acknowledgement

This implementation is built upon YOLOv5 (https://github.com/ultralytics/yolov5)


