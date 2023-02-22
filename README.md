# APRIORICS

[![Documentation Status](https://readthedocs.org/projects/apriorics/badge/?version=latest)](https://apriorics.readthedocs.io/en/latest/?badge=latest)

## Install

First, we need to install my fork of [HistoReg](https://github.com/CBICA/HistoReg) that
contains a ready-to-use Dockerfile.

```bash
cd
git clone https://github.com/schwobr/HistoReg.git
docker build -t historeg HistoReg
```

We then need to create a conda environment with pytorch.

```bash
conda create -n apriorics python=3.9
conda activate apriorics
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
conda install openslide -c conda-forge
```

NB: To check required `cudatoolkit` version, type `nvcc --version` in a shell. Cuda is always compatible with older versions of the same major release (for instance if your nvcc version is 11.5, you can install `cudatoolkit=11.3` here). Specific versions of `pytorch` are only available with few `cudatoolkit` versions, you can check it on [PyTorch official website](https://pytorch.org/get-started/locally/).

Make sure that you have blas and lapack installed:

```bash
sudo apt install libblas-dev libblapack-dev # debian-based systems
sudo yum install blas-devel lapack-devel # CentOS 7
sudo dnf install blas-devel lapack-devel # CentOS 8
```

We can then clone this repository and install necessary pip packages.

```bash
cd
git clone https://github.com/schwobr/apriorics.git
cd apriorics
pip install -r requirements.txt
```

You can also install this library as an editable pip package, which will install all dependencies automatically.

```bash
pip install -e .
```

---
# Specific documentation

## Transformers & Coco

### Data

In the context of the [Transformers](https://huggingface.co/docs/transformers/index) library, it was necessary to transform the data in the [COCO format](https://cocodataset.org/#format-data). First, please install the hugginface's transformer library following the official documentation. Then, to transform our dataset (for object detection) in this format:

```
python ./scripts/utils/tococodataset.py
```

### Training

Then, we can use the data created to train the transformers models using `/scripts/train/train_transformers.py`). Three models are supported : `detr`, `deformdabledetr` and `yolos`.

An example of the usage of this script :

```
python train_transformers.py -m detr
```

Transformer models are underperforming on this dataset, compared to Yolo.

## Yolo

### Data

We used [Yolov5](https://github.com/ultralytics/yolov5) and [Yolov8](https://github.com/ultralytics/ultralytics) is also usable. Please don't forget to clone the repository and install the dependencies as showed on the documentation of Yolo. Then, the first step is to transform the data in the Yolo format. To do so:

```
python ./scripts/utils/toyolo.py
```

You must also create a `yaml` file at the root of the yolo folder, following the format provided [here](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#11-create-datasetyaml). In our case, we only have 1 class : `Mitosis`.

### Training

Once we obtained the data, the training script can be used. An example of is the following :

```
python train.py --img 256 --batch 16 --epochs 3 --data yolo_format.yaml --weights yolov5s.pt
```

For more details about the parameters, please refer to the [official documentation](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)

### Inference

If you want to evaluate the model and obtain evaluation measures, use the following script. Different models trained on the data are available at `/data/elliot/runs/train/`. The model with the best result is in `exp12_62 epochs` : 

```
python val.py --weights ./runs/train/exp12_62epochs/weights/best.pt --data yolo_dataset.yaml --img 256
```

If you want to do the inference on unlabeled data, use :

```
python detect.py --weights ./runs/train/exp12_62epochs/weights/best.pt --source /path/to/data
```