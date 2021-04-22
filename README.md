# SLTnet: Short-Term Anchor Linking and Long-Term Self-Guided Attention for Video Object Detection

Daniel Cores, Vícor M. Brea, Manuel Mucientes

## Abstract

We present a new network architecture able to take advantage of spatio-temporal information available in videos to boost  object detection precision. First, box features are associated and aggregated by linking proposals that come from the same \emph{anchor box} in the nearby frames. Then, we design a new attention module that aggregates short-term enhanced box features to exploit long-term spatio-temporal information. This module takes advantage of geometrical features in the long-term for the first time in the video object detection domain. Finally, a spatio-temporal double head is fed with both spatial information from the reference frame and the aggregated information that takes into account the short- and long-term temporal context. We have tested our proposal in five video object detection datasets with very different characteristics, in order to prove its robustness in a wide number of scenarios. Non-parametric statistical tests show that our approach outperforms the state-of-the-art.

This implementation is based on [Detectron2](https://github.com/facebookresearch/detectron2).

## ImageNet VID results

We provide the models and configuration files to reproduce the results obtained in the paper.

Method | Mode | mAP<sub>@0.5</sub> | download
--- | --- | --- | ---
FPN-X101 baseline | Sequential | 78.6 | [model](https://nubeusc-my.sharepoint.com/:u:/g/personal/daniel_cores_usc_es/EeBGZ5L40YVJtwPdSWPf55UBlydctdPYZS49T0ZIJEzwmg?e=P24dgM) \| [config](https://nubeusc-my.sharepoint.com/:u:/g/personal/daniel_cores_usc_es/EcgoonuFLgRLpqRF1cWW_bEBlHPsYHofbDk6hoOAm78J4Q?e=Re3jpQ)
SLTnet FPN-X101 | Sequential | 81.3 | [model](https://nubeusc-my.sharepoint.com/:u:/g/personal/daniel_cores_usc_es/ETB5ZhhBT2RJjZ5aS54V7ecBcUVnAoJ3TL8uoycOTxM1YA?e=WKBOcN) \| [config](https://nubeusc-my.sharepoint.com/:u:/g/personal/daniel_cores_usc_es/EY5wdgQO4-NPngL3cFDsIyIB6hKCec9QM6LbIQsN4BNsQg?e=XxwLC5)
SLTnet FPN-X101 | Symmetric | 81.9 | [model](https://nubeusc-my.sharepoint.com/:u:/g/personal/daniel_cores_usc_es/ETB5ZhhBT2RJjZ5aS54V7ecBcUVnAoJ3TL8uoycOTxM1YA?e=WKBOcN) \| [config](https://nubeusc-my.sharepoint.com/:u:/g/personal/daniel_cores_usc_es/EXSaLPd3QwRAnx253rGgWfkBzuok1jcsykzgt4YsdPXRBA?e=ngVoA3)

## Setup

We provide a Docker image definition to run our algorithm. The image can be built as follows:

```bash
docker build -t detectron2-st:pytorch-cuda10.1-cudnn7 docker/detectron2_spatiotemporal
```

To train and test our network, ImageNet VID and ImageNet DET datasets are required. VID and DET annotations in a format compatible with our implementation can be downloaded from:

* [vid_val](https://nubeusc-my.sharepoint.com/:u:/g/personal/daniel_cores_usc_es/ERZQiD4HacRIivqhX8bjkkgBDxzw5Yt-X-0FDg2gWm9eXw?e=QA3ZNp)
* [det_train_subsampled](https://nubeusc-my.sharepoint.com/:u:/g/personal/daniel_cores_usc_es/EeyGAWuN-5NHmMyxzp4qgbUBi5cpbKPfTgvnIuTXFD1P5g?e=XO9BJi)
* [det_train_subsampled (images converted into short static videos)](https://nubeusc-my.sharepoint.com/:u:/g/personal/daniel_cores_usc_es/EekX6aiOLoROl_Olm5WC50wBl0M7XfGjk4ZKTiCVTURusw?e=z1dJfi)
* [vid_train_split0](https://nubeusc-my.sharepoint.com/:u:/g/personal/daniel_cores_usc_es/EfgAIZbGf0VIrBQ3WCtpORUBI7jXm9D_xKxOn3KcrFfFCQ?e=lm6qY2)
* [vid_train_split1](https://nubeusc-my.sharepoint.com/:u:/g/personal/daniel_cores_usc_es/EcCPcTw50XhDvD6oAoU2Dc0Bvsw9Smn2iriZ1xVD182yew?e=MhaZ7G)
* [vid_train_split2](https://nubeusc-my.sharepoint.com/:u:/g/personal/daniel_cores_usc_es/EfebOMz2-TVIvywtjvTAD_wBvHKvwBhjkUKFRN76VoGZ_A?e=uNSyWb)
* [vid_train_split3](https://nubeusc-my.sharepoint.com/:u:/g/personal/daniel_cores_usc_es/EbhvKjr6iZZGpRvO15-3P3QBW4gPrOGuZI3wAPESMzWoNA?e=ae5xQg)

## Training

To train the spatio-temporal network, we reuse the baseline weights keeping them frozen. Therefore, we first need to train our baseline running:

```bash
cd SLTnet
docker run --gpus all --rm -it -v $PWD:/workspace/detectron -v $datasets_dir:/datasets -v $models_dir:/models detectron2-st:pytorch-cuda10.1-cudnn7 python3 /workspace/detectron/tools/train_net.py --config-file $CONFIG_FILE OUTPUT_DIR /models/$DIRECTORY
```

Datasets definitions can be changed in spatiotemporal/data/dataset.py to set the correct image root directory and annotation paths. The final model can be found in OUTPUT_DIR/model_final.pth. However, this checkpoint also contains iteration number and other extra information apart from the weights. To initialize the spatio-temporal network we need to generate a new file that only contains the model (see <https://github.com/facebookresearch/detectron2/issues/429>):

```python
model = torch.load('OUTPUT_DIR/model_final.pth')
model['model']
```

Finally, the spatio-temporal network can be trained running:

```bash
docker run --gpus all --rm -it -v $PWD:/workspace/detectron -v $datasets_dir:/datasets -v $models_dir:/models detectron2-st:pytorch-cuda10.1-cudnn7 python3 /workspace/detectron/tools/train_net.py --config-file $CONFIG_FILE MODEL.WEIGHTS /models/$BASELINE_MODEL OUTPUT_DIR /models/$DIRECTORY SPATIOTEMPORAL.NUM_FRAMES 3 SPATIOTEMPORAL.FORWARD_AGGREGATION true
```

## Testing

To evaluate the network in the test susbset, use:

```bash
docker run --gpus all --rm -it -v $PWD:/workspace/detectron -v $datasets_dir:/datasets -v $models_dir:/models detectron2-st:pytorch-cuda10.1-cudnn7 python3 /workspace/detectron/tools/train_net.py --eval-only --config-file $CONFIG_FILE MODEL.WEIGHTS /models/$WEIGHTS_DIRECTORY/model_final.pth OUTPUT_DIR /models/$DIRECTORY
```

Our implementation reports the COCO style AP. To calculate the AP with the ImageNet oficial [Development kit](http://vision.cs.unc.edu/ilsvrc2015/download-videos-3j16.php), the output results can be converted running (inside a Docker container):

```bash
python3 tools/convert_output_to_vid.py
```

## Use Custom Datasets

A new dataset can be registered in spatiotemporal/data/dataset.py adding a new entry to *splits* following this format:

```bash
 _DATA_DIR = "/datasets" 

 ...

"vid_val": ( # dataset name
    "vid/ILSVRC/Data/VID",  # image root directory from _DATA_DIR
    "vid/annotations_pytorch/vid_val.json" # json annotations file from _DATA_DIR
)
```

### ST-COCO Dataset Format

We use a modified version of the COCO format dataset to support video datasets called ST-COCO. The main differences are:

* images: images are ordered by video and frame number in the annotation file.
  * video: video to which the image belongs.
  * frame_number: frame number in the video.
* annotations
  * id_track: 'trackid' field in the original ImageNet VID annotation files.

ST-COCO example:

```json
{
    'info': {}
    'images': [
        {
            'file_name': 'val/ILSVRC2015_val_00051001/000000.JPEG',
            'frame_number': 0,
            'height': 720,
            'id': 0,
            'video': 'val/ILSVRC2015_val_00051001',
            'width': 1280
        },
        ...
    ],
    
    'annotations':[
        {
            'area': 410130,
            'bbox': [0, 85, 651, 630],
            'category_id': 8,
            'id': 0,
            'id_track': '0',
            'ignore': 0,
            'image_id': 0,
            'iscrowd': 0,
            'occluded': '0'
        },
        ...
    ],

    'categories': [
        {'id': 0, 'name': 'airplane', 'supercategory': 'airplane'},
        ...
    ]
}
```

## Citing SLTnet

```
@article{CORES2021104179,
    title = {Short-term anchor linking and long-term self-guided attention for video object detection},
    journal = {Image and Vision Computing},
    pages = {104179},
    year = {2021},
    issn = {0262-8856},
    doi = {https://doi.org/10.1016/j.imavis.2021.104179},
    author = {Daniel Cores and Víctor M. Brea and Manuel Mucientes}
}
```

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.