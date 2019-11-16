# Analyze Leed Pattern

## Summary

卒業研究におけるプロジェクトです。

LEEDと呼ばれる電子線回折装置を用いて撮影された画像から、スポットを自動で検出し、解析を行います。

## Install
```
git clone git@github.com:fhiroki/analyze-leed-pattern.git
cd analyze-leed-pattern
pip install -e .
```

## Commands
```
usage: leed [-h] [--version] {plot-detected-image,detect-base-blob} ...

analyze leed pattern

positional arguments:
  {plot-detected-image,detect-base-blob}
    plot-detected-image
                        see `-h`
    detect-base-blob    see `-h`

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
```

### plot-detected-image
```
usage: leed plot-detected-image [-h] --input-image INPUT_IMAGE
                                [--output-image OUTPUT_IMAGE]

optional arguments:
  -h, --help            show this help message and exit
  --input-image INPUT_IMAGE
                        input image path
  --output-image OUTPUT_IMAGE
                        output image path

ex) plot-detected-image --input-image
data/Coronene_Ag111/image/Coronene/L16501.tif --output-image output/images
```

### detect-base-blob
```
usage: leed detect-base-blob [-h] --input-dir INPUT_DIR
                             [--output-dir OUTPUT_DIR] --kind {Au,Ag,Cu}
                             --surface {110,111} --voltages VOLTAGES
                             [--isplot] [--manual-r MANUAL_R]

optional arguments:
  -h, --help            show this help message and exit
  --input-dir INPUT_DIR
                        input images directory
  --output-dir OUTPUT_DIR
                        output image directory
  --kind {Au,Ag,Cu}     base type
  --surface {110,111}   base surface
  --voltages VOLTAGES   beam voltages (ex. 1.0,2.0,3.0)
  --isplot              draw a scatter plot of sinθ and X
  --manual-r MANUAL_R   calculated r by myself

ex) detect-base-blob --kind Ag --surface 111 --voltages
80.6,94.7,109.2,122.9,136.0,150.9,159.1,179.2,193.7,215.3,230.3,252.7,264.9
--input-dir data/Coronene_Ag111/image/Ag111/ --isplot --output-dir output/r
```
