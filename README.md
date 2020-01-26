# Analyze Leed Pattern

## Summary

卒業研究におけるプロジェクトです。

LEEDと呼ばれる電子線回折装置を用いて撮影された画像から、スポットを自動で検出し、解析を行います。

## Install
```
git clone git@github.com:fhiroki/analyze-leed-pattern.git
cd analyze-leed-pattern
pip install -e .
```

## Commands
```
usage: leed [-h] [--version] {plot-detected-spot,calc-rprime,plot-dinverse} ...

analyze leed pattern

positional arguments:
  {plot-detected-spot,calc-rprime,plot-dinverse}
    plot-detected-spot  see `-h`
    calc-rprime         see `-h`
    plot-dinverse       see `-h`

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
```

### detect-spot
```
usage: leed detect-spot [-h] --input-image-path INPUT_IMAGE_PATH [--output-image-path OUTPUT_IMAGE_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --input-image-path INPUT_IMAGE_PATH
                        input image path
  --output-image-path OUTPUT_IMAGE_PATH
                        output image path

ex) detect-spot --input-image-path images/L16501.tif --output-image-path output/images/L16501_detected.tif
```

### calc-rprime
```
usage: leed calc-rprime [-h] --input-images-dir INPUT_IMAGES_DIR
      --input-voltages-path INPUT_VOLTAGES_PATH --kind {Au,Ag,Cu} --surface {110,111}
      [--isplot] [--output-image-path OUTPUT_IMAGE_PATH] [--manual-r MANUAL_R]

optional arguments:
  -h, --help            show this help message and exit
  --input-images-dir INPUT_IMAGES_DIR
                        input images directory
  --input-voltages-path INPUT_VOLTAGES_PATH
                        input image, beam voltage csv file
  --kind {Au,Ag,Cu}     base type
  --surface {110,111}   base surface
  --isplot              draw a scatter plot of sinθ and X
  --output-image-path OUTPUT_IMAGE_PATH
                        output plot image path
  --manual-r MANUAL_R   calculated r by myself

ex) calc-rprime --kind Ag --surface 111 --input-images-dir image/Ag111/
  --input-voltages-path voltages.csv --isplot --output-image_path output/rprime.png
```

### plot-dinverse
```
usage: leed plot-dinverse [-h] --input-images-dir INPUT_IMAGES_DIR
       --input-voltages-path INPUT_VOLTAGES_PATH [--rprime RPRIME] [--output-image-path OUTPUT_IMAGE_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --input-images-dir INPUT_IMAGES_DIR
                        input images directory
  --input-voltages-path INPUT_VOLTAGES_PATH
                        input image, beam voltage csv file
  --rprime RPRIME       calculated rprime by calc-rprime
  --output-image-path OUTPUT_IMAGE_PATH
                        output plot image path

ex) plot-dinverse --input-images-dir images/Coronene/ --input-voltages-path voltages.csv
  --output-image-path output/dinverse.png
```
