#!/bin/sh
for d in $1/*; do 
    python learning/pytorch/predict.py --verbose --save-embed --extend  --model ithemal-models/hsw.dump  --save-cycles  --model-data ithemal-models/hsw.mdl --files $d
done
