!/bin/bash

set -e

jupyter nbconvert --to notebook --execute Car_price_Feature_Pipeline.ipynb
jupyter nbconvert --to notebook --execute Car_Price_Batch_Inference_Pipeline.ipynb
