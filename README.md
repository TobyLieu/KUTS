## Setup the Environment
This software was implemented a system running `Ubuntu 16.04.4 LTS`, with `Python 3.7.6`, `PyTorch 1.8.1`, and `CUDA 11.4`. We have tried to reduced the number of dependencies for running the code. Nonetheless, you still need to install some necessary packages, such as `sklearn`.

You can adjust the batch size to adapt to your own hardware environment. Personally, we recommend the use of four NVIDIA GPUs.

## Code Description
The main architecture of KUTS lies in the `models/` folder. The `modeling.py` is the main backbone of KUTS. Please refer to each file to acquire more implementation details. 

`run.sh` includes the running script, which is:
```
python -W ignore main.py --CLS 4 --BSZ 256 --DATA_DIR ./data --SET_TYPE test --NUM_EPOCHS 20 --DEVICE 0
```
**Parameter description**:

`--CLS`: number of diseases.

`--BSZ`: batch size.

`--DATA_DIR`: location of data.
```