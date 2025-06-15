# Neuro-Ins

Neuro-Ins is a learning based framework for solving the one-shot node insertion for dynamic routing problems. This repo implements our paper:

Zhiqin Zhang, Jingfeng Yang, Zhiguang Cao, and Hoong Chuin Lau, "[Neuro-Ins: A Learning-based One-shot Node Insertion for Dynamic Routing Problems](https://www.)" in the IEEE Transactions on Knowledge and Data Engineering. Please cite our paper if the work is useful to you.

```

``` 

## Dependencies
* Python>=3.8
* PyTorch>=1.7
* tensorboard_logger
* tqdm

## Usage

### Training

Here, we take the instance with 14 static nodes and 6 dynamic nodes to be inserted (DPDP_14_6) as an example:

```bash
python run.py 
--train_dataset
./datasets/pdp_7_3.pkl
--problem
pdtsp
--graph_size
20
--sta_orders
7
--max_grad_norm
0.05
--val_dataset
./datasets/pdp_7_3_val.pkl
--run_name
'example_training_DPDP_7_3'
--K_epochs
10
```

For other instances, please replace the corresponding values accordingly.


### Inference

To load the model and perform inference, simply add the following after the training step:

```bash
--eval_only 
--load_path '{add model to load here}'
```

## Acknowledgements
The code and the framework are derived from the repos [yining043/PDP-N2S](https://github.com/yining043/PDP-N2S).
