<div align="center">

# Contrastive 3D Protein Prediction

 [Angelo Nazzaro](https://github.com/angelonazzaro), [Luigina Costante](https://github.com/Luigina2001)
</div>

# Table of Contents

1. [Introduction](#introduction)
2. [Results](#results)
3. [Installation Guide](#installation-guide)
   - [Installing Python](#installing-python)
   - [Cloning the Repository](#cloning-the-repository)
   - [Creating the Virtual Environment](#creating-the-virtual-environment)
   - [Installing Requirements](#installing-requirements)
4. [Training and Fine-Tuning](#training-and-fine-tuning)
5. [Evaluating](#evaluating)
6. [Citation](#citation)

# Introduction 
In this research, we explore a contrastive approach called C3DPNet, that combines Graph Neural Networks (GNNs) models to analyze 3D structures and natural language models, 
such as DNABERT-2, to process primary sequences in order to to identify relevant features that correlate protein sequences and structures. 

<div align="center">
 <img src="https://github.com/angelonazzaro/contrastive-3d-protein-prediction/assets/58223071/3df604bd-35a3-4501-a994-70fc82527552" alt="c3dp-1" style="width:50%;">
</div>

The model uses a GNN-based graph-encoder to extract structural information from graphs and a DNABERT-2-based text-encoder to acquire semantic 
information and contextual relationships from the DNA sequence.
For the graph-encoder, we explored several architectures: GraphSAGE, GCN, GIN, and GAT.

The representations extracted from the encoders are then approached through the use of the InfoNCE contrastive loss. 
## Results 

In order to ascertain the effectiveness of the integration of the contrastive approach, we carried out a comparison between our best model and the corresponding GNN baseline. In this specific case, 
we compared C3DPNet's GraphSAGE encoder its correspondive baseline. The two models were trained and evaluated on the ENZYMES dataset.

The Table below shows the results of the comparison; where the GraphSAGE baseline highlights almost random performance, 
our model demonstrates superior discriminative capabilities, highlighting the effectiveness of the contrastive approach employed.

| Model    | Accuracy | Precision | F1-Score | Recall |
|----------|----------|-----------|----------|--------|
| C3DPNet  | **0.542** | **0.613** | **0.572** | **0.667** |
| GSAGE    | 0.508    | 0.376     | 0.402    | 0.499  |


# Installation Guide
To install the necessary requirements for the project, please follow the steps below.

## Installing Python
Verify you have Python installed on your machine. The project is compatible with Python `3.9`.

If you do not have Python installed, please refer to the official [Python Guide](https://www.python.org/downloads/).

## Cloning the Repository 
To clone this repository, download and extract the `.zip` project files using the `<Code>` button on the top-right or run the following command in your terminal:
```shell 
git clone https://github.com/angelonazzaro/contrastive-3d-protein-prediction.git
```

## Creating the Virtual Environment 
It's strongly recommended to create a virtual environment for the project and activate it before proceeding. 
Feel free to use any Python package manager to create the virtual environment. However, for a smooth installation of the requirements we recommend you use `pip`. Please refer to [Creating a virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment).

You may skip this step, but please keep in mind that doing so could potentially lead to conflicts if you have other projects on your machine. 
## Installing Requirements
To install the requirements, please: 
1. Make sure you have **activated the virtual environment where you installed the project's requirements**. If activated, your terminal, assuming you are using **bash**, should look like the following: ``(name-of-your-virtual-environment) user@user path``

2. Install the project requirements using `pip`:
```shell 
pip install -r requirements.txt
```

# Training and fine-tuning 
To train and finetune the model, run the `train_c3dp.py` with desired parameters:

1. **--seed** (type: int, default: 42): Seed for random number generation to ensure reproducibility.

2. **--dataset_name** (type: str, default: "proteins"): Name of the dataset to be used.

3. **--data_root_dir** (type: str, default: current working directory + "/data"): Root directory where the dataset is stored.

4. **--experiment_dir** (type: str, default: current working directory + "/experiments"): Directory to store experiment-related files.

5. **--checkpoint_path** (type: str, default: None): Path to a checkpoint file to resume training from a specific point.

6. **--graph_model** (type: str, default: "GraphSAGE"): Type of graph neural network model to use.

7. **--dna_embeddings_pool** (type: str, default: "mean"): Pooling strategy for DNA embeddings.

8. **--graph_embeddings_pool** (type: str, default: "mean"): Pooling strategy for graph embeddings.

9. **--out_features_projection** (type: int, default: 768): Dimensionality of output features for projection.

10. **--use_sigmoid** (action: "store_true", default: False): Flag to use sigmoid activation function.

11. **--in_channels** (type: int, default: None): Number of input channels.

12. **--hidden_channels** (type: int, default: 10): Number of hidden channels.

13. **--num_layers** (type: int, default: 3): Number of layers in the model.

14. **--dim_embedding** (type: int, default: 128): Dimensionality of embeddings for DiffPool.

15. **--gnn_dim_hidden** (type: int, default: 64): Dimensionality of GNN embeddings for DiffPool.

16. **--dim_embedding_MLP** (type: int, default: 50): Dimensionality of MLP embeddings for DiffPool.

17. **--max_num_nodes** (type: int, default: 2699): Maximum number of nodes to consider for DiffPool.

18. **--depth** (type: int, default: None): Number of nodes to consider when pooling for GraphUNet.

19. **--out_channels** (type: int, default: None): Size of each output sample for GraphUNet.

20. **--training_split_percentage** (type: float, default: defined in `training.constants`): Percentage of data used for training.

21. **--val_split_percentage** (type: int, default: defined in `training.constants`): Percentage of data used for validation.

22. **--batch_size** (type: int, default: defined in `training.constants`): Batch size for training.

23. **--shuffle** (action: "store_false", default: True): Flag to control whether to shuffle the dataset during training.

24. **--n_epochs** (type: int, default: defined in `training.constants`): Number of training epochs.

25. **--project_name** (type: str, default: "c3dp"): Name of the WandB project.

26. **--run_name** (type: str, default: None): Name of the WandB run (experiment).

27. **--sweep_config** (type: str, default: current working directory + "/c3dp_sweep.yaml"): Path to the sweep configuration file.

28. **--sweep_count** (type: int, default: 10): Number of runs to execute when hyperparameter tuning.

29. **--early_stopping_patience** (type: int, default: 5): Patience parameter for early stopping during training.

30. **--early_stopping_delta** (type: float, default: 0.0): Minimum change in the monitored quantity to qualify as an improvement.

31. **--optimizer** (type: str, default: "AdamW"): Optimizer to use for training.

32. **--lr_scheduler** (type: str, default: "LinearLR"): Learning rate scheduler to use.

33. **--learning_rate** (type: float): Learning rate for training. Must be set if not tuning hyperparameters.

34. **--weight_decay** (type: float): Weight decay for training. Must be set if not tuning hyperparameters.

35. **--tune_hyperparameters** (action: "store_true", default: False): Flag to indicate whether to perform hyperparameter tuning.

# Evaluating 

To evaluate the model, run `evaluate_c3dp.net` with the desired parameters: 
1. **--seed** (type: int, default: 42): Seed for random number generation to ensure reproducibility.

2. **--scores_dir** (type: str, default: current working directory + "/eval_results"): Directory to store evaluation results.

3. **--scores_file** (type: str, default: "scores.tsv"): File name for storing evaluation scores.

4. **--model_checkpoint** (type: str, required): Path of the model checkpoint to be evaluated.

5. **--dataset** (type: str, default: "proteins"): Dataset to be used for evaluation ("proteins", "fold", or "enzymes").

6. **--model_basename** (type: str, default: None): Base name for the model (used in exporting scores).

7. **--training_split_percentage** (type: float, default: defined in `training.constants`): Percentage of data used for training.

8. **--val_split_percentage** (type: int, default: defined in `training.constants`): Percentage of data used for validation.

9. **--batch_size** (type: int, default: defined in `training.constants`): Batch size for evaluation.

## Citation 

If you have have found this work useful and have decided to include it in your work, please consider citing
```BibTeX
@online{rsDatasetsHub,
    author={Angelo Nazzaro, Luigina Costante}, 
    title = {Contrastive 3D Protein Prediction},
    url={https://github.com/angelonazzaro/contrastive-3d-protein-prediction},
    year={2024}
}
```
