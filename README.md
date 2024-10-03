# LLMRecSys_with_KnowledgeDistilation
Set of tools above RecBole framework to experimenting with distillation of knowledge from LLM

# Documentation

Structure of folders could particularly tune by config-file, but the typical project structures like this:
```

.
├── datasets 
│   ├── llm_embeddings
│   │   └── ml-1m # profile embeddings for datasets
│   └── ml-1m  # secific datasets
│       ├── README.md
│       ├── ml-1m.inter
│       ├── ml-1m.item
│       └── ml-1m.user
├── experiments
│   ├── autoint
│   │   ├── ...
│   │   └── ...
│   ├── neumf
│   │   ├── ...some folder with bandles of experiments
│   │   ├── 01_exp_bandle_...
│   │   ├── 02_exp_bandle_...
│   │   ├── 03_exp_bandle_kar_ml1m
│   │   ├── 04_exp_bandle_kar_ml1m
│   │   ├── 05_exp_bandle_kar_avc
│   │   ├── ... or some separated experiment (.yaml)
│   │   ├── exp_1_NeuMF_ml1m_..._.yaml
│   │   ├── exp_2_NeuMF_ACV_..._.yaml
│   │   └── exp_3_NeuMF_ml1m_..._.yaml
├── log
│   └── NeuMF
│       ├── ... 
│       └── ...
├── models
│   ├── losses.py     # implementation of distil losses
│   ├── ...external custom models:
│   ├── ..._.py
│   └── twotowers.py
├── saved
│   ├── .... saved best resulted models: 
│   └── NeuMF-Oct-02-2024_22-12-30.pth
├── utils            # all modules is here
│   ├── journal.py
│   ├── tmanager.py
│   ├── utils.py
│   └── wrapper.py
├── LICENSE
├── README.md
├── poetry.lock      # required dependencies
├── pyproject.toml   # required dependencies
└── run_exps.py      # main enter poit for running
```

The way of running specific experiment or the bunch of experiments which collected in folder:
```
> python run_exps.py ./experiments/path_to_exps/experiment_folder_or_yaml [--start_with=n]
```


## 2. Structure of config part
Configuration and scenario of experiments is setting up by  `.yaml` configuration file. The most typical structure of config file generally comprise of two parts.

```
# config.yaml

###
# Part 1. Setup typical RecBole parameters
###

    # RecBole Model setup
    ...

    # RecBole Dataset processing setup
    ...

### 
# Part 2. Setup for distilation experimenting
###

    # LLM-Profiles processing setup
    ...

    # Scenario of experiment's flow setup

    scenario: # list of commands
    [
        { # scenario command
            'command': 'some_name',
            'params' : 'some_params'
        },

        ...
    ]


```

### LLM-profiles section
This config section can address this main issues:
 - transforming profiles embeddings from given `.json` profiles to `.pt` matrix, or
 - produce random `.pt` matrix for debuging purpose

### Scenario setup 
Can execute following commands:
- `print`
- `set`
- `init_model`
- `reduce_dim`
- `wrap_model`
- `set_config`
- `init_trainer`
- `info`
- `break`
- `set_train_dataset`
- `set_output`
- `set_trainable`
- `test_eval`
- `remove_outputs`

## 2. Common usecase for distillation
### Initial setting
```
# config.yaml

# model configuration part
model: 'NeuMF'  # Model to use; 
mf_embedding_size: 64  # Some settings according to RecBole
mlp_embedding_size: 512
mlp_hidden_size: [256, 128, 64]
dropout_prob: 0.1
...
```

### Preliminary steps of scenario

```
...
scenario:
    [
    # Part 1 Destilation

        {
            # Output information to console
            'command': 'print',
            'params': '-----Part #1: distilation'
        },
        {
            # set the variable which use in output report
            'command': 'set',
            'params': {'train_part': 0}
        },
        {
            # init model base on RecBole logic
            'command': 'init_model',
            'params': None
        },
        {
            # dimentionality reduction
            'command': 'reduce_dim',
            'params': {
                'file_in': 'user1536.pt',  
                'file_out': 'user64.pt', 
                'dimension': 64, 
                'overwrite': True
                }
        },
        {
            # Set the wroper around model, which provide additional methods
            # Set distilation method, which declared in losses.py
            'command': 'wrap_model',
            # Available params: users_emb_file, items_emb_file
            'params': {
                'distil_loss': 'NeumfUserRMSE', 
                'users_emb_file': 'user64.pt',
                'particular': True 
                }
        },
        {
            # change the parameters define in config in a run-time
            'command': 'set_config',
            'params': {'epochs': 2}
        },            
        {
            # init trainer based on RecBole logic
            'command': 'init_trainer',
            'params': None
        },
        ...
```

### Step 1. Get initial information about model structure

For output information about model's strusture there is a `info` command. 

```
    ...
    {
        'command': 'info',
        'params': 1  # will output structure of model,  verbose=1
    },
    {
        # used for debug purpose
        'command': 'break'  # interrupt scenario execution
    }, 
    ...  
```

This instruction will result to:
```
cmd = 'info'
    1 |  | root | NeuMF ()
    2 | ---- | user_mf_embedding | Embedding (84, 64)
    3 | ---- | item_mf_embedding | Embedding (2975, 64)
    4 | ---- | user_mlp_embedding | Embedding (84, 512)
    5 | ---- | item_mlp_embedding | Embedding (2975, 512)
    6 | ---- | mlp_layers | MLPLayers ()
    7 | -------- | mlp_layers | Sequential ()
    8 | ------------ | 0 | Dropout (p=0.1, inplace=False)
    9 | ------------ | 1 | Linear (in_features=1024, out_features=256, bias=True)
   10 | ------------ | 2 | ReLU ()
   11 | ------------ | 3 | Dropout (p=0.1, inplace=False)
   12 | ------------ | 4 | Linear (in_features=256, out_features=128, bias=True)
   13 | ------------ | 5 | ReLU ()
   14 | ------------ | 6 | Dropout (p=0.1, inplace=False)
   15 | ------------ | 7 | Linear (in_features=128, out_features=64, bias=True)
   16 | ------------ | 8 | ReLU ()
   17 | ---- | predict_layer | Linear (in_features=128, out_features=1, bias=True)
   18 | ---- | sigmoid | Sigmoid ()
   19 | ---- | loss | BCEWithLogitsLoss ()
```
So it would give a sense on what layer the `hook` for grabing `input` and/or `output` should be setted.
For sake of example, let's say, that we are interesting in `layer 9`.

### Step 2. Set the hook on specific layer
In order to to access for all inputs and outputs passing through specific layer `set_outputs` command is used.


```
    ...
    {
        'command': 'set_outputs',
        'params': [9]  # could be an list of layers
    },
    ...
```
The data structure, includes input[s] and outout[s] will pass as an input to `distil_loss` method, which was set by `wrap_model` command.


### Step 3. Consequence steps (depends on the experiment's setup)

**Change the layers with trainable/freezen parameters:**

```
    ...
    {
        'command': 'set_trainable',
        'params': [[*], False]  # Freeze all layers
    },
    {
        'command': 'set_trainable',
        'params': [[4, 5, 6, 7], Train]  # Make tranable specific layers
    },  
    ...  
```

**Remove distilation loss**

For stop processing outputs from intermediate layers, and eliminate all hook's function on layers, there is an `remove_outputs` command

```
    ...
    {
        'command': 'remove_outputs',
        'params': None            
    },
    ...
```
