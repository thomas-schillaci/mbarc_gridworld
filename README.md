# MBARC (Gridworld)

Official repository of the *Model-Based Aware Reward Classification* (MBARC) thesis (Gridworld version).

- [Installation](#installation)
- [How to use](#how-to-use)

## Installation

This program uses **python 3.7**, **CUDA 10.2** if enabled, and was tested on Ubuntu 20.04.1.

Run the following command to install the dependencies:
  ```shell script
  pip install torch==1.7.1  gym==0.15.7 gym-minigrid==1.0.2 tqdm
  ```

### Install wandb (optional)

You can use [wandb](https://www.wandb.com/) to track your experiments:
```shell script
pip install wandb
```

To use wandb, pass the flag `--use-wandb` when running the program. See [How to use](#how-to-use) for more details about flags.

## How to use

CUDA is enabled by default, see the following section to disable it.

To run the program, run the following command from the folder containing the `mbarc` package:
```shell script
python -m mbarc
```

### Disable CUDA

To disable CUDA, pass the flag `--device cpu` to the command line. See the next section for more information about flags.

### Flags

You can pass multiple flags to the command line, a summary is printed at launch time.
The most useful flags are described in the following table:

| Flag | Value | Default | Description |
| ---- | ----- | ------- | ----------- |
| --device | Any string accepted by [torch.device](https://pytorch.org/docs/stable/tensor_attributes.html#device-doc) | cuda | Sets the PyTorch's device |
| --env-name | Any game name (without the suffixes) as depicted [here](https://github.com/maximecb/gym-minigrid#included-environments) | MiniGrid-SimpleCrossingS9N2-v0 | Sets the gym environment | 
| --strategy | "online", "class_balanced", "square_root", "progressively_balanced" or "mbarc" | mbarc | The training strategy to use |

The following boolean flags are set to `False` if not passed to the command line:

| Flag | Description |
| ---- | ----------- |
| --use-modified-model | Setting this flag will replace MBARC's proposed reward prediction head with SimPLe's | 
| --use-wandb | Enables [wandb](https://www.wandb.com/) to track the experiment |

For example, to execute the program without CUDA, run:
```shell script
python -m mbarc --device cpu
```
