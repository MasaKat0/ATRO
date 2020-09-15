### Training (Binary)
Use `scripts/train_binary.py` to train the network. Example usage:
```bash
# Example usage
cd scripts
python train_binary.py --dataset cifar10 --log_dir ../logs/train --cost 0.3
```

We also provide cript generator for training by [ABCI](https://abci.ai/) which is the world's first large-scale open AI computing infrastructure.
Use `scripts/experiments/train_binary_abci.py` to generate shell scripts for ABCI. 
If user specify cost, at_norm, at_eps, script generator makes only scripts for training on such values.
If the values are not specified, the generator takes for loop in some range of values.   
Example usage:
```bash
# Example usage
cd scripts
python experiments/train_binary_abci.py -d cifar10 -l ../logs/train --script_root ../logs/abci_script --run_dir . --abci_log_dir ../logs/abci_log --user ${your_abci_user_id} --env ${abci_conda_environment} --cost 0.3