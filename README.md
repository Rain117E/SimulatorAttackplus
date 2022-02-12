# SimulatorAttackplus

This repo is based on CVPR 2021 Ma's Simulator Attack. The reference is [here](https://github.com/machanic/SimulatorAttack).

# Requirement
Pytorch 1.4.0 or above, torchvision 1.3.0 or above, bidict, pretrainedmodels 0.7.4, opencv-python

# Folder structure
```
+-- configures
|   |-- meta_simulator_attack_conf.json  # the hyperparameters setting of simulator attack
|   |-- Bandits.json  # the hyperparameters setting of Bandits attack
|   |-- prior_RGF_attack_conf.json  # the hyperparameters setting of RGF and P-RGF attack
|   |-- meta_attack_conf.json  # the hyperparameters setting of Meta Attack
|   |-- NES_attack_conf.json  # the hyperparameters setting of NES Attack
|   |-- SWITCH_attack_conf.json  # the hyperparameters setting of SWITCH Attack
+-- dataset
|   |-- standard_model.py  # the wrapper of standard classification networks, and it converts the input image's pixels to the range of 0 to 1 before feeding.
|   |-- defensive_model.py # the wrapper of defensive networks, and it converts the input image's pixels to the range of 0 to 1 before feeding.
|   |-- dataset_loader_maker.py  # it returns the data loader class that includes 1000 attacks images for the experiments.
|   |-- npz_dataset.py  # it is the dataset class that includes 1000 attacks images for the experiments.
|   |-- meta_two_queries_dataset.py  # it is the dataset class that trains the Simulator.
|   |-- meta_img_grad_dataset.py  # it is the dataset class that trains the auto-encoder meta-learner of Meta Attack.
+-- meta_simulator_bandits
|   +-- learning
|       +-- script
|           |-- generate_bandits_training_data_script.py   # it can generate the training data that is generated by using Bandits to attack multiple pre-trained networks.
|       |-- train.py  # the main class for training the Simulator.
|       |-- meta_network.py  # the wrapper class of the meta network, and it can transform any classification network to the meta network in meta-learning.
|       |-- meta_distillation_learner.py  # it includes the main procedure of meta-learning.
|       |-- inner_loop.py  # it includes the inner update of meta-learning.
|   +-- attack
|       |-- meta_model_finetune.py  # it includes the class used for fine-tuning the Simulator in the attack.
|       |-- simulate_bandits_attack_shrink.py  # it includes the main procedure of Simulator Attack.
+-- cifar_models   # this folder includes the target models of CIFAR-10, i.e., PyramidNet-272, GDAS, WRN-28, and WRN-40 networks.
+-- tiny_imagenet_models   # this folder includes the target models of TinyImageNet, e.g., DenseNet and ResNeXT
+-- xxx_attack  # other attacks for the compared experiments in the paper.
|-- config.py   # the main configuration of Simulator Attack, remember to modify PY_ROOT to be the project's folder path in your machine environment.
```
# How to attack
Option 1: attack PN272, GDAS, WRN-28, and WRN-40 one by one with one command line untargeted.

`python simulate_bandits_attack_shrink1_for_untargeted.py --gpu 0 --norm linf --epsilon 0.031372 --batch-size 100 --dataset CIFAR-10 --data_loss cw
 --distillation_loss mse --meta_arch resnet34 --test_archs`

Option 2: attack PN272, GDAS, WRN-28, and WRN-40 one by one with one command line targted

`python simulate_bandits_attack_shrink_kmeans_for_targeted.py --gpu 0 --norm l2 --epsilon 4.6 --batch-size 100 --dataset CIFAR-100 --data_loss cw --distillation_loss mse --meta_arch resnet34 --test_archs --targeted`

Option 3: attack defensive model of adversarially trained ResNet-50:

`python simulate_bandits_attack_shrink1_for_untargeted.py --gpu 0 --norm linf --dataset CIFAR-10 --data_loss cw --distillation_loss mse --meta_arch resnet34 --arch resnet50 --attack_defense --defense_model adv_train --batch-size 100`

Folder named `logs` will be generated when attack begins. The log and experimental result file (a `.json` file that includes all queries and the success rate) will be saved to the `logs` folder.`
