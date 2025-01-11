# Optimizing Deep Neural Networks for Resource-Limited Embedded Systems

This repository provides the implementation and experimentation code used to optimize deep neural networks (DNNs) for deployment on resource-constrained embedded systems, specifically targeting the GAP9 processor, presented in the thesis *Optimizing Deep Neural Networks for Resource-Limited Embedded Systems*.

The repository is organized as follows:

- **`drone_bandit/`**
  - **`server_simulation/`**  
    Files for the cloud implementation of the edge-to-cloud strategy.
  - **`context_vectors_extraction/`**  
    Scripts to extract context vectors for DroneBandit.
  - **`model_sequentializer/`**  
    Code to convert functional models into sequential blocks.

- **`one_task_one_ee_model/`**
  - **`results/`**  
    Code and scripts for results extraction of the model.
  - **`training/`**  
    Files for training the considered model.

- **`three_tasks_one_ee_model/`**
  - **`early_exit/`**  
    Files for results extraction and training for the model with early exits.
  - **`knowledge_distillation/`**  
    Files for results extraction and training for the model with knowledge distillation.

- **`two_task_two_ee_model/`**
  - Similar structure for results extraction and training files.

... 


### Description of key directories

- **`drone_bandit/`**  
  Contains files for the DroneBandit project, including:
  - Implementation of the edge-to-cloud server.
  - Scripts to extract context vectors for use with DroneBandit.
  - Code for creating functional models as sequential blocks.

- **Other directories**  
  Each directory corresponds to a specific model used for early exits, knowledge distillation, or special quantization. These directories include:
  - Code for training the models.
  - Scripts used to obtained the results presented in the thesis.
