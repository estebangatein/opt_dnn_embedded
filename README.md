# Optimizing Deep Neural Networks for Resource-Limited Embedded Systems

This repository provides the implementation and experimentation code used to optimize deep neural networks (DNNs) for deployment on resource-constrained embedded systems, specifically targeting the GAP9 processor, presented in the thesis *Optimizing Deep Neural Networks for Resource-Limited Embedded Systems*.

The repository is organized as follows:

├── drone_bandit/  
│   ├── server_simulation/    # files for the cloud implementation of the edge-to-cloud strategy  
│   ├── context_vectors_extraction/    # scripts to extract context vectors for DroneBandit  
│   ├── model_sequentializer/   # code to convert functional models as sequential blocks  
│
├── one_task_one_ee_model/    # includes the code for the results extraction and the training of the considered model
|   ├── results/
|   ├── training/
│
├── three_tasks_one_ee_model/  
│   ├── early_exit/    # files for results and training of the model for early exits
                ...
│   ├── knowledge_distillation/    # files for results and training of the model for knowledge distillation
                ...
│
...    # same structure for the other models
│
└── two_task_two_ee_model/        


### Description of Key Directories

- **`drone_bandit/`**  
  Contains files for the DroneBandit project, including:
  - Implementation of the edge-to-cloud server.
  - Scripts to extract context vectors for use with DroneBandit.
  - Code for creating functional models as sequential blocks.

- **Other directories**  
  Each directory corresponds to a specific model used for early exits, knowledge distillation, or special quantization. These directories include:
  - Code for training the models.
  - Scripts used to obtained the results presented in the thesis.
