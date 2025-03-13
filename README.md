<h1 align="center">
Integrating Synaptic Synergy Prompt and Optimized Prototype Classifiers to Enhance Few-Shot Class-Incremental Learning<br/>
</h1>

## Abstract
Few-Shot Class-Incremental Learning (FSCIL) aims to enable deep neural networks to learn new tasks incrementally using only a small number of samples, while retaining previously acquired knowledge, thereby mimicking human learning patterns. With the rise of prompt-based learning, dual-prompt mechanisms have been introduced into the FSCIL domain. In this paper, we propose a novel approach called the Synaptic Synergy Prompt (SSP), which balances the relationship between the Overall Prompt (OP) and Incremental Prompt (IP). Specifically, OP focuses more on global knowledge, while IP concentrates on incremental tasks, allowing both components to perform their respective roles effectively. In the classification phase of FSCIL, prototype classifiers are commonly employed. To enhance the model's performance in classification tasks, we made two key improvements. First, we applied a centering operation in the feature space to increase inter-class feature distances and encourage class features to align more closely with their respective prototype vectors. Second, we incorporated a global self-attention mechanism into the prototype classifier to enable the model to more effectively utilize global information. We validated the effectiveness of our approach on widely used benchmark datasets, CIFAR-100 , CUB-200 and MiniImageNet, where it achieved highly competitive performance, demonstrating its potential for applications in scenarios with limited access to high-quality data.



## Environment
run `pip install -r requirements.txt`

## Datasets
The required datasets have been placed under the `data/` folder.

## Training scripts
- CUB-200
```
python main.py fscil_cub200 --model vit_base_patch16_224 --batch-size 25 --d_prompt_length 10 --length 10 --data-path ./data --output_dir ./output
```


- CIFAR-100
```
python main.py fscil_cifar100 --model vit_base_patch16_224 --batch-size 25 --d_prompt_length 10 --length 10 --data-path ./data --output_dir ./output
```

- Mini-ImageNet
```
python main.py fscil_miniImageNet --model vit_base_patch16_224 --batch-size 25 --d_prompt_length 5 --length 5 --data-path ./data --output_dir ./output
```
