# Overview:
- Training large models on GPUs requires several optimization techniques & distributed training. 
- The model components that consume resources are:
    - Weights
    - Gradients
    - Optimizer states
- Some techniques are:
    - Data parallelism
        - Efficient compute & communication  
        - Poor memory efficiency  
    - Model parallelism
        - Efficient memory efficiency  
        - Poor compute & communication  
    - ZeRO (Zero redundancy optimization)
    - FSDP

# Deepspeed ZeRO
    - 

# References
- https://developer.nvidia.com/blog/mastering-llm-techniques-training/
- 