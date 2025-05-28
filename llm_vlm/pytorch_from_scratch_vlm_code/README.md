# Overview:
- This directory talks about Vision Transformers (ViTs) & Vision Language Models (VLMs)
- It also shows how to inherit and locally modify Huggingface transformers models to build custom models/losses in Pytorch and train them

# Vision Language Models (VLMs)
- Key components/hyperparameters of VLMs
    - Vision Encoder / Vision Transformers (ViTs)
        - Key components/hyperparameters of ViTs
            - Image resolution
            - Patch size
            - Depth (number of layers in encoder)
            - Width (internal rep dimension)
            - MLP dim (MLP hidden dimension)
        - Some models reviewed before writing this up
            - Shape optimized VIT [1], ViT BLMs [13]
            - Siglip [4] [5]
    - Modality projector
    - Language model (LLM) 
- Some models / writeups reviewed before writing this up
    - VLM overview [9] [10]
    - nanoVLM [2] = Siglip-85M + SmolLM2-135M
    - SmolVLM, SmolVLM2 [3] = Siglip-SO-400M + SmolLM2 
    - Idefics2, Idefics3 [11] = Siglip-SO-400M + Mistral7B, Siglip-SO-400M + Llama3.1
    - Gemma3 [7] = Siglip-SO-400M + GemmaLLM
    - Qwen2.5 [6] = ViT(from scratch) + LLM(from scratch)
    - Aria [8] = Custom ViT (with Siglip) + LLM (from scratch)
    - Molmo [12] = Clip + Olmo/Olmoe/Qwen2LLM 

# References
1. SO ViT: https://arxiv.org/pdf/2305.13035
2. nanoVLM: https://github.com/huggingface/nanoVLM/tree/main
3. SmolVLM: https://huggingface.co/blog/smolvlm , SmolVLM2: https://huggingface.co/blog/smolvlm2, https://arxiv.org/pdf/2504.05299
4. Siglip: https://huggingface.co/docs/transformers/en/model_doc/siglip
5. Siglip: https://arxiv.org/pdf/2303.15343
6. Qwen2.5: https://arxiv.org/pdf/2502.13923
7. Gemma3: https://arxiv.org/pdf/2503.19786
8. Aria: https://arxiv.org/pdf/2410.05993
9. VLMs: https://huggingface.co/blog/vlms-2025
10. VLMs: https://magazine.sebastianraschka.com/p/understanding-multimodal-llms
11. Idefics3: https://huggingface.co/docs/transformers/en/model_doc/idefics3
12. Molmo: https://www.arxiv.org/pdf/2409.17146 , Olmoe: https://arxiv.org/pdf/2409.02060
13. ViT BKMs: https://arxiv.org/pdf/2203.09795
