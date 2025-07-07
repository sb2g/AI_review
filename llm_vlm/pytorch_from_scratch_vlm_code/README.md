# Overview:
- This directory talks about Vision Transformers (ViTs) & Vision Language Models (VLMs)
- It shows 2 approaches:
    - Writing a model from scratch in Pytorch. 
        - This is shown in "smallvlm_nanovlm_model"
    - How to inherit and locally modify Huggingface transformers models to build custom models/losses in Pytorch and train them. 
        - This is shown in "largevlm_aria_model"
        - README_customize_hf_models.md" goes over some of details on this topic.

# Note on Attention
- What are the q, k, v in different architectures usually?
    - In self attention: 
        - q, k, v all come from the same input
    - In cross attention (e.g. encoder-decoder models):
        - q comes from decoder
        - k, v come from encoder

# Vision Language Models (VLMs)
- Key components/hyperparameters of VLMs
    - Vision Encoder / Vision Transformers (ViTs)
        - Key components/hyperparameters of ViTs
            - Image resolution
            - Patch size
            - Depth (number of layers in encoder)
            - Width (internal rep dimension)
            - MLP dim (MLP hidden dimension)
        - Some BKMs
            - MLP dimension should be scaled faster than depth, and depth faster than width. [1]
            - Can update image resolution by finetuning only Attention layers [13]
        - Some models reviewed before writing this up
            - Shape optimized VIT [1], ViT BKMs [13]
            - Siglip [4] [5]
    - Modality projector
    - Language model (LLM) 
- Some models / writeups reviewed before writing this up
   - General:
        - VLM overview [9] [10]
        - nanoVLM [2]: Siglip-85M + SmolLM2-135M. #Params:
        - Idefics2, Idefics3 VLM [11]: Siglip-SO-400M + Mistral7B, Siglip-SO-400M + Llama3.1. 
        - Gemma3 VLM [7]: Siglip-SO-400M + GemmaLLM. #Params: 4B
        - Molmo VLM [12]: Clip + Olmo/Olmoe/Qwen2LLM. #Params: 1B, 2B, ..
    - Following are trained for Videos as well
        - Aria VLM [8]: Custom ViT (with Siglip) + LLM (from scratch). #Params: 3.9B/25.3B, 
        - Qwen2.5 VLM [6]: ViT(from scratch) + Qwen2.5 LLM. #Params: 3B, 7B, 72B
        - InternVL2.5 [15]: InternViT + Qwen2.5/InternLM2.5. #Params: 2.2B, 3.7B, 8B, .. , InternVL3 [15]: InternViT + Qwen2.5/InternLM2.5. #Params: 1B, 3B, 8B,
        - SmolVLM, SmolVLM2 [3]: Siglip-SO-400M + SmolLM2 (updated from Idefics). #Params: 256M, 512M, 2B
        - Llama3.2 [16]: #Params: 11B, 90B;  Llama 4 [16]: #Params: 17B
        - Apollo LMM [14]: Video sampling, etc.. #Params: 3B

# References
1. SO ViT: https://arxiv.org/pdf/2305.13035
2. nanoVLM: https://github.com/huggingface/nanoVLM/tree/main
3. SmolVLM: https://huggingface.co/blog/smolvlm , SmolVLM2: https://huggingface.co/blog/smolvlm2, https://arxiv.org/pdf/2504.05299
4. Siglip: https://huggingface.co/docs/transformers/en/model_doc/siglip
5. Siglip: https://arxiv.org/pdf/2303.15343
6. Qwen2.5 VLM, LLM: https://arxiv.org/pdf/2502.13923, https://arxiv.org/pdf/2412.15115
7. Gemma3: https://arxiv.org/pdf/2503.19786
8. Aria VLM: https://arxiv.org/pdf/2410.05993
9. VLMs overview: https://huggingface.co/blog/vlms-2025
10. VLMs overview: https://magazine.sebastianraschka.com/p/understanding-multimodal-llms
11. Idefics3 VLM: https://huggingface.co/docs/transformers/en/model_doc/idefics3
12. Molmo VLM: https://www.arxiv.org/pdf/2409.17146 , Olmoe: https://arxiv.org/pdf/2409.02060
13. ViT BKMs: https://arxiv.org/pdf/2203.09795
14. Apollo LMM: https://arxiv.org/pdf/2412.10360
15. InternVL2.5, InternVL3: https://arxiv.org/pdf/2412.05271, https://arxiv.org/pdf/2504.10479
16. Llama3: https://github.com/meta-llama/llama-models/tree/main , https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/
