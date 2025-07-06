# Overview
- This directory shows how to customize Huggingface transformers models

# Loading Models
- 2 types of models:
    - Auto class: A barebones model that outputs hidden state. E.g. AutoModel, LlamaModel.
    - Model specific class: A model with a specific head attached for performing specific tasks. E.g. AutoModelForCausalLM or LlamaForCausalLM.
        ```
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype="auto", device_map="auto")
        model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype="auto", device_map="auto")
        ```
- API
    - Use same API to load different models. Example:
        ```
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
        ```
    - Use similar API for different tasks. Example:
        ```
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-2-7b-hf")
        model = AutoModelForQuestionAnswering.from_pretrained("meta-llama/Llama-2-7b-hf")
        ```
- Loading pretrianed models/weights
    - How to get model's pretrained weights from HF hub or local dir? 
        - Use `.from_pretrained()`
    - What happens under the hood?
        - Steps:
            - Create model with random weights
            - Load pretrained weights
            - Assign pretrained weights to model
        - Need memory to hold 2 copies of model
        - How does HF handle this for large models?
            - Fast initialization: Skips random weight initialization (`_fast_init`)
            - Sharded checkpoints: Checkpoints are sharded (e.g. max 5GB) when saved (`max_shard_size`). During loading, these shards are loaded sequentially, so max shard size is the max memory used. 
            - HF accelerate's Big model inference (`device_map = "auto"`): 1) Creates only model skeleton with no real data like random weights, so no 2 copies of weights when loading pretrained. 2) Weights are loaded & dispatched across all devices efficiently (GPU, CPU, disk in that order. Model has hooks per weight to transfer weights from CPU to GPU when that layer is used and back to CPU when layer is not in use)
            - Data type (`torch.dtype = torch.float16|"auto"`): Avoids loading the weights 2 times, first in torch.float32 and then in desired. Use `"auto"` to load weights in data type that they were saved in.
    - How to load Custom models:
        - Custom models can be built on top of transformers config & modeling classes. But the modeling code is not from tranformers library. Hub includes scanning the repo, but be careful with security. 
        - Needs `trust_remote_code=True` to load them. Use commit hash to load required version.
        - Example:
            ```
            commit_hash = "ed94a7c6247d8aedce4647f00f20de6875b5b292"
            model = AutoModelForImageClassification.from_pretrained(
                "sgugger/custom-resnet50d", trust_remote_code=True, revision=commit_hash
            )
            ```   
        

# Customize Models
- General policy:
    - Transformers library follows single model single file policy.
        - All forward pass (i.e. inference) code for a model is in modeling_<model>.py in <model> subfolder under https://github.com/huggingface/transformers/tree/main/src/transformers/models 
- How to customize a model
    - The <model> subfolder has 2 important files. Copy these files and modify them as needed 
        - `configuration_<>`.py
        - `modeling_<>`.py
    - Configuration:
        - This is needed to build the model. 
        - Need to subclass `PretrainedConfig` and pass all `kwargs` to it. Enables functionalities like `from_pretrained()`, `save_pretrained()`, `push_to_hub()`, etc. functions
            ```
            class ResnetConfig(PretrainedConfig):
                model_type = "resnet"
                def __init__(
                    self,
                    layers: List[int] = [3, 4, 6, 3],
                    num_classes: int = 1000,
                    ...
                    **kwargs,
                ):
                    self.layers = layers
                    self.num_classes = num_classes
                    ...
                    super().__init__(**kwargs)
            ```
        - How to write config to JSON in model folder
            ```
            resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
            resnet50d_config.save_pretrained("custom-resnet")            ```

    - Model
        - Create and customize the model.
        - Need to subclass `PreTrainedModel` and pass `config` object to `__init__()`. Enables functionalities like saving & loading models.
        - Return dictionariy with losses to be compatible with `Trainer`
            ```
            class ResnetModel(PreTrainedModel):
                config_class = ResnetConfig
                def __init__(self, config):
                    super().__init__(config)
                    self.model = ResNet(
                        config.layers,
                        num_classes=config.num_classes,
                        ...
                    )
                def forward(self, tensor):
                    return self.model.forward_features(tensor)
            ```
    - How to enable the model to be loaded with AutoClass APIs - AutoModel or AutoModelFor[Task]
        ```
        AutoConfig.register("resnet", ResnetConfig)  # should match model_type variable
        AutoModel.register(ResnetConfig, ResnetModel) # should match config_class variable
        AutoModelForImageClassification.register(ResnetConfig, ResnetModelForImageClassification)
        ```
    - How to save model
        - Model's `save_pretrained()` calls `PreTrainedModel` function, which calls `PretrainedConfig` function, so that both model weights & config are saved together.
        - `PreTrainedModel.save_pretrained()` automatically calls `PretrainedConfig.save_pretrained()` so that both the model and configuration are saved together.
        - Model is saved to a model.safetensors file and a configuration is saved to a config.json file.
    - How to upload model to HF Hub to share with others?
        - Follow this model directory structure
            ```
            resnet_model
                - __init__.py  (allows <model> to be used as module)
                - configuration_resnet.py
                - modeling_resnet.py
            ```
        - Steps & Commands
            - Import
                ```
                from resnet_model.configuration_resnet import ResnetConfig
                from resnet_model.modeling_resnet import ResnetModel, ResnetModelForImageClassification
                ```
            - Register to allow `save_pretrained()` with AutoClass API
                ```
                ResnetConfig.register_for_auto_class()
                ResnetModel.register_for_auto_class("AutoModel")
                ResnetModelForImageClassification.register_for_auto_class("AutoModelForImageClassification")
                ```
            - Create config, model & load model weights
                ``` 
                resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
                resnet50d = ResnetModelForImageClassification(resnet50d_config)
                pretrained_model = timm.create_model("resnet50d", pretrained=True)
                resnet50d.model.load_state_dict(pretrained_model.state_dict())
                ```
            - Push to hub
                ``` 
                huggingface-cli login
                resnet50d.push_to_hub("custom-resnet50d")
                ```
- How to add new model to Transformer's library? Two options:
    - Define full model
        - Create your own repo's model directory structure (similar to customizing the model)
            ```
            resnet_model
                - __init__.py  (allows <model> to be used as module)
                - configuration_resnet.py
                - modeling_resnet.py
            ```
        - New models require a configuration, for example `BrandNewLlamaConfig`, that is stored as an attribute of `PreTrainedModel`.
            ```
            model = BrandNewLlamaModel.from_pretrained("username/brand_new_llama")
            model.config
            ```
        - Clone transformers library, make changes, verify/debug it works and push to hub
            ```
            # Dev environment
            git clone https://github.com/[your Github handle]/transformers.git
            cd transformers
            git remote add upstream https://github.com/huggingface/transformers.git
            python -m venv .env
            source .env/bin/activate
            pip install -e ".[dev]" or pip install -e ".[quality]"

            # Create initial pull request
            cd transformers
            transformers add-new-model-like  # Will create the files for model, config, image processor, processor
            git checkout -b add_brand_new_bert
            git add .
            git commit
            git fetch upstream
            git rebase upstream/main
            git push -u origin a-descriptive-name-for-my-changes
            git fetch upstream
            git merge upstream/main

            # Check forward pass
            model = BrandNewLlamaModel.load_pretrained_checkpoint("/path/to/checkpoint/")
            input_ids = [0, 4, 5, 2, 3, 7, 9]  # vector of input ids
            original_output = model.generate(input_ids)

            # Make code changes to the files

            # Model initialization
            model = BrandNewLlama(BrandNewLlamaConfig())
            def _init_weights(self, module):
                """Initialize the weights"""
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                    if module.bias is not None:
                        module.bias.data.zero_()
                ...

            # Convert pretrained checkpoints to transformers:
            - Use conversion script: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/convert_bart_original_pytorch_checkpoint_to_pytorch.pys

            # Save model to local folder
            model.save_pretrained("/path/to/converted/checkpoint/folder")

            # Check forward pass & make sure 
            model = BrandNewLlamaModel.from_pretrained("/path/to/converted/checkpoint/folder")
            input_ids = [0, 4, 4, 3, 2, 4, 1, 7, 19]
            output = model.generate(input_ids).last_hidden_states
            torch.allclose(original_output, output, atol=1e-3)

            # Implement tokenizer and compare from_pretrained, loading checkpoint give same tokens
            input_str = "This is a long example input string containing special characters .$?-, numbers 2872 234 12 and words."
            model = BrandNewLlamaModel.load_pretrained_checkpoint("/path/to/checkpoint/")
            input_ids = model.tokenize(input_str)
            tokenizer = BrandNewLlamaTokenizer.from_pretrained("/path/to/tokenizer/folder/")
            input_ids = tokenizer(input_str).input_ids

            # Implement Image processor
            - Converts images to model input format
            - Fast image processors use GPU for processing.
            transformers add-fast-image-processor --model-name your_model_name
            - Add tests for the image processor in tests/models/your_model_name/test_image_processing_your_model_name.py.

            # Implement processor
            - The processor centralizes the preprocessing of different modalities before passing them to the model.
            def __call__(
                self,
                images: ImageInput = None,
                text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
                audio=None,
                videos=None,
                **kwargs: Unpack[YourModelProcessorKwargs],
            ) -> BatchFeature:
                ...

            # Add tests
            - Add model tests in tests/models/brand_new_llama/test_modeling_brand_new_llama.py
            - Add tests for the image processor in tests/models/your_model_name/test_image_processing_your_model_name.py.
            - Add tests for the processor in tests/models/your_model_name/test_processor_your_model_name.py. 

            # Add documentation in model card and model code docstrings
            - In model card: docs/source/model_doc/brand_new_llama.md
            - In model docstrings: src/transformers/models/brand_new_llama/modeling_brand_new_llama.py

            # Check style & quality
            make style
            make quality

            # Upload to hub
            - Convert and upload all checkpoints to the
            brand_new_bert.push_to_hub("brand_new_llama")

            # Merge the git pull request

            ```

    - Use Modular transformers
        -  Significantly reduces the code required to add a model by allowing imports and inheritance.
        - A modular file contains model, processor, and configuration class code that would otherwise be in separate files under the single model, single file policy.
        - A linter “unravels” the modular file into a modeling.py file to preserve the single model, single file directory structure (modeling, processor, etc.). Inheritance is flattened to only a single level.
            ```
            python utils/modular_model_converter.py --files_to_parse src/transformers/models/<your_model>/modular_<your_model>.py
            ```
        - You should be able to write everything (tokenizer, image processor, model, config, etc.) in a modular and their corresponding single-files are generated.
        - Classes that are a dependency of an inherited class but aren’t explicitly defined are automatically added as a part of dependency tracing. Example if mlp is same in new model, and is called, modular adds this to `modeling_<>.py`
            ```
            from ..olmo.modeling_olmo import OlmoMLP
            class Olmo2MLP(OlmoMLP):
                pass
            ```
        - Confirm the content matches
            ```
            python utils/check_modular_conversion.py --files src/transformers/models/<your_model>/modular_<your_model>.py
            ```
        - Example RoBERTa from BERT
            ```
            from ..bert.configuration_bert import BertConfig
            from ..bert.modeling_bert import (
                BertModel,
                BertEmbeddings,
                BertForMaskedLM
            )

            # RoBERTa and BERT config is identical
            class RobertaConfig(BertConfig):
                model_type = 'roberta'

            # Redefine the embeddings to highlight the padding id difference, and redefine the position embeddings
            class RobertaEmbeddings(BertEmbeddings):
                def __init__(self, config):
                    super().__init__(config())
                    self.padding_idx = config.pad_token_id
                    self.position_embeddings = nn.Embedding(
                        config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
                    )

            # RoBERTa and BERT model is identical except for the embedding layer, which is defined above, so no need for additional changes here
            class RobertaModel(BertModel):
            def __init__(self, config):
                super().__init__(config)
                self.embeddings = RobertaEmbeddings(config)

            # The model heads now only need to redefine the model inside to `RobertaModel`
            class RobertaForMaskedLM(BertForMaskedLM):
            def __init__(self, config):
                super().__init__(config)
                self.model = RobertaModel(config)
            ```
        - Example Olmo2 from Olmo
            - https://huggingface.co/docs/transformers/modular_transformers

- How to customize model layer / components (i.e. pytorch modules)
    - Define a new component class by subclassing the original model component. Define `__init__()`, `forward()`. If needed, define hook functions (say `hook_func`) for loading pretrained weights and call it in `__init__()` using `self.__register_load_state_dict_pre_hook(hook_func)`
        ```
        class SamVisionAttentionSplit(SamVisionAttention, nn.Module):  
            def __init__(self, config, window_size):  
                super().__init__(config, window_size)  
                ...  
                self._register_load_state_dict_pre_hook(self.split_q_k_v_load_hook)  
        ```
    - Load the model and update the component with new component.
        ```
        model = SamModel.from_pretrained("facebook/sam-vit-base")
        for layer in model.vision_encoder.layers:
            if hasattr(layer, "attn"):
                layer.attn = SamVisionAttentionSplit(model.config.vision_config, model.config.vision_config.window_size)
        ```

# Preprocessors
- Tokenizers
    - Convert text into subwords (tokens) & convert them to tensors (inputs ids to model). Also returns attention mask to tell which tokens should be attended.
    - Some import
- Image processors
    - Image processors converts images into pixel values, tensors that represent image colors and size. The pixel values are inputs to a vision model. To ensure a pretrained model receives the correct input, an image processor can perform the following operations to make sure an image is exactly like the images a model was pretrained on.
        - center_crop() to resize an image
        - normalize() or rescale() pixel values
- Video processors
    - A Video Processor is a utility responsible for preparing input features for video models, as well as handling the post-processing of their outputs. It provides transformations such as resizing, normalization, and conversion into PyTorch.
    - Currently, if using base image processor for videos, it processes video data by treating each frame as an individual image and applying transformations frame-by-frame. While functional, this approach is not highly efficient. Using AutoVideoProcessor allows us to take advantage of fast video processors, leveraging the torchvision library. Fast processors handle the whole batch of videos at once, without iterating over each video or frame. These updates introduce GPU acceleration and significantly enhance processing speed, especially for tasks requiring high throughput.
- Backbones
    - Higher-level computer visions tasks, such as object detection or image segmentation, use several models together to generate a prediction. A separate model is used for the backbone, neck, and head. The backbone extracts useful features from an input image into a feature map, the neck combines and processes the feature maps, and the head uses them to make a prediction.
- Feature extractors
    - Feature extractors preprocess audio data into the correct format for a given model. It takes the raw audio signal and converts it into a tensor that can be fed to a model. The tensor shape depends on the model, but the feature extractor will correctly preprocess the audio data for you given the model you’re using. Feature extractors also include methods for padding, truncation, and resampling.
    - Pass the audio signal, typically stored in array, to the feature extractor and set the sampling_rate parameter to the pretrained audio models sampling rate. It is important the sampling rate of the audio data matches the sampling rate of the data a pretrained audio model was trained on.
- Processors

# References
- https://huggingface.co/docs/transformers/models
- https://huggingface.co/docs/transformers/fast_tokenizers