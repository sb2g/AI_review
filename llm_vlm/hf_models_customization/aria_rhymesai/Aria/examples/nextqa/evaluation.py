import argparse
import json
import os
import random

import numpy as np
import torch
from peft import PeftConfig, PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from aria.load_video import load_video
from aria.lora.layers import GroupedGemmLoraLayer
from aria.model import AriaForConditionalGeneration, AriaProcessor, GroupedGEMM

# Add command-line argument parsing
parser = argparse.ArgumentParser(description="NextQA Evaluation")
parser.add_argument(
    "--base_model_path", type=str, required=True, help="Path to the base model"
)
parser.add_argument(
    "--peft_model_path", type=str, default=None, help="Path to the PEFT model"
)
parser.add_argument(
    "--tokenizer_path", type=str, required=True, help="Path to the tokenizer"
)
parser.add_argument(
    "--save_root", type=str, required=True, help="The root path of output."
)
parser.add_argument("--image_size", type=int, default=490, help="Maximum image size")
parser.add_argument(
    "--batch_size", type=int, default=4, help="Batch size for evaluation"
)
parser.add_argument(
    "--num_workers", type=int, default=16, help="Number of workers for data loading"
)

args = parser.parse_args()
os.makedirs(args.save_root, exist_ok=True)


class NextQA_Dataset(Dataset):
    def __init__(self):
        super().__init__()
        annos = "datasets/nextqa/val.jsonl"
        vis_root = "datasets/nextqa"

        self.dataset = []
        lines = open(annos).readlines()
        for line in tqdm(lines):
            anno = json.loads(line.strip())
            anno["video"]["path"] = os.path.join(vis_root, anno["video"]["path"])
            self.dataset.append(anno)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def load_model_and_tokenizer(args):
    processor = AriaProcessor.from_pretrained(
        args.base_model_path, tokenizer_path=args.tokenizer_path
    )
    processor.tokenizer.padding_side = "left"
    tokenizer = processor.tokenizer

    model = AriaForConditionalGeneration.from_pretrained(
        args.base_model_path, device_map="auto", torch_dtype=torch.bfloat16,
        revision="4844f0b5ff678e768236889df5accbe4967ec845"
    ).eval()
    model.pad_token_id = tokenizer.pad_token_id

    if args.peft_model_path:
        peft_config = PeftConfig.from_pretrained(args.peft_model_path)
        custom_module_mapping = {GroupedGEMM: GroupedGemmLoraLayer}
        peft_config._register_custom_module(custom_module_mapping)
        model = PeftModel.from_pretrained(
            model,
            args.peft_model_path,
            config=peft_config,
            is_trainable=False,
            autocast_adapter_dtype=False,
        )

    return model, tokenizer, processor


def process_batch(model, tokenizer, inputs, original_batch, prompts):
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            stop_strings=["<|im_end|>"],
            tokenizer=tokenizer,
        )

    for i, prompt in enumerate(prompts):
        print(f"i: {i}, prompt: {prompt}")
        prompt_len = len(inputs["input_ids"][i])
        output_text = tokenizer.decode(
            output[i][prompt_len:], skip_special_tokens=True
        ).replace("<|im_end|>", "")
        print(f"output_text: {output_text}")
        original_batch[i]["pred"] = output_text

    return original_batch


def collate_fn(batch, processor, tokenizer):
    messages = []
    images = []
    print(f"batch: {batch}")
    for item in batch:
        images.extend(load_video(item["video"]["path"], item["video"]["num_frames"]))
        for message in item["messages"]:
            for cont_idx, cont in enumerate(message["content"]):
                if cont["type"] == "video":
                    del message["content"][cont_idx]
                    for img_i in range(item["video"]["num_frames"]):
                        insert_item = {
                            "text": None,
                            "type": "image",
                        }
                        message["content"].insert(cont_idx + img_i, insert_item)
        messages.append(item["messages"])
    print(f"messages: {messages}")

    texts = [
        processor.apply_chat_template(msg, add_generation_prompt=True)
        for msg in messages
    ]
    print(f"texts: {texts}")
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding="longest",
        max_image_size=args.image_size,
    )
    return inputs, batch, texts


def main():
    model, tokenizer, processor = load_model_and_tokenizer(args)

    dataset = NextQA_Dataset()
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_fn(batch, processor, tokenizer),
    )

    results = []
    for batch in tqdm(dataloader, desc="Processing batches"):
        inputs, original_batch, prompts = batch
        results.extend(process_batch(model, tokenizer, inputs, original_batch, prompts))

    with open(f"{args.save_root}/nextqa_result.json", "w") as fo:
        json.dump(results, fo, indent=4, ensure_ascii=False)
    return results


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


def eval_multi_choice(gold_i, pred_i):
    """
    Evaluate a multiple choice instance.
    """
    correct = False
    # only they are exactly the same, we consider it as correct
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:  # gold_i is a string
        if gold_i == pred_i:
            correct = True
    return correct


def evaluate(samples):
    pred_correct = 0
    for sample in samples:
        gold_i = sample["gt"]
        pred_i = sample["pred"]

        correct = eval_multi_choice(gold_i, pred_i)

        if correct:
            pred_correct += 1

    if len(samples) == 0:
        return {"acc": 0}
    return {"acc": pred_correct / len(samples)}


def get_score(output):
    for out in output:
        pred = out["pred"]
        out["gt"]
        index2ans = out["index2ans"]
        all_choices = out["all_choices"]

        pred = parse_multi_choice_response(pred, all_choices, index2ans)
        out["pred"] = pred

    with open(f"{args.save_root}/nextqa_result_parsed.json", "w") as fo:
        json.dump(output, fo, indent=4, ensure_ascii=False)

    print(evaluate(output))


if __name__ == "__main__":
    output = main()
    get_score(output)
