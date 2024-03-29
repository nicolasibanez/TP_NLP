from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import gc
from typing import List, Dict, Union
import ctypes
from datasets import load_dataset
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import json

STARLING_PROMPT_TEMPLATE = "GPT4 Correct User: {text}<|end_of_turn|>GPT4 Correct Assistant:"
STARLING_MODEL_NAME = "starling"

DEFAULT_MAX_LENGTH = 500
STARTLING_SPLIT_TEXT = 'GPT4 Correct Assistant: '

ENTAILMENT_TEMPLATE = """I want you to solve the task of textual entailment: Given the premise, is the hypothesis correct?
The labels are the following:
0 = entailment
1 = neutral
2 = contradiction
{examples}
Premise = "{premise}" Hypothesis = "{hypothesis}" -> answer = """

def pre_format_prompts(example_data, example_ids: Union[List[int],List[List[int]]],test_data, template: str=ENTAILMENT_TEMPLATE):
    
    test_data_length = len(test_data['label'])
    if isinstance(example_ids[0], int):
        example_ids = [example_ids[0] for _ in range(test_data_length)]
    # print(f'Example ids {len(example_ids)}, test_data {test_data_length}')
    assert len(example_ids)==test_data_length

    examples_list = []
    for subset_ids in example_ids:
        example_data_subset = example_data[subset_ids]

        examples_list.append("\n".join(f'Premise = "{premise}" Hypothesis = "{hypothesis}" -> answer = {label}'
                        for premise, hypothesis, label in 
                        zip(example_data_subset['premise'],example_data_subset['hypothesis'],example_data_subset['label'])
        ))

    return [template.format(
        examples=examples,
        premise=premise,
        hypothesis=hypothesis
    ) for examples, premise, hypothesis in zip(examples_list, test_data['premise'],test_data['hypothesis'])
    ]

def format_prompt(prompts: Union[str, List[str]], system_prompt_template=STARLING_PROMPT_TEMPLATE):
    """works to format system prompt with a single entry
    and a user prompt with a single entry
    """  
    if isinstance(prompts, str):
        prompts = [prompts]
    final_prompts = [system_prompt_template.format(
        text=prompt
    ) for prompt in prompts
    ]
    return final_prompts

def free_memory():
    print(f'Before cleaning: {torch.cuda.memory_reserved(0) // (1000 **2)} MiB')
    # ctypes.CDLL("libc.so.6").malloc_trim(0)
    gc.collect()
    torch.cuda.empty_cache()
    print(f'After cleaning: {torch.cuda.memory_reserved(0) // (1000 **2)} MiB')


def generate_response(
        model,
        tokenizer,
        prompts: List[str], 
        split_text: str=STARTLING_SPLIT_TEXT,
        print_anwser: bool=False,
        max_length: int=DEFAULT_MAX_LENGTH,
        max_new_tokens: int=200,
        do_sample: bool=False,
        temperature: float=1.0,
        top_p: float=1.0,
        output_scores: bool=True,
        tokens_of_interest = (28734, 28740, 28750),
    ) -> List[int]:
    """
    Parameters
    ----------
    split_text: str
        Some models need post processing (e.g. need to split by 
        `GPT4 Correct Assistant: ` with Startling)
        No splitting if None
    """
    device = model.device
    print(f'Before inference {torch.cuda.memory_reserved(0) // (1000 **2)} MiB')

    tokenized_elements = tokenizer(
        prompts, return_tensors="pt", padding='longest'
    )
    input_ids = tokenized_elements.input_ids.to(device)
    attention_mask = tokenized_elements.attention_mask.to(device)
 
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        return_dict_in_generate=True,
        output_scores=output_scores,
    )

    scores = outputs.scores[0][:, tokens_of_interest].cpu().argmax(dim=1)
    
    output_scores = scores.tolist()
    
    # this actually works to free the memory
    del input_ids
    del attention_mask
    del outputs
    del scores

    free_memory()

    return output_scores

def get_predictions_from_llm(model, tokenizer, train_ds, eval_ds, train_ids, eval_ids) -> List[int]:
    # print(f"eval_ids length = {len(eval_ids)}, eval_ds[eval_ids] length {len(eval_ds[eval_ids])}, {len(eval_ds[eval_ids]['label'])}")
    prompts = pre_format_prompts(train_ds, train_ids, eval_ds[eval_ids])
    prompts = format_prompt(prompts)
    # print(f"\n---------------Prompts\n-------------{prompts[:2]}\n\n")
    labels = generate_response(
        model,
        tokenizer,
        prompts,
        split_text=STARTLING_SPLIT_TEXT,
        max_length=500,
        max_new_tokens=1
    )
    return labels



def evaluate_model_with_n_examples(model, tokenizer, dataset, n_examples, is_random=True):

    # n_examples * n_eval = 550 max for RTX3090ti
    max_multiplier = 500 # 100 #550
    # if is_random:
    #     train_ids = random.sample(range(len(dataset['train'])),n_examples)
    # else:
    #     train_ids = list(range(dataset))
    # train_ids = [268344, 520118, 103965, 112978, 462350, 10866]

    label_list = []

    n_eval = max_multiplier // max(1,n_examples)
    test_length = len(dataset['test']) #// 2 # /2 for faster evaluation
    max_iterations = test_length // n_eval +1 
    print(f"n_examples={n_examples}, will be using n_eval={n_eval}, max_iterations={max_iterations}, test length={test_length}")
    for i in tqdm(range(max_iterations)):
    
        eval_ids = list(range(i*n_eval,min((i+1)*n_eval, test_length)))
        train_ids = [random.sample(range(len(dataset['train'])),n_examples) for _ in range(len(eval_ids))]
        # print(f"eval_ids length = {len(eval_ids)}")

        labels = get_predictions_from_llm(model, tokenizer, dataset['train'], dataset['test'], train_ids, eval_ids)
        if len(labels) != len(eval_ids):
            print(f'Mismatch in lengths {len(labels)} vs {len(eval_ids)}')
            labels = [-1]*len(eval_ids)
        label_list.extend(labels)

    
    y_true = dataset['test']['label'][:test_length]
    print(len(y_true),len(label_list))
    if len(label_list)!= len(y_true):
        print(f'Mismatch in lengths {len(y_true)} vs {len(label_list)}')
        y_true = y_true[:len(label_list)]

    total_accuracy = accuracy_score(y_true, label_list)
    print(f'Final accuracy is {total_accuracy}')

    return total_accuracy, y_true, label_list
    

# test_prompts = pre_format_prompts(snli_ds['train'][0:4],snli_ds['test'][0:3])
# test_prompts = format_prompt(test_prompts)
# print(test_prompts[0])
# tokenizer.encode("2",add_special_tokens=False)
# [28734, 28740, 28750]

if __name__=="__main__":
    
    snli_ds = load_dataset("snli")
    #Removing sentence pairs with no label (-1)
    snli_ds = snli_ds.filter(lambda example: example['label'] != -1) 
    # 0: entails, 1: neutral, 2: contradicts, -1: no label

    model_id = "berkeley-nest/Starling-LM-7B-alpha"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", quantization_config=bnb_config
    )

    n_examples = 20
    print(f'Running experiments with {n_examples}-shot prompting')
    accuracy, true_labels, predicted_labels = evaluate_model_with_n_examples(
        model, tokenizer, snli_ds, n_examples=n_examples, is_random=True
    )

    obj_ = {'accuracy':accuracy,
            'labels':true_labels,
            'predictions':predicted_labels}
    
    with open(f'results_{n_examples}.json', 'w') as f:
        json.dump(obj_,f)
