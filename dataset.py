import torch
from torch.utils.data import Dataset


class NLI_Dataset(Dataset):
    def __init__(self, tokenized_datasets, tokenizer, max_len, split = 'train', augmentation = False, size: int = None):
        self.input_ids = None
        self.attention_mask = None
        self.token_type_ids = None

        test = tokenizer('test')

        if size is not None:
            if 'input_ids' in tokenized_datasets[split].features.keys():
                self.input_ids = tokenized_datasets[split]['input_ids'][:size]
            if 'attention_mask' in tokenized_datasets[split].features.keys():
                self.attention_mask = tokenized_datasets[split]['attention_mask'][:size]
            if 'token_type_ids' in tokenized_datasets[split].features.keys():
                self.token_type_ids = tokenized_datasets[split]['token_type_ids'][:size]
            self.labels = tokenized_datasets[split]['label'][:size]
        else:
            if 'input_ids' in tokenized_datasets[split].features.keys():
                self.input_ids = tokenized_datasets[split]['input_ids']
            if 'attention_mask' in tokenized_datasets[split].features.keys():
                self.attention_mask = tokenized_datasets[split]['attention_mask']
            if 'token_type_ids' in tokenized_datasets[split].features.keys():
                self.token_type_ids = tokenized_datasets[split]['token_type_ids']
            self.labels = tokenized_datasets[split]['label']

        self.augmentation = augmentation
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.input_ids[idx]).int()
        attention_mask = torch.tensor(self.attention_mask[idx]).int()
        labels = torch.tensor(self.labels[idx])

        pad_token_id = self.tokenizer.pad_token_id
        sep_token_id = self.tokenizer.sep_token_id
            
        if self.augmentation:
            input_ids = self.transform(input_ids)

        if input_ids.size(0) > self.max_len:
            input_ids = input_ids[:self.max_len]
            attention_mask = attention_mask[:self.max_len]
        # elif input_ids.size(0) < self.max_len:
            # input_ids = torch.cat((input_ids, torch.tensor([pad_token_id] * (self.max_len - input_ids.size(0))))).int()
            # attention_mask = torch.cat((attention_mask, torch.tensor([0] * (self.max_len - attention_mask.size(0))))).int()
        if self.token_type_ids is not None:
            if self.augmentation:
                # Recode the token_type_ids in case we transformed the input_ids
                sep_indices = torch.where(input_ids == sep_token_id)[0]
                # Just 0 until the first token after the sep --> 1 and thats all (we dont look for more sep)
                zero_to_sep = torch.zeros(sep_indices[0] + 1)
                sep_to_end = torch.ones(input_ids.size(0) - sep_indices[0] - 1)
                # TO CHANGE
                token_type_ids = torch.cat((zero_to_sep, sep_to_end)).int()
                if token_type_ids.size(0) > self.max_len:
                    token_type_ids = token_type_ids[:self.max_len]
                # elif token_type_ids.size(0) < self.max_len:
                #     token_type_ids = torch.cat((token_type_ids, torch.tensor([0] * (self.max_len - token_type_ids.size(0)))).int())
            else:
                token_type_ids = torch.tensor(self.token_type_ids[idx]).int()
                if token_type_ids.size(0) > self.max_len:
                    token_type_ids = token_type_ids[:self.max_len]
                # elif token_type_ids.size(0) < self.max_len:
                #     token_type_ids = torch.cat((token_type_ids, torch.tensor([0] * (self.max_len - token_type_ids.size(0)))).int())
            
            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}, labels
        return {'input_ids': input_ids, 'attention_mask': attention_mask}, labels

    # Data augmentation : replace premise and hypothesis
    def transform(self, input_id, force_transform = False):
        sep_token_id = self.tokenizer.sep_token_id
        cls_token_id = self.tokenizer.cls_token_id

        sep_indices = torch.where(input_id == sep_token_id)[0]

        if "bert" in self.tokenizer.__class__.__name__.lower():
            premise = input_id[1:sep_indices[0]]
            hypothesis = input_id[sep_indices[0]+1:sep_indices[1]]
            input_id_transformed = torch.cat((torch.tensor([cls_token_id]), hypothesis, torch.tensor([sep_token_id]), premise, torch.tensor([sep_token_id])))
        if "roberta" in self.tokenizer.__class__.__name__.lower():
            # Redesign it : 
            # The sequences are like that : [0] + premise + 2*[sep_token_id] + hypothesis + [sep_token_id]
            premise = input_id[1:sep_indices[0]]
            hypothesis = input_id[sep_indices[1]+1:]
            input_id_transformed = torch.cat((torch.tensor([cls_token_id]), hypothesis, torch.tensor([sep_token_id, sep_token_id]), premise))
        else:
            premise = input_id[1:sep_indices[0]]
            hypothesis = input_id[sep_indices[0]+1:sep_indices[1]]
            input_id_transformed = torch.cat((torch.tensor([cls_token_id]), hypothesis, torch.tensor([sep_token_id]), premise, torch.tensor([sep_token_id])))



        # Equal prob of transforming or no :
        if torch.rand(1) > 0.5 or force_transform:
            return input_id_transformed
        else:
            return input_id

def custom_collate_fn(batch, pad_token_id):
    input_ids_list, attention_masks_list, token_type_ids_list, labels_list = [], [], [], []
    max_len = max([sample[0]["input_ids"].size(0) for sample in batch])

    for sample in batch:
        if sample[0]["input_ids"].size(0) < max_len:
            precomputed_input = torch.cat((sample[0]["input_ids"], torch.tensor([pad_token_id] * (max_len - sample[0]["input_ids"].size(0)))))
            input_ids_list.append(precomputed_input)

            precomputed_attention = torch.cat((sample[0]["attention_mask"], torch.tensor([0] * (max_len - sample[0]["attention_mask"].size(0)))))
            attention_masks_list.append(precomputed_attention)

            if "token_type_ids" in sample[0].keys():
                precomputed_token_type = torch.cat((sample[0]["token_type_ids"], torch.tensor([0] * (max_len - sample[0]["token_type_ids"].size(0)))))
                token_type_ids_list.append(precomputed_token_type)
        else:
            input_ids_list.append(sample[0]["input_ids"])
            attention_masks_list.append(sample[0]["attention_mask"])
            if "token_type_ids" in sample[0].keys():
                token_type_ids_list.append(sample[0]["token_type_ids"])

        labels_list.append(sample[1])

    input_ids = torch.stack(input_ids_list)
    attention_masks = torch.stack(attention_masks_list)

    if "token_type_ids" in batch[0][0].keys():
        token_type_ids = torch.stack(token_type_ids_list)
        return {'input_ids': input_ids, 'attention_mask': attention_masks, 'token_type_ids': token_type_ids}, torch.tensor(labels_list)

    return {'input_ids': input_ids, 'attention_mask': attention_masks}, torch.tensor(labels_list)
