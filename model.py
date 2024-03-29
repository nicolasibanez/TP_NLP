import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer

class MLPNLIModel(nn.Module):
    def __init__(self, model_name, model_cfg):
        super(MLPNLIModel, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        # nb_layers being the number of hidden sizes provided in the cgf : 
        # [768, 512, 256] -> 3 layers
        nb_layers = len(model_cfg["hidden_sizes"])
        if nb_layers == 0:
            self.classifier = nn.Linear(self.transformer.config.hidden_size, 3)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.transformer.config.hidden_size, model_cfg["hidden_sizes"][0]),
                nn.ReLU(),
                *[nn.Sequential(
                    nn.Linear(model_cfg["hidden_sizes"][i], model_cfg["hidden_sizes"][i+1]),
                    nn.ReLU()
                ) for i in range(nb_layers-1)],
                nn.Linear(model_cfg["hidden_sizes"][-1], 3)
            )
    def forward(self, inputs):
        outputs = self.transformer(**inputs)
        hidden_state = outputs[0]
        pooled_output = hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        return logits
