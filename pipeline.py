from model import MLPNLIModel
from transformers import AutoConfig, AutoTokenizer, AutoModel
import torch

class Pipeline():
    def __init__(
            self,
            tokenizer: AutoTokenizer,
            model: MLPNLIModel,
            device: torch.device = None
            ):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

        if self.device:
            self.model.to(self.device)

    def __call__(
            self,
            premise: str,
            hypothesis: str,
            ):
        return self.predict(premise, hypothesis)

    def predict_proba(
            self,
            premise: str,
            hypothesis: str,
            ):
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt")
        if self.device:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        return probs

    def predict(
            self, 
            premise: str, 
            hypothesis: str
            ):
        probs = self.predict_proba(premise, hypothesis)
        label = probs.argmax().item()
        if label == 0:
            return "entailment"
        elif label == 1:
            return "neutral"
        elif label == 2:
            return "contradiction"
        else:
            return "unknown"
        
        