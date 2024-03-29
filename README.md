# Natural Language Inference Project
This repository contains the code and results for our Natural Language Inference (NLI) project.

## Project Members
- Nicolas Ibanez (nicolas.ibanez@student-cs.fr)
- Paul Cayet (paul.cayet@student-cs.fr)

## File Description
The main file of interest is ``NLI_TP.ipynb``, which contains the explanation for data, training and evaluating. The files ``main.py``, ``utils.py``, ``checkpoint.py``, ``model.py`` and ``dataset.py`` were used for the train.

For the LLM-based approach, the inference code can be found in the ``llm_code`` folder. Detailed results can be found in the ``llm_results`` folder. 

## Results
The results of the different models are summarized in the following table :

| Checkpoint                                     | Model                        | Test Accuracy | Test Accuracy Both | Test Accuracy Swap |
|-----------------------------------------------|------------------------------|---------------|--------------------|--------------------|
| ckpt/deberta-v3-large_20240329_115256.pth    | microsoft/deberta-v3-large   | **0.9275244299674267** | **0.7808428338762216** | **0.594564332247557** |
| ckpt/roberta-large_20240327_174024.pth       | roberta-large                | 0.9228420195439739 | 0.7039902280130294 | 0.5855048859934854 |
| ckpt/bert-large-uncased_20240327_173829.pth  | bert-large-uncased           | 0.913578990228013 | 0.7034812703583062 | 0.5700325732899023 |
| ckpt/roberta-base_20240328_200823.pth        | roberta-base                 | 0.9098127035830619 | 0.729336319218241 | 0.5594462540716613 |
| ckpt/bert-base-uncased_20240328_201206.pth   | bert-base-uncased            | 0.9072679153094463 | 0.7406351791530945 | 0.5565960912052117 |
| ckpt/TinyBERT_General_4L_312D_20240328_145724.pth | huawei-noah/TinyBERT_General_4L_312D | 0.877442996742671 | 0.7351384364820847 | 0.5162866449511401 |

**Note:**
- Test Accuracy: When tokenizing the "hypothesis" + "premise" sentence in the same order between training and inference.
- Test Accuracy Swap: When tokenizing the "hypothesis" + "premise" sentence in opposite order between training and inference.
- Test Accuracy Both: Using a probability-level ensemble of the normal and swap inference

For more details on fine-tuning and results for the LLM-based approach, please read the notebook ``NLI_TP.ipynb``.
