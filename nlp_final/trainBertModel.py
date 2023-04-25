from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.tokenization_utils_base import BatchEncoding

model: AutoModelForSequenceClassification = (
    AutoModelForSequenceClassification.from_pretrained(
        "NicholasSynovic/AutoTrain-LUC-COMP429-VEAA-Classification", use_auth_token=True
    )
)

tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
    "NicholasSynovic/AutoTrain-LUC-COMP429-VEAA-Classification", use_auth_token=True
)

inputs: BatchEncoding = tokenizer("I love AutoTrain", return_tensors="pt")

outputs: SequenceClassifierOutput = model(**inputs)
