from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
import torch

classes = [  # There are two classes in Sentiment Analysis, one for negative and one for positive
    "negative",
    "positive"
]
dataset = [  # For simplicity, there's only two examples
    # text_a is the input text of the data, some other datasets may have multiple input sentences in one example.
    InputExample(
        guid=0,
        text_a="Albert Einstein was one of the greatest intellects of his time.",
    ),
    InputExample(
        guid=1,
        text_a="The film was badly made.",
    ),
]

plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
promptTemplate = ManualTemplate(
    text='{"placeholder":"text_a"} It was {"mask"}',
    tokenizer=tokenizer,
)

promptVerbalizer = ManualVerbalizer(
    classes=classes,
    label_words={
        "negative": ["bad"],
        "positive": ["good", "wonderful", "great"],
    },
    tokenizer=tokenizer,
)
promptModel = PromptForClassification(
    template=promptTemplate,
    plm=plm,
    verbalizer=promptVerbalizer,
)
data_loader = PromptDataLoader(
    dataset=dataset,
    tokenizer=tokenizer,
    template=promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
)

# making zero-shot inference using pretrained MLM with prompt
promptModel.eval()
with torch.no_grad():
    for batch in data_loader:
        logits = promptModel(batch)
        preds = torch.argmax(logits, dim=-1)
        print(classes[preds])

# predictions would be 1, 0 for classes 'positive', 'negative'
