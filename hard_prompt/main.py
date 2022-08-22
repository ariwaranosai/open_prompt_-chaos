import torch, json
from openprompt import PromptDataLoader
from openprompt import PromptForClassification
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from torch.utils.data import Dataset

plm, tokenizer, model_config, WrapperClass = load_plm("bert", "hfl/chinese-roberta-wwm-ext")

prompt_template = ManualTemplate(
    text='{"placeholder": "text_a"}, 我 {"mask"}',
    tokenizer=tokenizer
)

classes = ["positive", "negative"]

prompt_verbalizer = ManualVerbalizer(
    classes=classes,
    label_words={
        "positive": ["喜欢"],
        "negative": ["讨厌"]
    },
    tokenizer=tokenizer
)

prompt_model = PromptForClassification(
    template=prompt_template,
    plm=plm,
    verbalizer=prompt_verbalizer
)

use_cuda = False
if use_cuda:
    prompt_model.cuda()


class InputExampleDataset(Dataset):
    def __init__(self, path):
        """
        label,text_a
        """
        self.text = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                label = line[0]
                text_a = line[2:]
                self.text.append(InputExample(text_a=text_a, label=int(label)))

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        return self.text[index]

    def __iter__(self):
        return iter(self.text)


dataset = {}
dataset_name = "waimai_10k_small"
for name in ["train", "validation"]:
    dataset[name] = InputExampleDataset(path=f"../dataset/{dataset_name}_{name}.csv")

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=prompt_template, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=WrapperClass, decoder_max_length=3,
                                    batch_size=4, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                    truncate_method="head")

from transformers import AdamW

loss_func = torch.nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.weight']
# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters = [
    {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)

MAX_STEPS = 10000
EVAL_STEPS = 2
LOG_STEPS = 1

for epoch in range(MAX_STEPS):
    tot_loss = 0
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        tot_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        if step % LOG_STEPS == 0:
            print("Epoch {}, average loss: {}".format(epoch, tot_loss / (step + 1)), flush=True)

        if (step // LOG_STEPS != 0) and (step % EVAL_STEPS == 0):
            validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=prompt_template,
                                                     tokenizer=tokenizer,
                                                     tokenizer_wrapper_class=WrapperClass, decoder_max_length=3,
                                                     batch_size=4, shuffle=False, teacher_forcing=False,
                                                     predict_eos_token=False,
                                                     truncate_method="head")
            all_preds = []
            all_labels = []
            for step, inputs in enumerate(validation_dataloader):
                if use_cuda:
                    inputs = inputs.cuda()
                logits = prompt_model(inputs)
                labels = inputs['label']
                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

            acc = sum([int(i == j) for i, j in zip(all_preds, all_labels)]) / len(all_preds)
            print("============ eval acc =========")
            print(acc)
            print("============ eval acc =========")
