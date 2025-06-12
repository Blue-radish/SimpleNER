from transformers import AutoTokenizer
import transformers
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModel
from transformers import AdamW
import random
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    transformers.set_seed(seed)

same_seeds(7890)

model_path = "../model/GujiRoBERTa_jian_fan"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

class TextLabelDataset(Dataset):
    def __init__(self, text_file, label_file, max_length=128):
        """
            text_file: 文本文件的路径.
            label_file: 标签文件的路径.
            tokenizer_name: 使用的 tokenizer 名称，默认为 'bert-base-chinese'.
            max_length: 最大序列长度，默认 128.
        """
        self.text_file = text_file
        self.label_file = label_file
        self.max_length = max_length
        self.texts, self.labels = self._load_data()
        
        self.dataset = self._filter_long_sentences() # 过滤掉过长的句子

    def _filter_long_sentences(self):
        """过滤掉过长的句子."""
        filtered_texts = []
        filtered_labels = []
        for text, label in zip(self.texts, self.labels):
            if len(text) <= self.max_length:
                filtered_texts.append(text)
                filtered_labels.append(label)

        return list(zip(filtered_texts,filtered_labels))

    def _load_data(self):
        """
        加载文本和标签数据。返回包含文本列表和标签列表的元组.
        """
        texts = []
        labels = []
        with open(self.text_file, 'r', encoding='utf-8') as f_text, \
                open(self.label_file, 'r', encoding='utf-8') as f_label:
            for text, label in zip(f_text, f_label):
                texts.append(list(text.strip()))
                labels.append(eval(label.strip()))  # 使用 eval 将字符串转换为 list
        return texts, labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens, labels = self.dataset[idx]
        return tokens, labels


text_file = '../data/text_A.txt'  # 文本文件的路径
label_file = '../data/label_A.txt'  # 标签文件的路径
dataset = TextLabelDataset(text_file, label_file)
tokens, labels = dataset[5]

len(dataset), tokens, labels

def collate_fn(data):
    tokens = [i[0] for i in data]
    labels = [i[1] for i in data]

    inputs = tokenizer.batch_encode_plus(tokens,
                                         truncation=True,
                                         padding=True,
                                         return_tensors='pt',
                                         is_split_into_words=True) 

    lens = inputs['input_ids'].shape[1]
    # print(lens)

    for i in range(len(labels)):
        labels[i] = [25] + labels[i]
        labels[i] += [25] * lens
        labels[i] = labels[i][:lens]

    return inputs.to(device), torch.LongTensor(labels).to(device)  # 将输入和标签都移动到设备上
    # return inputs, torch.LongTensor(labels)


loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=2,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)

#加载预训练模型
model_path = "../model/GujiRoBERTa_jian_fan"
pretrained = AutoModel.from_pretrained(model_path, local_files_only=True).to(device)

#定义下游模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tuneing = False
        self.pretrained = None

        # self.rnn = torch.nn.GRU(768, 768, batch_first=True)
        self.fc1 = torch.nn.Linear(768, 512)
        self.fc2 = torch.nn.Linear(512, 26)

    def forward(self, inputs):
        if self.tuneing:
            out = self.pretrained(**inputs).last_hidden_state
        else:
            with torch.no_grad():
                out = pretrained(**inputs).last_hidden_state

        # out, _ = self.rnn(out)

        out = self.fc1(out)
        out = F.softmax(self.fc2(out), dim=2)

        return out

    def fine_tuneing(self, tuneing):
        self.tuneing = tuneing
        if tuneing:
            for i in pretrained.parameters():
                i.requires_grad = True

            pretrained.train()
            self.pretrained = pretrained
        else:
            for i in pretrained.parameters():
                i.requires_grad_(False)

            pretrained.eval()
            self.pretrained = None


model = Model().to(device)

#对计算结果和label变形,并且移除pad
def reshape_and_remove_pad(outs, labels, attention_mask):
    #变形,便于计算loss
    outs = outs.reshape(-1, 26)
    labels = labels.reshape(-1)

    #忽略对pad的计算结果
    select = attention_mask.reshape(-1) == 1
    outs = outs[select]
    labels = labels[select]

    return outs, labels

#获取正确数量和总数
def get_correct_and_total_count(labels, outs):
    outs = outs.argmax(dim=1)
    correct = (outs == labels).sum().item()
    total = len(labels)

    #计算除了0以外元素的正确率
    select = (labels != 0)
    outs = outs[select]
    labels = labels[select]
    correct_content = (outs == labels).sum().item()
    total_content = len(labels)

    return correct, total, correct_content, total_content

#训练
def train(epochs):
    lr = 2e-5 if model.tuneing else 5e-4
    # lr = 1e-5 if model.tuneing else 1e-4

    optimizer = AdamW(model.parameters(), lr=lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 计算每个类别的样本数量
    label_counts = [0] * 26  # 有26个类别
    for _, labels in dataset:
        for label in labels:
            if label != 25:
              label_counts[label] += 1

    # 计算权重，做倒数
    weights = [1.0 / count if count > 0 else 0 for count in label_counts]
    weights = torch.tensor(weights).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=25) 
    
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        progress_bar = tqdm(loader, desc="Training", unit="batch")

        for step, (inputs, labels) in enumerate(progress_bar):

            # 将输入移动到设备
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            
            outs = model(inputs)

            # 对outs和label变形,并且移除pad
            outs, labels = reshape_and_remove_pad(outs, labels,
                                                inputs['attention_mask'])

            # 梯度下降
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 100 == 0:
                counts = get_correct_and_total_count(labels, outs)
                accuracy = counts[0] / counts[1] if counts[1] > 0 else 0
                accuracy_content = counts[2] / counts[3] if counts[3] > 0 else 0

                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "accuracy": f"{accuracy:.4f}",
                    # "accuracy_content": f"{accuracy_content:.4f}",
                    "accuracy_content": f"{accuracy_content}",
                })
        
        torch.save(model, '../model/NER_ZH.model')

model.fine_tuneing(False)
print(sum(p.numel() for p in model.parameters()) / 10000)
train(20)