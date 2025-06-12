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
from torchcrf import CRF  # 引入 CRF

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
model = AutoModel.from_pretrained(model_path, local_files_only=True)


# 示例文本
text = "主唱太拼命了"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state

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
        self.crf = CRF(26, batch_first=True)  # 添加 CRF 层

    def forward(self, inputs, labels=None): # 修改 forward 函数
        if self.tuneing:
            out = self.pretrained(**inputs).last_hidden_state
        else:
            with torch.no_grad():
                out = pretrained(**inputs).last_hidden_state

        # out, _ = self.rnn(out)

        out = self.fc1(out)
        out = self.fc2(out)

        if labels is not None:
            # 如果提供了 labels，则计算 CRF loss
            mask = inputs['attention_mask'].bool()
            loss = -self.crf(out, labels, mask=mask, reduction='mean')
            return loss
        else:
            # 否则，使用 CRF 进行解码
            mask = inputs['attention_mask'].bool()
            prediction = self.crf.decode(out, mask=mask)
            return prediction

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

def get_correct_and_total_count(labels, outs, attention_mask):
    # 将预测结果和标签都转换为一维列表
    active_outs = [item for sublist in outs for item in sublist]  # outs 本身已经是 list of lists，直接展平
    active_labels = []
    
    # 遍历每个样本的标签和对应的 attention_mask
    for label_seq, mask_seq in zip(labels, attention_mask):
        for label, mask in zip(label_seq, mask_seq):
            if mask:  # 只保留 attention_mask 中为 True 的部分
                active_labels.append(label.item())

    # 确保 active_outs 和 active_labels 都是 list
    active_outs = [int(item) for item in active_outs]
    active_labels = [int(item) for item in active_labels]

    # 转换成 tensor
    active_outs = torch.tensor(active_outs).to(device)
    active_labels = torch.tensor(active_labels).to(device)

    correct = (active_outs == active_labels).sum().item()
    total = len(active_labels)

    # 计算除了0以外元素的正确率
    select = (active_labels != 0)
    active_outs = active_outs[select]
    active_labels = active_labels[select]
    correct_content = (active_outs == active_labels).sum().item()
    total_content = len(active_labels)

    return correct, total, correct_content, total_content

def train(epochs):
    lr = 2e-5 if model.tuneing else 5e-4
    # lr = 1e-5 if model.tuneing else 1e-4

    optimizer = AdamW(model.parameters(), lr=lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # # 计算每个类别的样本数量
    # label_counts = [0] * 26  # 有26个类别
    # for _, labels in dataset:
    #     for label in labels:
    #         if label != 25:
    #           label_counts[label] += 1

    # # 计算权重，做倒数
    # weights = [1.0 / count if count > 0 else 0 for count in label_counts]
    # weights = torch.tensor(weights).to(device)

    # criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=25) 
    
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        progress_bar = tqdm(loader, desc="Training", unit="batch")

        for step, (inputs, labels) in enumerate(progress_bar):

            # 将输入移动到设备
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            
            # 梯度下降
            loss = model(inputs, labels) # 直接用 model 计算 loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 100 == 0:
                with torch.no_grad():
                  outs = model(inputs) # 得到预测结果
                  counts = get_correct_and_total_count(labels, outs, inputs['attention_mask'])
                  accuracy = counts[0] / counts[1] if counts[1] > 0 else 0
                  accuracy_content = counts[2] / counts[3] if counts[3] > 0 else 0

                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "accuracy": f"{accuracy:.4f}",
                    # "accuracy_content": f"{accuracy_content:.4f}",
                    "accuracy_content": f"{accuracy_content}",
                })
        
        torch.save(model, '../model/NER_crf.model')

text_file = '../data/text_A_test.txt'
label_file = '../data/label_A_test.txt'

#测试
def predict():
    model_load = torch.load('../model/NER_crf.model', weights_only=False)
    model_load.eval()

    loader_test = torch.utils.data.DataLoader(dataset=TextLabelDataset(text_file, label_file),
                                              batch_size=2,
                                              collate_fn=collate_fn,
                                              shuffle=True,
                                              drop_last=True)

    for i, (inputs, labels) in enumerate(loader_test):
        break

    with torch.no_grad():
        outs = model_load(inputs) # outs 直接是解码后的结果，是一个 list of lists

    for i in range(2):
        # 移除 pad
        select = inputs['attention_mask'][i] == 1
        input_id = inputs['input_ids'][i, select]
        out = outs[i]  # 直接使用 outs[i] 获取当前样本的预测标签序列
        label = labels[i, select]

        # 输出原句子
        print(tokenizer.decode(input_id).replace(' ', ''))

        # 输出 tag
        # 标签
        s = ''
        for j in range(len(label)):
            if label[j] == 0:
                s += '·'
                continue
            s += tokenizer.decode(input_id[j])
            s += str(label[j].item())
        print("Label:", s)

        # 预测
        s = ''
        for j in range(len(out)):
            if out[j] == 0:
                s += '·'
                continue
            s += tokenizer.decode(input_id[j])
            s += str(out[j])
        print("Out:", s)

        print('==========================')

predict()