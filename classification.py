from datasets import load_dataset
from transformers import LlamaTokenizer
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig
import torch

def load_and_prepare_data(tokenizer, batch_size=1):
    # 加载IMDB数据集
    dataset = load_dataset("imdb")
    
    # 预处理函数
    def preprocess_function(examples):
        # 对文本进行编码
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)
    
    # 应用预处理
    dataset = dataset.map(preprocess_function, batched=True)
    
    # 转换为PyTorch格式
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # 创建DataLoader
    train_loader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset['test'], batch_size=batch_size,shuffle=True)
    
    return train_loader, test_loader

# 初始化tokenizer
model_path = 'llama-3b'
tokenizer = LlamaTokenizer.from_pretrained(model_path)

# 设置EOS令牌作为填充令牌
tokenizer.pad_token = tokenizer.eos_token

train_loader, test_loader = load_and_prepare_data(tokenizer)
print('dataset created.')

class SentimentClassifier(torch.nn.Module):
    def __init__(self, model, num_labels):
        super(SentimentClassifier, self).__init__()
        self.llama = model.model
        self.classifier = torch.nn.Linear(model.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            # 获取Llama模型的输出.float
            outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)
            # 使用hidden_states属性
            hidden_states = outputs.last_hidden_state.float()  # 或 outputs[0] 如果outputs是元组
            # 创建一个mask，以忽略padding token对平均值的贡献
            #input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            
            #sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            #sum_mask = input_mask_expanded.sum(1)
            #sum_mask = torch.clamp(sum_mask, min=1e-9)
            #sequence_output = sum_embeddings / sum_mask
            sequence_output = hidden_states[:,attention_mask.sum(1).item()-1,:]
            #sequence_output = hidden_states[:, 0, :]  # 取序列的最后一个token的隐藏
        logits = self.classifier(sequence_output)
        return logits

# 加载预训练的LLaMa模型
pretrained_model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='auto',
)
for param in pretrained_model.parameters():
    param.requires_grad = False
    
model = SentimentClassifier(pretrained_model, num_labels=2)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.shape)

from transformers import AdamW
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print('model loaded.')

mode = 'test'

if mode == 'train':
    num_epochs = 10

    optimizer = AdamW(model.parameters(), lr=2e-4)  # 调整学习率
    accumulation_steps = 64  # 调整梯度累积步骤

    for epoch in range(num_epochs):
        model.eval()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            #print(batch['input_ids'][0].shape)

            #print(tokenizer.decode(batch['input_ids'][0]))
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            loss = torch.nn.CrossEntropyLoss()(outputs, batch['label'])
            loss = loss / accumulation_steps  # 标准化损失
            loss.backward()
            total_loss += loss.item()

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            if step == 1000:
                    break

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss}")



    torch.save(model.classifier.state_dict(), 'classifier_linear_layer.pth')
else:
    linear_layer_state = torch.load('classifier_linear_layer.pth')
    model.classifier.load_state_dict(linear_layer_state)
    
from sklearn.metrics import accuracy_score

model.eval()
predictions = []
labels = []

with torch.no_grad():
    step = 0
    for batch in tqdm(test_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        preds = torch.argmax(outputs, dim=1)
        predictions.extend(preds.cpu().numpy())
        labels.extend(batch['label'].cpu().numpy())
        step +=1
        if step == 1000:
            break
print(labels)
print(predictions)
accuracy = accuracy_score(labels, predictions)
print(f"Test Accuracy: {accuracy}")
