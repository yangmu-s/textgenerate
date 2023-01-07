import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F 
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
device=torch.device("cuda:0"if torch.cuda.is_available()else "cpu")
text="st lesson that we always ignored is that to ask ourself who we are ,and what we real need .I do remember that most childs had the desire to dig out the secret about universe and life while i was\
also a child .But a few years later ,most of them even dropped the simplist faculty that thinking with themselves .They do learnt more words ,poems,and maybe some MOT ,but they'd be trained only\
chasing for money and never thought about that what the money real mean .\
Once someone betrayed in the circumstance made of money ,they will never get the oppotunities to gain happy nor money .Must know that money is just the representation of wealth ,work hard,\
work for dream,work for the case that bringing more freedom ,love and intelligence for more people ,God usually brings the reward .Through the whole line of human's history ,we learned that focusing\
on one proper term makes more possibility of success .The worstest life is that never focusing on one target .Patience maybe is the most important talent in this world ,and always build the miracle\
every second .\
Somethings we must say that ,in most of our life ,there's few delighted moment .Look through the past ,i have to say there's so much time that i regret .But it's surely the past ,i can't do anything for\
it .Sometimes i made some choice ,but it more depend on the issues that i just suffered .That's to say that actually nobody can real handle their whole life ,different fate builds different life ,but that's\
not mean we can do nothing .God preparing all the beauties before the way ,but it still need us to search it .\
That's a really unpleasant term that someone always said that he can handle his whole fate when he was yound ,but instead say that his whole life is just aranged by God when he's old .There's no absulute things\
that are destined to be ,every second ,as we still alive ,that surely have the chance to make something different ."
words=[s for s in text.lower().split(' ')]
word2index={}
index2word={}
n_words=0
for i in words: 
	if i not in word2index:
		word2index[i]=n_words
		index2word[n_words]=i
		n_words+=1
length=5
indexes=[word2index[w] for w in words]
vocab_size=len(word2index)
class LstmNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.embedding=nn.Embedding(vocab_size,128)
		self.lstm1=nn.LSTM(input_size=128,hidden_size=128,num_layers=1)
		self.dropout1=nn.Dropout(0.2)
		self.fc=nn.Linear(128,vocab_size)

	def forward(self,x):
		x=self.embedding(x)
		x,_=self.lstm1(x)
		x=self.dropout1(x)
		x=self.fc(x)
		return x
model=LstmNet()
model.to(device)
print(model)
class Data(Dataset):
	def __init__(self,index):
		super().__init__()
		self.index=index 
	def __getitem__(self,item):
		return(torch.tensor(self.index[item:item+4]),
			torch.tensor(self.index[item+1:item+5]))
	def __len__(self):
		return len(self.index)-4

data=Data(index=indexes)
loader=DataLoader(data,batch_size=16,shuffle=True)
optimizer=optim.Adam(model.parameters(),lr=1e-3)
epochs=10 
loss_fn=nn.CrossEntropyLoss()
for epoch in range(epochs):
	print("epoch:",epoch)
	for x_train,y_train in loader:
		print(x_train.shape)
		x_train=x_train.to(device)
		y_train=y_train.to(device)
		pred=model(x_train)
		loss=loss_fn(pred.transpose(1,2),y_train)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	print(epoch,loss.data)
model.eval()
text='most of life'
next_words=20
words=text.split(' ')
for i in range(next_words):
	input=torch.tensor([[word2index[w]for w in words[i:]]]).to(device)
	out=model(input)
	out=out[0][-1].cpu()
	p=torch.nn.functional.softmax(out,dim=0).detach().numpy()
	predict=np.random.choice(len(out),p=p)
	words.append(index2word[predict])

print(words)
wordgenerate=' '.join(w for w in words)
print(wordgenerate)
