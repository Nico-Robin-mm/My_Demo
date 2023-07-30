import torch
import torch.nn as nn
# import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
# import torch.utils.data.sampler as sampler
# import torchvision
# from torchvision import datasets, transforms
import re
import numpy as np
# import sys
import os
import random
import json
# import nltk
from nltk.translate.bleu_score import sentence_bleu
# from nltk.translate.bleu_score import SmoothingFunction


class LabelTransform(object):
    """pad to the same length"""
    def __init__(self, size, pad):
        self.size = size
        self.pad = pad

    def __call__(self, label):
        # arr = [1, 3, 2, 5, 4]
        # pad_arr = np.pad(arr, (3, 2), 'constant', 
        #              constant_values=(6, 4))
        # ans: [6 6 6 1 3 2 5 4 4 4]  # 前pad3个6 后pad2个4
        label = np.pad(label, (0, (self.size - label.shape[0])), mode='constant', constant_values=self.pad)
        return label


# training data num: 18000
# validation data num: 500
# test data num: 2636
class EN2CNDataset(data.Dataset):
    """
    special token:  <PAD>, <BOS>, <EOS>, <UNK>
    e.g. <BOS>, we, are, friends, <EOS> --> 1, 28, 29, 205, 2 # BOS-1, EOS-2
    """
    def __init__(self, root, max_output_len, set_name):
        self.root = root
      
        self.word2int_cn, self.int2word_cn = self.get_dictionary('cn')
        self.word2int_en, self.int2word_en = self.get_dictionary('en')
      
        # load data 
        self.data = []
        with open(os.path.join(self.root, f'{set_name}.txt'), "r", encoding='utf-8') as f:
            for line in f:
                self.data.append(line)
        print(f'{set_name} dataset size: {len(self.data)}')
      
        self.cn_vocab_size = len(self.word2int_cn)
        self.en_vocab_size = len(self.word2int_en)
        self.transform = LabelTransform(max_output_len, self.word2int_en['<PAD>'])
    
    def get_dictionary(self, language):
        # load dictionary
        with open(os.path.join(self.root, f'word2int_{language}.json'), "r", encoding='utf-8') as f:
            word2int = json.load(f)
        with open(os.path.join(self.root, f'int2word_{language}.json'), "r", encoding='utf-8') as f:
            int2word = json.load(f)
        return word2int, int2word
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, Index):
        # split the CN and ENG
        sentences = self.data[Index]
        sentences = re.split('[\t\n]', sentences)
        sentences = list(filter(None, sentences))
        # print (sentences)
        assert len(sentences) == 2
      
        # special token
        BOS = self.word2int_en['<BOS>']
        EOS = self.word2int_en['<EOS>']
        UNK = self.word2int_en['<UNK>']
      
        en, cn = [BOS], [BOS]

        sentence = re.split(' ', sentences[0])
        sentence = list(filter(None, sentence))
        #print (f'en: {sentence}')
        for word in sentence:
            en.append(self.word2int_en.get(word, UNK))
        en.append(EOS)
      
        # e.g. < BOS >, we, are, friends, < EOS > --> 1, 28, 29, 205, 2
        sentence = re.split(' ', sentences[1])
        sentence = list(filter(None, sentence))
        # print (f'cn: {sentence}')
        for word in sentence:
            cn.append(self.word2int_cn.get(word, UNK))
        cn.append(EOS)
      
        en, cn = np.asarray(en), np.asarray(cn)
      
        # pad to the same length
        en, cn = self.transform(en), self.transform(cn)
        en, cn = torch.LongTensor(en), torch.LongTensor(cn)
        return en, cn


class Encoder(nn.Module):
    """
    input = [batch size, sequence len, vocab size]
    outputs = [batch size, sequence len, hid dim * directions]
    hidden =  [num_layers * directions, batch size, hid dim]
    """
    def __init__(self, en_vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(en_vocab_size, emb_dim)
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input):
        # input = [batch size, sequence len, vocab size]
        embedding = self.embedding(input)
        outputs, hidden = self.rnn(self.dropout(embedding))
        # outputs = [batch size, sequence len, hid dim * directions]
        # hidden =  [num_layers * directions, batch size, hid dim]
        # outputs 是最上层RNN的输出           
        return outputs, hidden



class Decoder(nn.Module):  # decoder的hidden dim是encoder的2倍。 Cuz encoder is bidirectional.
    """
    isatt 决定是否使用 Attention Mechanism
    not isatt 输入为 embedding dim
    isatt 输入为 embedding dim + attention dim 
    self.input_dim = emb_dim + hid_dim * 2 if isatt else emb_dim
    output = [batch size, 1, hid dim] --> [batch size, hid dim]
    hidden = [num_layers, batch size, hid dim] 
    """
    def __init__(self, cn_vocab_size, emb_dim, hid_dim, n_layers, dropout, isatt):
        super().__init__()
        self.cn_vocab_size = cn_vocab_size
        self.hid_dim = hid_dim * 2
        self.n_layers = n_layers
        self.embedding = nn.Embedding(cn_vocab_size, config.emb_dim)
        self.isatt = isatt
        self.attention = Attention(hid_dim)
        # 如果使用 Attention Mechanism 输入维度会改变
        self.input_dim = emb_dim + hid_dim * 2 if isatt else emb_dim
        
        self.rnn = nn.GRU(self.input_dim, self.hid_dim, self.n_layers, dropout = dropout, batch_first=True)
        self.embedding2vocab1 = nn.Linear(self.hid_dim, self.hid_dim * 2)
        self.embedding2vocab2 = nn.Linear(self.hid_dim * 2, self.hid_dim * 4)
        self.embedding2vocab3 = nn.Linear(self.hid_dim * 4, self.cn_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input = [batch size, vocab size]
        # 注意： hidden 顺序应该为 [layers, batch_size, hid_dim*2]
        # Decoder 是单向，所以 directions=1
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        # embedded = [batch size, 1, emb dim]
        if self.isatt:
            context = self.attention(encoder_outputs, hidden)[0]

        # context 2d [batch_size, dec_hid_dim]
        context = context.unsqueeze(dim=1)
        new_input = torch.cat((embedded, context), dim=2)
        # output, hidden = self.rnn(embedded, hidden)
        output, hidden = self.rnn(new_input, hidden)
        # output = [batch size, 1, hid dim]
        # hidden = [num_layers, batch size, hid dim]
      
        # 将rnn的输出转为每个词出现的概率
        output = self.embedding2vocab1(output.squeeze(1))
        output = self.embedding2vocab2(output)
        prediction = self.embedding2vocab3(output)
        # prediction = [batch size, vocab size]
        return prediction, hidden


class Attention(nn.Module):
    """注意力机制，此处采取一个linear network方式"""
    def __init__(self, hid_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hid_dim * 4, hid_dim*2)
        self.v = nn.Linear(hid_dim*2, 1, bias=False)
      
    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs = [batch size, sequence len, hid dim * directions]
        # decoder_hidden = [num_layers, batch size, hid dim]
        # 一般取 Encoder 最后一层的 hidden state 做 attention
        seq_len = encoder_outputs.shape[1]
        
        decoder_hidden = decoder_hidden[-1, :, :]  # [batch_size, decoder_hid_dim]
        decoder_hidden = decoder_hidden.unsqueeze(dim=1)  # [batch_size, 1, decoder_hid_dim]
        decoder_hidden = decoder_hidden.repeat(1, seq_len, 1) # [batch_size, seq_len, decode_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((encoder_outputs, decoder_hidden), dim=2))) # [batch_size, seq_len, decoder_hid_dim*2)] -> [batch_size, seq_len, decoder_hid_dim]
        attention = self.v(energy).squeeze(2) # [batch_size, seq_len, dec_hid_dim->1] -> [batch_size, seq_len]
        attention = F.softmax(attention, dim=1) # [batch_size, seq_len]
        context = torch.bmm(encoder_outputs.permute(0, 2, 1), attention.unsqueeze(2)).squeeze(2)
        # bmm is batch matrix multiplication
        # [batch_size, dec_hid_dim, seq_len] bmm [batch_size, seq_len, 1]
        # [batch_size, dec_hid_dim, 1] -> [batch_size, dec_hid_dim]
        return context, attention


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.n_layers == decoder.n_layers, \
                "Encoder and decoder must have equal number of layers!"
              
    def forward(self, input, target, teacher_forcing_ratio):
        # input  = [batch size, input len, vocab size]
        # target = [batch size, target len, vocab size]
        # teacher_forcing_ratio 有多少概率使用正确答案来做training
        batch_size = target.shape[0]
        target_len = target.shape[1]
        vocab_size = self.decoder.cn_vocab_size
      
        # 准备一个 存储空间 存储 输出
        outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(input)
        # Encoder 最后的隐藏层 hidden state 来 初始化 Decoder
        # encoder_outputs 主要是使用在 Attention
        # Cuz Encoder bidirectional, so tie 2-directions together
        # hidden =  [num_layers * directions, batch size  , hid dim]  --> [num_layers, directions, batch size  , hid dim]
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2) # hidden[:, -2, :, :] 的维度为3维
        # now, hidden becomes [layers, batch_size, hid_dim*2]
        
        # 取的 <BOS> token
        input = target[:, 0]
        preds = []
        for t in range(1, target_len):
            # 下面一行中 encoder_outputs 处 应为 attention vector
            output, hidden = self.decoder(input, hidden, encoder_outputs)  # 此处的hidden [num_layers, batch_size, hid_dim*2] 
            outputs[:, t] = output  # output 2维 [batch_size, cn_vocab_size]
            # 一定的概率选取正确答案
            teacher_force = random.random() <= teacher_forcing_ratio
            # 选取几率最大的单词
            top1 = output.argmax(dim=1)  # output 2维 [batch_size, cn_vocab_size], top1 1维 [batch_size]
            # 如果是 teacher force 则用正解训练，反之用自己预测的单词
            input = target[:, t] if teacher_force and t < target_len else top1
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)  # preds 2维 [batch_size, t or target_len]
        return outputs, preds  # outputs 3维 [batch_size, target_len, vocab_size]
    
    
    def inference(self, input, target, beam_size=3):
        """Beam Search"""
        # 此函式的 batch size = 1  
        # input  = [batch size, input len, vocab size]
        # target = [batch size, target len, vocab size]
        batch_size = input.shape[0]
        input_len = input.shape[1]        # 取得最大字數
        vocab_size = self.decoder.cn_vocab_size
      
        # 预备一个存储空间
        outputs = torch.zeros(batch_size, input_len, vocab_size).to(self.device)
        
        encoder_outputs, hidden = self.encoder(input)
        # Encoder 最后隐藏层(hidden state) 初始化 Decoder
        # encoder_outputs 主要是使用在 Attention
        # Encoder bi-directional, so tie together
        # hidden =  [num_layers * directions, batch size  , hid dim]  --> [num_layers, directions, batch size  , hid dim]
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
        # 取的 <BOS> token
        # input = target[:, 0]
        
        # Initialize beam
        beam = [(
            torch.tensor([1], dtype=torch.long, device=device),  # 1-<BOS>
            0,  # score-0
            hidden,
            []
                  )]  
        
        for t in range(1, input_len):
            candidates = []
            for seq, score, hidden, output_list in beam:
                temp = output_list[:]
                last_tok = seq[-1]
                # If the last token is the end of sequence token, add the sequence to candidates
                # if last_tok == 2:
                #     # candidates.append((seq, score))  # ?
                #     continue
                
                output, hidden = self.decoder(last_tok.unsqueeze(0), hidden, encoder_outputs)
                # output 2维 [batch_size, cn_vocab_size]
                # 此处的hidden [num_layers, batch_size, hid_dim*2]
                    
                # output_list.append(output)
                temp.append(output)
                # Get the top k predicted tokens and their corresponding scores
                scores, indices = torch.topk(F.log_softmax(output, dim=1), k=beam_size)
                # Iterate through each predicted token and update the beam
                for _score, index in zip(scores.squeeze(), indices.squeeze()):
                    new_seq = torch.cat([seq, index.unsqueeze(0)])
                    candidates.append((new_seq, _score.item() + score, hidden, temp))
                
            # Sort candidates by score
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
            
            # Update beam with top k candidates
            beam = candidates[:beam_size]
            
        preds = beam[0][0].unsqueeze(dim=0)

        # print(beam)
        for i in range(1, input_len):
            outputs[:, i] = beam[0][3][i-1]
            
        return outputs, preds


def build_model(config, en_vocab_size, cn_vocab_size):
    encoder = Encoder(en_vocab_size, config.emb_dim, config.hid_dim, config.n_layers, config.dropout)
    decoder = Decoder(cn_vocab_size, config.emb_dim, config.hid_dim, config.n_layers, config.dropout, config.attention)
    model = Seq2Seq(encoder, decoder, device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    print(optimizer)
    if config.load_model:
        model = load_model(model, config.load_model_path)
    model = model.to(device)
    
    return model, optimizer


def save_model(model, optimizer, store_model_path, step):
    torch.save(model.state_dict(), f'{store_model_path}/model_{step}.ckpt')
    return


def load_model(model, load_model_path):
    print(f'Load model from {load_model_path}')
    
    # TODO
    # here need to select the model to load: step! 
    model.load_state_dict(torch.load(f'{load_model_path}/model_{step}.ckpt'))
    
    return model


def tokens2sentence(outputs, int2word):
    sentences = []
    for tokens in outputs:  # here outputs 2维 [batch_size, t or target_len]]
                            # tokens 1维 [target_len]
        sentence = []
        for token in tokens:  # token is a number
            word = int2word[str(int(token))]
            if word == '<EOS>':
                break
            sentence.append(word)
        sentences.append(sentence)    
    return sentences


def computebleu(sentences, targets):  
    """BLEU score"""
    score = 0 
    assert (len(sentences) == len(targets))
    
    def cut_token(sentence):
        tmp = []
        for token in sentence:
            if token == '<UNK>' or token.isdigit() or len(bytes(token[0], encoding='utf-8')) == 1:
                tmp.append(token)
            else:
                tmp += [word for word in token]
        return tmp 
    
    for sentence, target in zip(sentences, targets):
        sentence = cut_token(sentence)
        target = cut_token(target)
        score += sentence_bleu([target], sentence, weights=(1, 0, 0, 0))                                                                                          
    
    return score


def infinite_iter(data_loader):
    it = iter(data_loader)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(data_loader)


# linear decay
def schedule_sampling(step, num_steps):
    return 1 - step / num_steps


def train(model, optimizer, train_iter, loss_function, total_steps, summary_steps, train_dataset):
    model.train()
    model.zero_grad()
    losses = []
    loss_sum = 0.0
    for step in range(summary_steps):
        sources, targets = next(train_iter)
        # sources 2d [batch_size, seq_len]
        # targets 2d [batch_size, seq_len]
        sources, targets = sources.to(device), targets.to(device)
        outputs, preds = model(sources, targets, schedule_sampling(step, summary_steps))
        # outputs 3维 [batch_size, target_len, vocab_size]
        # preds 2维 [batch_size, target_len-1]
        
        # targets 的第一个 token 是 <BOS> 所以忽略
        outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
        # outputs 2d [batch_size * (target_len - 1), vocab_size]
        targets = targets[:, 1:].reshape(-1)
        # targets 1d [batch_size * (target_len - 1)]
        
        loss = loss_function(outputs, targets)
        # loss_function = nn.CrossEntropyLoss(ignore_index=0)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
      
        loss_sum += loss.item()
        if (step + 1) % 5 == 0:
            loss_sum = loss_sum / 5
            print ("\r", "train [{}] loss: {:.3f}, Perplexity: {:.3f}".format(total_steps + step + 1, loss_sum, np.exp(loss_sum)), end=" ")
            losses.append(loss_sum)
            loss_sum = 0.0
    
    return model, optimizer, losses
    
    
def test(model, dataloader, loss_function):
    model.eval() 
    with torch.no_grad():
        loss_sum, bleu_score= 0.0, 0.0
        n = 0
        result = []
        for sources, targets in dataloader:
            # sources 2d [batch_size, seq_len]
            # targets 2d [batch_size, seq_len]
            sources, targets = sources.to(device), targets.to(device)
            batch_size = sources.size(0)
            outputs, preds = model.inference(sources, targets)
            # outputs 3维 [batch_size, input_max_len, vocab_size]
            # targets 的第一个 token 是 <BOS> 所以忽略
            outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
            targets = targets[:, 1:].reshape(-1)
            # outputs 2d [batch_size * (target_len - 1), vocab_size]
            # targets 1d [batch_size * (target_len - 1)]
            loss = loss_function(outputs, targets)
            loss_sum += loss.item()
          
            # 将预测结果转换成文字
            targets = targets.view(sources.size(0), -1)
            preds = tokens2sentence(preds, dataloader.dataset.int2word_cn)
            sources = tokens2sentence(sources, dataloader.dataset.int2word_en)
            targets = tokens2sentence(targets, dataloader.dataset.int2word_cn)
            for source, pred, target in zip(sources, preds, targets):
                result.append((source, pred, target))
            # get Bleu Score
            bleu_score += computebleu(preds, targets)
          
            n += batch_size
    
    return loss_sum / len(dataloader), bleu_score / n, result


def train_process(config):
    
    train_dataset = EN2CNDataset(config.data_path, config.max_output_len, 'training')
    train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    train_iter = infinite_iter(train_loader)

    val_dataset = EN2CNDataset(config.data_path, config.max_output_len, 'validation')
    val_loader = data.DataLoader(val_dataset, batch_size=1)

    model, optimizer = build_model(config, train_dataset.en_vocab_size, train_dataset.cn_vocab_size)
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    
    train_losses, val_losses, bleu_scores = [], [], []
    total_steps = 0
    while (total_steps < config.num_steps):

        model, optimizer, loss = train(model, optimizer, train_iter, loss_function, total_steps, config.summary_steps, train_dataset)
        train_losses += loss

        val_loss, bleu_score, result = test(model, val_loader, loss_function)
        val_losses.append(val_loss)
        bleu_scores.append(bleu_score)
      
        total_steps += config.summary_steps
        print ("\r", "val [{}] loss: {:.3f}, Perplexity: {:.3f}, bleu score: {:.3f}".format(total_steps, val_loss, np.exp(val_loss), bleu_score))
        
        # save model and result
        if total_steps % config.store_steps == 0 or total_steps >= config.num_steps:
            save_model(model, optimizer, config.store_model_path, total_steps)
            with open(f'{config.store_model_path}/output_{total_steps}.txt', 'w') as f:
                for line in result:
                    print (line, file=f)
      
    return train_losses, val_losses, bleu_scores


def test_process(config):

    test_dataset = EN2CNDataset(config.data_path, config.max_output_len, 'testing')
    test_loader = data.DataLoader(test_dataset, batch_size=1)

    model, optimizer = build_model(config, test_dataset.en_vocab_size, test_dataset.cn_vocab_size)
    print ("Finish build model")
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    model.eval()
    with torch.no_grad():

        test_loss, bleu_score, result = test(model, test_loader, loss_function)

        with open('./test_output.txt', 'w') as f:
            for line in result:
                print (line, file=f)
      
    return test_loss, bleu_score


class configurations(object):
    def __init__(self):
        self.batch_size = 60
        self.emb_dim = 256
        self.hid_dim = 512
        self.n_layers = 3
        self.dropout = 0.5
        self.learning_rate = 0.00005
        self.max_output_len = 50              # 最后输出句子的最大length
        self.num_steps = 12000                # 总训练次数
        self.store_steps = 300                # 每多少步存储一次模型
        self.summary_steps = 300              # 每多少步检验是否overfitting
        self.load_model = False               
        self.store_model_path = "./ckpt"      
        self.load_model_path = None            
        self.data_path = "./cmn-eng"          
        self.attention = True                # 是否使用 Attention Mechanism


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = configurations()
    print ('config:\n', vars(config))
    train_losses, val_losses, bleu_scores = train_process(config)

    test_loss, bleu_score = test_process(config)
    print (f'test loss: {test_loss}, bleu_score: {bleu_score}')
