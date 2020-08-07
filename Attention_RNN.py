import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.device = config.device
        self.n_layers = config.num_rnn
        self.hidden_size = config.hidden_size
        self.embedding = nn.Embedding(config.n_vocab, config.embed_dim)
        self.lstm = nn.LSTM(config.embed_dim, self.hidden_size, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, x, mask):
        h0 = torch.zeros(self.n_layers * 2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.n_layers * 2, x.size(0), self.hidden_size).to(self.device)

        embed = self.embedding(x)
        out, (hn, cn) = self.lstm(embed, (h0, c0))

        real_content = []

        for i, o in enumerate(out):
            rea_length = mask[i].data.tolist().count(0)
            real_content.append(o[rea_length-1])

        x1 = torch.cat(real_content)
        context = x1.view(x.size(0), -1).unsqueeze(dim=1)

        return out, context


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.device = config.device
        self.n_layers = config.num_rnn
        self.hidden_size = config.hidden_size * 2
        self.embedding = nn.Embedding(config.slot_size, config.decoder_embed_size)
        self.lstm = nn.LSTM(config.decoder_embed_size+self.hidden_size*2, self.hidden_size, num_layers=1, batch_first=True)
        self.attention = nn.Linear(self.hidden_size, self.hidden_size)
        self.slot_out = nn.Linear(self.hidden_size*2, config.slot_size)
        self.intent_out = nn.Linear(self.hidden_size*2, config.intent_size)

    def Attention(self, hidden, encoder_outputs, masking):
        """
        :param hidden: 1, B, D
        :param encoder_outputs: B, T, D
        :param masking: B, T
        :return: B, 1, D
        """
        hidden = hidden.squeeze(0).unsqueeze(2)
        bsz = encoder_outputs.size(0)
        max_len = encoder_outputs.size(1)

        x1 = encoder_outputs.contiguous()
        energies = self.attention(x1.view(bsz * max_len, -1))  # B*T, D -->
        energies = energies.view(bsz, max_len, -1)

        attn_energies = energies.bmm(hidden).transpose(1, 2)
        attn_energies = attn_energies.squeeze(1).masked_fill(masking, -1e12)

        alpha = F.softmax(attn_energies, dim=1)
        alpha = alpha.unsqueeze(1)

        context = alpha.bmm(encoder_outputs)

        return context

    def forward(self, x, context, encoder_outputs, encoder_mask):
        embed = self.embedding(x)  # B,1 -> B,1,D
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(self.device)
        decode = []
        aligns = encoder_outputs.transpose(0, 1)  # B,T,D -> T,B,D
        length = encoder_outputs.size(1)

        for i in range(length):
            aligned = aligns[i].unsqueeze(1)  # B,D -> B,1,D
            # print(embed.size(), context.size(), aligned.size())
            lstm_input = torch.cat((embed, context, aligned),dim=2)
            _, (hn, cn) = self.lstm(lstm_input, (h0, c0))

            # intent Detection
            if i == 0:
                intent_hidden = hn.clone()  # 1,B,D
                intent_context = self.Attention(intent_hidden, encoder_outputs, encoder_mask)  # B, 1, D
                concated = torch.cat((intent_hidden, intent_context.transpose(0, 1)), 2)
                intent_score = self.intent_out(concated.squeeze(0))

            concated = torch.cat((hn, context.transpose(0, 1)), 2)
            score = self.slot_out(concated.squeeze(0))
            softmaxed = F.log_softmax(score, dim=1)
            decode.append(softmaxed)

            _, x = torch.max(softmaxed, 1)

            embed = self.embedding(x.unsqueeze(1))

            context = self.Attention(hn, encoder_outputs, encoder_mask)

        slot_score = torch.stack(decode, dim=1)

        return slot_score.view(-1, slot_score.size(2)), intent_score
