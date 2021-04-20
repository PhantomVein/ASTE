from driver.Layer import *
import torch.nn.functional as F
from driver.BertTokenHelper import BertTokenHelper
from driver.BertModel import BertExtractor


def drop_sequence_sharedmask(inputs, dropout, batch_first=True):
    if batch_first:
        inputs = inputs.transpose(0, 1)
    seq_length, batch_size, hidden_size = inputs.size()
    drop_masks = inputs.data.new(batch_size, hidden_size).fill_(1 - dropout)
    drop_masks = Variable(torch.bernoulli(drop_masks), requires_grad=False)
    drop_masks = drop_masks / (1 - dropout)
    drop_masks = torch.unsqueeze(drop_masks, dim=2).expand(-1, -1, seq_length).permute(2, 0, 1)
    inputs = inputs * drop_masks

    return inputs.transpose(1, 0)


class ParserModel(nn.Module):
    def __init__(self, vocab, config):
        super(ParserModel, self).__init__()
        self.config = config
        self.bert = BertExtractor(config)

        self.tag_embed = nn.Embedding(vocab.tag_size, config.tag_dims, padding_idx=0)
        tag_init = np.random.randn(vocab.tag_size, config.tag_dims).astype(np.float32)
        self.tag_embed.weight.data.copy_(torch.from_numpy(tag_init))

        self.mlp_arc_dep = NonLinear(
            input_size = config.word_dims + config.tag_dims,
            hidden_size = config.mlp_arc_size+config.mlp_rel_size,
            activation = nn.LeakyReLU(0.1))
        self.mlp_arc_head = NonLinear(
            input_size = config.word_dims + config.tag_dims,
            hidden_size = config.mlp_arc_size+config.mlp_rel_size,
            activation = nn.LeakyReLU(0.1))

        self.total_num = int((config.mlp_arc_size+config.mlp_rel_size) / 100)
        self.arc_num = int(config.mlp_arc_size / 100)
        self.rel_num = int(config.mlp_rel_size / 100)

        self.arc_biaffine = Biaffine(config.mlp_arc_size, config.mlp_arc_size, \
                                     1, bias=(True, False))
        self.rel_biaffine = Biaffine(config.mlp_rel_size, config.mlp_rel_size, \
                                     vocab.rel_size, bias=(True, True))

    def initial_pretrain(self, pretrained_embedding):
        self.extword_embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        self.extword_embed.weight.requires_grad = False


    def forward(self, bert_indice_input, bert_segments_ids, bert_piece_ids, bert_mask, tags, masks):
        # bert_indice_input = (batch size, sequence length, dimension of embedding)
        last_hidden = self.bert(bert_indice_input, bert_segments_ids, bert_piece_ids, bert_mask)
        x_embed = F.dropout(last_hidden, p=self.config.dropout_emb, training=self.training)
        x_tag_embed = self.tag_embed(tags)
        x_tag_embed = F.dropout(x_tag_embed, p=self.config.dropout_emb, training=self.training)

        x_lexical = torch.cat((x_embed, x_tag_embed), dim=2)

        x_all_dep = self.mlp_arc_dep(x_lexical)
        x_all_head = self.mlp_arc_head(x_lexical)

        if self.training:
            x_all_dep = drop_sequence_sharedmask(x_all_dep, self.config.dropout_mlp)
            x_all_head = drop_sequence_sharedmask(x_all_head, self.config.dropout_mlp)

        x_all_dep_splits = torch.split(x_all_dep, 100, dim=2)
        x_all_head_splits = torch.split(x_all_head, 100, dim=2)

        x_arc_dep = torch.cat(x_all_dep_splits[:self.arc_num], dim=2)
        x_arc_head = torch.cat(x_all_head_splits[:self.arc_num], dim=2)

        arc_logit = self.arc_biaffine(x_arc_dep, x_arc_head)
        arc_logit = torch.squeeze(arc_logit, dim=3)

        x_rel_dep = torch.cat(x_all_dep_splits[self.arc_num:], dim=2)
        x_rel_head = torch.cat(x_all_head_splits[self.arc_num:], dim=2)

        rel_logit_cond = self.rel_biaffine(x_rel_dep, x_rel_head)
        return arc_logit, rel_logit_cond