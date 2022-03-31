import torch.nn as nn
import torch
import copy
import math


class Transformer(nn.Module):
    """
    :param n_patches: 速度的维度
    :param hidden_size: 通道数
    """

    def __init__(self, n_patches, hidden_size, num_layers=4, dropout_rate=0.1, num_heads=1,
                 attention_dropout_rate=0.0):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(n_patches=n_patches, hidden_size=hidden_size, dropout_rate=dropout_rate)
        self.encoder = Encoder(hidden_size, num_layers, dropout_rate, num_heads, attention_dropout_rate)

    def forward(self, input_ids):
        embedding_output, x_size = self.embeddings(input_ids)
        encoded = self.encoder(embedding_output)  # (B, n_patch, hidden)
        encoded = encoded.reshape(x_size)
        encoded = encoded.permute(0, 4, 1, 2, 3)
        encoded = encoded.sum(dim=4)/x_size[3]
        return encoded


class Embeddings(nn.Module):
    """Construct the embeddings from position embeddings.
    """

    def __init__(self, n_patches, hidden_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.n_patches = n_patches
        self.hidden_size = hidden_size
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        x_size = x.shape
        x = x.flatten(0, 2)
        # x.size = (B*256*256, n_patches, hidden)
        embeddings = x + self.position_embeddings
        return embeddings, x_size


class Encoder(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout_rate, num_heads, attention_dropout_rate):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        for _ in range(num_layers):
            layer = Block(hidden_size, dropout_rate, num_heads, attention_dropout_rate)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded


class Block(nn.Module):
    def __init__(self, hidden_size, dropout_rate, num_heads, attention_dropout_rate):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size, dropout_rate)
        self.attn = Attention(num_heads, hidden_size, attention_dropout_rate)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


class Mlp(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(Mlp, self).__init__()
        mlp_hidden_size = hidden_size * 3
        self.fc1 = nn.Linear(hidden_size, mlp_hidden_size)
        self.fc2 = nn.Linear(mlp_hidden_size, hidden_size)
        ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


def swish(x):
    return x * torch.sigmoid(x)


class Attention(nn.Module):
    def __init__(self, num_heads, hidden_size, attention_dropout_rate):
        super(Attention, self).__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        self.proj_dropout = nn.Dropout(attention_dropout_rate)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output
