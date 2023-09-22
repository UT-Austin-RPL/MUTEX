import copy
from .transformer_modules import *

class TransformerCrossEncoder(nn.Module):
    def __init__(self,
                 input_size,
                 context_size,
                 num_layers,
                 num_heads,
                 head_output_size,
                 mlp_hidden_size,
                 dropout,
                 attn_dropout,
                 cross_attn_ind,
                 **kwargs):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()

        self.attention_output = {}
        self.cross_attn_ind = cross_attn_ind

        for ind in range(num_layers):
            if ind in cross_attn_ind:
                self.layers.append(
                    nn.ModuleList([
                        Norm(input_size),
                        Norm(context_size),
                        CrossAttention(input_size,
                                       kv_dim=context_size,
                                       num_heads=num_heads,
                                       head_output_size=head_output_size,
                                       dropout=dropout,
                                       attn_dropout=attn_dropout,
                        ),
                        Norm(input_size),
                        TransformerFeedForwardNN(input_size,
                                                 mlp_hidden_size,
                                                 dropout=dropout)
                    ])
                )
            else:
                self.layers.append(
                    nn.ModuleList([
                        Norm(input_size),
                        nn.Identity(),
                        Attention(input_size,
                                  num_heads=num_heads,
                                  head_output_size=head_output_size,
                                  dropout=dropout,
                                  attn_dropout=attn_dropout,
                        ),
                        Norm(input_size),
                        TransformerFeedForwardNN(input_size,
                                                 mlp_hidden_size,
                                                 dropout=dropout)
                    ])
                )

            self.attention_output[ind] = None
        self.seq_len = None
        self.num_elements = None
        self.mask = None
        self.cross_mask = None

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def compute_mask(self, input_shape, context_shape, context_mask=None):
        # input_shape = (:, seq_len, num_elements)
        if (self.num_elements is None) or \
                (self.seq_len is None) or \
                (self.num_elements != input_shape[2]) or \
                (self.seq_len != input_shape[1]):

            self.seq_len = input_shape[1]
            self.num_elements = input_shape[2]
            self.original_mask = (
                    torch.triu(torch.ones(self.seq_len, self.seq_len)) - \
                    torch.eye(self.seq_len, self.seq_len)
            )
            self.mask = 1 - self.original_mask.repeat_interleave(
                    self.num_elements, dim=-1
            ).repeat_interleave(self.num_elements, dim=-2).unsqueeze(0)
            # (1, N, N), N = seq_len * num_elements

        ## Always recompute
        if context_mask is None:
            self.cross_mask = torch.ones(1, self.num_elements*self.seq_len, context_shape[1])
        else:
            self.cross_mask = context_mask.unsqueeze(dim=1).repeat_interleave(self.num_elements*self.seq_len, dim=1)

    def forward(self, x, context, mask=None):
        self.mask = self.mask.to(x.device)
        self.cross_mask = self.cross_mask.to(x.device)
        for layer_idx, (att_norm, cross_norm, att, ff_norm, ff) in enumerate(self.layers):
            if layer_idx in self.cross_attn_ind:
                x = x + drop_path(att(att_norm(x), context=cross_norm(context), mask=self.cross_mask))
            else:
                if mask is not None:
                    x = x + drop_path(att(att_norm(x), mask=mask))
                elif self.mask is not None:
                    x = x + drop_path(att(att_norm(x), mask=self.mask))
                else: # no masking, just use full attention
                    x = x + drop_path(att(att_norm(x)))

            #if not self.training: ## Removing unnecessary storage of everything
            #    self.attention_output[layer_idx] = att.att_weights
            x = x + self.drop_path(ff(ff_norm(x)))
        return x

    @property
    def device(self):
        return next(self.parameters()).device


if __name__ == '__main__':
    bs, t = 2, 4
    test_ind = 2
    num_modalities = 5
    num_context_tokens = 128
    model = TransformerCrossEncoder(
        input_size=64,
        context_size=64,
        num_layers=2,
        num_heads=6,
        head_output_size=64,
        mlp_hidden_size=256,
        dropout=0.1,
        attn_dropout=0.1,
        cross_attn_ind=[0],
    )
    model.eval()
    print('# Learnable Params: %d' % sum(p.numel() for p in model.parameters() if p.requires_grad))

    context = torch.rand((bs, num_context_tokens, 64))
    inputs = torch.rand((bs, t, num_modalities, 64))
    sh = inputs.shape
    inputs_ch = copy.deepcopy(inputs)
    inputs_ch[:, test_ind:] += 1

    model.compute_mask(sh, context_shape=context.shape)


    inputs = inputs.reshape(sh[0], sh[1]*sh[2], sh[3])
    inputs_ch = inputs_ch.reshape(sh[0], sh[1]*sh[2], sh[3])

    feats = model(inputs, context=context)
    feats = feats.reshape(sh[0], sh[1], sh[2], feats.shape[-1])
    feats_ch = model(inputs_ch, context=context)
    feats_ch = feats_ch.reshape(sh[0], sh[1], sh[2], feats_ch.shape[-1])
    print(feats.shape)
    print(feats_ch.shape)
    print("Output should NOT change here")
    print(feats[:, :test_ind] - feats_ch[:, :test_ind])
    print("Output should change here")
    print(feats[:, test_ind:] - feats_ch[:, test_ind:])
