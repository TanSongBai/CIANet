import torch
import timm
import math
from collections import OrderedDict
import clip
import torch.nn as nn
import torch
import torch.nn.functional as F
import os


def _weights_init_kaiming(m):
    # classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)


class MLP(nn.Module):
    def __init__(self, ch_in, ch_mid, ch_out):
        super(MLP, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(ch_in, ch_mid),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(ch_mid, ch_out),
            nn.ReLU()
        )

    def forward(self, v):
        v = self.fc1(v)
        v = self.fc2(v)
        return v


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.ln_2 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_3 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, y: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, y, y, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """

        Args:
            x:  query vector
            y: key and value vectors

        Returns: cross-attention vector

        """
        v = y + self.attention(self.ln_1(x), self.ln_2(y))
        v = v + self.mlp(self.ln_3(v))
        return v


class CIALayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        # The self-attention Layer
        self.self_attention11 = SelfAttentionBlock(d_model, n_head, attn_mask)
        self.self_attention12 = SelfAttentionBlock(d_model, n_head, attn_mask)
        self.self_attention21 = SelfAttentionBlock(d_model, n_head, attn_mask)
        self.self_attention22 = SelfAttentionBlock(d_model, n_head, attn_mask)
        # The cross-attention Layer
        self.cross_attention1 = CrossAttentionBlock(d_model, n_head, attn_mask)
        self.cross_attention2 = CrossAttentionBlock(d_model, n_head, attn_mask)

    def forward(self, img_feats, txt_feats):

        img_att_out = self.self_attention11(img_feats)
        txt_att_out = self.self_attention21(txt_feats)

        t2i_att_out = self.cross_attention1(txt_att_out, img_att_out)
        i2t_att_out = self.cross_attention2(img_att_out, txt_att_out)

        t2i_att_out = self.self_attention12(t2i_att_out)
        i2t_att_out = self.self_attention22(i2t_att_out)

        return t2i_att_out, i2t_att_out

class CIANet(nn.Module):
    def __init__(self, device, clip_net='ViT-B/32', in_size=512, init_mode='kaiming_norm'):
        super(CIANet, self).__init__()

        self.in_size = in_size
        self.base, _ = clip.load(clip_net, device=device)
        self.logit_scale = self.base.logit_scale
        self.encode_image = self.base.encode_image
        self.encode_text = self.base.encode_text
        self.CIALayer = CIALayer(d_model=512, n_head=8, attn_mask=None)
        self.MLP = MLP(ch_in=1024, ch_mid=256, ch_out=1)

    def forward(self, img, tokens_c, tokens_a):
        img_feature = self.encode_image(img).to(torch.float32)
        con_feature = self.encode_text(tokens_c).to(torch.float32)
        aes_feature = self.encode_text(tokens_a).to(torch.float32)


        img_feature_n = img_feature / img_feature.norm(dim=1, keepdim=True)
        con_feature_n = con_feature / con_feature.norm(dim=1, keepdim=True)
        aes_feature_n = aes_feature / aes_feature.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        con_feature_n = con_feature_n.view(-1, 1, self.in_size)
        aes_feature_n = aes_feature_n.view(-1, 5, self.in_size)
        img_feature_n = img_feature_n.view(-1, 1, self.in_size)
        logits_per_con = [logit_scale * img_feature_n[i] @ con_feature_n[i].t() for i in range(con_feature_n.shape[0])]
        logits_per_aes = [logit_scale * img_feature_n[i] @ aes_feature_n[i].t() for i in range(aes_feature_n.shape[0])]
        logits_per_con = torch.cat(logits_per_con, dim=0)
        logits_per_aes = torch.cat(logits_per_aes, dim=0)

        img_feature_in = img_feature.unsqueeze(dim=0)
        con_feature_in = con_feature.unsqueeze(dim=0)
        img_feature_out, con_feature_out = self.CIALayer(img_feature_in, con_feature_in)
        fusion_feature = torch.cat([img_feature_out, con_feature_out], dim=2)
        logits_per_qua = self.MLP(fusion_feature.squeeze(dim=0))
        return logits_per_qua, logits_per_con, logits_per_aes


if __name__ == '__main__':
    from thop import profile
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']
    batch_size = 1
    text_c = ["a beautiful women"]*batch_size
    text_a = [[f"The aesthetic quality of the image is {d}" for d in qualitys]]*batch_size
    token_c = torch.stack([clip.tokenize(prompt) for prompt in text_c]).to(device)
    token_a = torch.stack([clip.tokenize(prompt) for prompt in text_a], dim=0).to(device)
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    brisque_feat = torch.randn(batch_size, 36).to(device)
    model = CIANet(device=device).to(device)
    pred_c = []
    pred_q = []
    pred_a = []
    pred_nss = []

    input_img = x
    input_token_c = token_c.view(-1, 77)
    input_token_a = token_a.view(-1, 77)
    logits_per_qua, logits_per_con, logits_per_aes = model(input_img, input_token_c, input_token_a)
    logits_per_aes = F.softmax(logits_per_aes, dim=1)

    flops, params = profile(model, inputs=(input_img, input_token_c, input_token_a))
    print('Flops: % .4fG' % (2 * flops / 1000000000))  # 计算量
    print('params: % .4fM' % (params / 1000000))  # 参数量：等价与上面的summary输出的Total params值

