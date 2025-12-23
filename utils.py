import torch.nn.functional as F
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, \
    balanced_accuracy_score, classification_report, confusion_matrix
from torch_geometric.data import Data


def graph_augment(data):
    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    keep_prob = 0.95
    mask = torch.rand(num_edges) < keep_prob
    return Data(x=data.x, edge_index=edge_index[:, mask], batch=data.batch)


def smiles_augment(sequence, vocab_size, device='cuda', mask_prob=0.05):
    sequence = torch.tensor(sequence, dtype=torch.long, device=device)
    with torch.no_grad():
        mask = torch.rand(sequence.shape).to(device) < mask_prob
        random_tokens = torch.randint(0, vocab_size, sequence.shape, device=device)
        augmented = torch.where(mask, random_tokens, sequence)
    return augmented


def fp_augment(desc, noise_level=0.05):
    noise = torch.randn_like(desc) * noise_level
    return desc + noise


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels=None, mask=None):
        device = features.device
        if len(features.shape) < 2:
            raise ValueError('`features` needs to be at least 2D')
        if len(features.shape) == 2:
            features = features.unsqueeze(1)
        batch_size = features.shape[0]
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        if labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float()
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits) * logits_mask
        epsilon = 1e-8
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + epsilon)
        mean_log_prob = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob.mean()

        return loss


class AttentionFusion(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, embeddings):
        x = torch.stack(embeddings, dim=1)
        residual = x
        B, N, D = x.size()
        q = self.query(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1) + 1e-8
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)

        fused_emb = torch.matmul(attn_weights, v)
        fused_emb = fused_emb.transpose(1, 2).contiguous().view(B, N, D)
        fused_emb = self.out_proj(fused_emb)
        fused_emb = self.norm(fused_emb + residual)
        final_emb = fused_emb.mean(dim=1)
        return final_emb


def evaluate(preds, probs, label):
    acc = accuracy_score(label, preds)
    auc = roc_auc_score(label, probs)
    f1 = f1_score(label, preds)
    precision = precision_score(label, preds)
    recall = recall_score(label, preds)
    matrix = {'acc': acc, 'auc': auc, 'f1': f1, 'precision': precision, 'recall': recall}
    return matrix


def evaluate_multi(preds, probs, label):
    probs = torch.softmax(torch.tensor(probs), dim=1)
    acc = balanced_accuracy_score(label, preds)
    auc = roc_auc_score(label, probs, multi_class='ovr', average='macro')
    f1 = f1_score(label, preds, average='macro')
    precision = precision_score(label, preds, average='macro')
    recall = recall_score(label, preds, average='macro')
    matrix = {'acc': acc, 'auc': auc, 'f1': f1, 'precision': precision, 'recall': recall}
    return matrix


def evaluate_total(preds, probs, label):
    probs = torch.softmax(torch.tensor(probs), dim=1)
    acc = balanced_accuracy_score(label, preds)
    auc = roc_auc_score(label, probs, multi_class='ovr', average='macro')
    f1 = f1_score(label, preds, average='macro')
    precision = precision_score(label, preds, average='macro')
    recall = recall_score(label, preds, average='macro')
    class_report = classification_report(label, preds, output_dict=False)
    conf_matrix = confusion_matrix(label, preds).tolist()
    matrix = {'acc': acc, 'auc': auc, 'f1': f1, 'precision': precision, 'recall': recall, 'class_report': class_report,
              'confusion_matrix': conf_matrix, 'unique_preds': np.unique(preds).tolist(), 'unique_labels': np.unique(label).tolist()}
    return matrix


def mean_std(data_list):
    return str(f'{np.mean(data_list):.3f}') + '±' + str(f'{np.std(data_list):.3f}')


def get_results(predicts, columns):
    smiles = predicts['smiles']
    probs = predicts[columns[0]]
    probs_sum = probs.sum(axis=1)
    scaled_probs = probs / probs_sum[0]
    scaled_probs.columns = [0, 1, 2]
    preds = scaled_probs.idxmax(axis=1).tolist()
    scaled_probs = scaled_probs.values.tolist()
    labels_ = predicts[columns[1]]
    labels_.columns = [0, 1, 2]
    true = labels_.idxmax(axis=1).tolist()
    return [smiles, preds, scaled_probs, true]
