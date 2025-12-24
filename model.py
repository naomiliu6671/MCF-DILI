import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn import global_mean_pool, GATConv, BatchNorm
from parsers import write_record
from utils import evaluate, AttentionFusion, graph_augment, SupConLoss, fp_augment, \
    smiles_augment, get_results, evaluate_multi


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SmilesEncoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, embedding_channels, num_heads=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(input_channels, hidden_channels, padding_idx=0)
        self.positional_encoding = PositionalEncoding(hidden_channels, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=num_heads,
            dim_feedforward=hidden_channels * 4,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Sequential(
            nn.Linear(hidden_channels, embedding_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, input_ids, src_key_padding_mask=None):
        x = self.embedding(input_ids)
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = x.mean(dim=0)
        x = self.output(x)
        return x


class TotalEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, emb_channels, fp_channels, sequence_channels, dropout, device):
        super(TotalEncoder, self).__init__()
        self.device = device
        self.embedding = emb_channels
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = GATConv(hidden_channels, emb_channels)
        self.bn2 = BatchNorm(emb_channels)

        self.fp_encoder = nn.Sequential(
            nn.Linear(fp_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, emb_channels)
        )

        self.smiles_encoder = SmilesEncoder(sequence_channels, hidden_channels, emb_channels, dropout=dropout)

    def graph_process(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        return global_mean_pool(x, batch)

    def fp_process(self, desc):
        return self.fp_encoder(fp)

    def smiles_process(self, sequences):
        sequence = torch.tensor(sequences, device=self.device)
        return self.smiles_encoder(sequence)

    def forward(self, x=None, edge_index=None, fp=None, batch=None, sequences=None, mode='multi_modal'):
        if mode == 'graph':
            return F.normalize(self.graph_process(x, edge_index, batch), p=2, dim=1)
        elif mode == 'fp':
            return F.normalize(self.fp_process(fp), p=2, dim=1)
        elif mode == 'smiles':
            return F.normalize(self.smiles_process(sequences), p=2, dim=1)
        elif mode == 'multi_modal':
            graph_emb = self.graph_process(x, edge_index, batch)
            fp_emb = self.fp_process(desc)
            smiles_emb = self.smiles_process(sequences)
            return graph_emb, fp_emb, smiles_emb


class FineTuneModel(nn.Module):
    def __init__(self, encoder, num_classes=2, freeze_encoder=False):
        super(FineTuneModel, self).__init__()
        self.encoder = encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        in_channels = self.encoder.embedding

        self.severity_embedding = nn.Linear(9, in_channels)
        self.attention_fusion = AttentionFusion(in_channels)

        self.classifier = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.ReLU(),
            nn.BatchNorm1d(in_channels * 4),
            nn.Dropout(0.1),
            nn.Linear(in_channels * 4, in_channels * 2),
            nn.ReLU(),
            nn.BatchNorm1d(in_channels * 2),
            nn.Dropout(0.1)
        )

        self.task_no = nn.Linear(in_channels * 2, num_classes)
        self.task_less = nn.Linear(in_channels * 2, num_classes)
        self.task_most = nn.Linear(in_channels * 2, num_classes)

        self.relation_head = nn.Sequential(
            nn.Linear(num_classes * 2, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, num_classes)
        )

    def compute_relations(self, prototypes, query_embeddings):
        query_ = query_embeddings.unsqueeze(1).expand(-1, prototypes.size(0), -1)
        prototypes_ = prototypes.unsqueeze(0).expand(query_embeddings.size(0), -1, -1)
        combined = torch.cat([query_, prototypes_], dim=-1)
        relations = self.relation_head(combined)
        relation_scores = relations.max(dim=-1).values
        return relation_scores

    def forward(self, x, edge_index, fp, batch, sequences, severity, mode, in_con, task):
        graph_emb, fp_emb, smiles_emb = self.encoder(x, edge_index, fp, batch, sequences, mode)
        severity_emb = self.severity_embedding(severity)
        total_ls = [graph_emb, fp_emb, smiles_emb, severity_emb]
        ls = []
        for i in range(len(in_con)):
            if in_con[i]:
                ls.append(total_ls[i])
        combined_emb = self.attention_fusion(ls)
        out = self.classifier(combined_emb)
        if task == 0:
            out = self.task_no(out)
        elif task == 1:
            out = self.task_less(out)
        elif task == 2:
            out = self.task_most(out)
        return out, {
            'graph_emb': graph_emb,
            'fp_emb': fp_emb,
            'smiles_emb': smiles_emb,
            'severity_emb': severity_emb,
            'combined_emb': combined_emb
        }


def pretrain_step(encoder, args, train_data_batches, train_task_id_list, current_time):
    model = encoder
    optimizer = optim.Adam(model.parameters(), lr=args.lr_pretrain)
    constract_criterion = SupConLoss(temperature=args.temperature)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    record_path = f'{args.result_path}/{args.train_dataset}/{args.train_shot}/'
    device = args.device
    model.to(device)
    model.train()
    best_loss = float('inf')
    cnt = 0
    patience = 3
    for epoch in range(args.train_epochs):
        losses_total = 0
        for task_id in train_task_id_list:
            t_data = train_data_batches[task_id]
            t_data = t_data['data_loader']
            loss_q = 0
            for data in t_data:
                data = data.to(device)
                graph_1 = graph_augment(data)
                graph_emb1 = model(graph_1.x, graph_1.edge_index, None, graph_1.batch, None, mode='graph')
                fp_emb = model(None, None, data.fp, None, None, mode='fp')
                smiles_emb = model(None, None, None, None, data.sequence, mode='smiles')
                loss_gf = constract_criterion(torch.stack([graph_emb1, fp_emb], dim=1), data.y)
                loss_gs = constract_criterion(torch.stack([graph_emb1, smiles_emb], dim=1), data.y)
                loss_fs = constract_criterion(torch.stack([fp_emb, smiles_emb], dim=1), data.y)
                loss_sum = loss_gf + loss_gs + loss_fs
                loss_q += loss_sum.item()
                optimizer.zero_grad()
                loss_sum.backward()
                optimizer.step()
            losses_total += loss_q / len(t_data)
        scheduler.step()
        losses_mean = losses_total / len(train_task_id_list)
        word = f'Epoch: {epoch + 1}, Loss: {losses_mean:.4f}'
        print(word)
        write_record(word, record_path, current_time)
        if losses_mean < best_loss:
            best_loss = losses_mean
            cnt = 0
        else:
            cnt += 1
            if cnt >= patience:
                break
    return model


def finetune_step(encoder, args, test_data_batches, test_task_id_list, query_smiles, current_time, in_con, times):
    model = FineTuneModel(encoder, num_classes=args.num_classes, freeze_encoder=False)
    optimizer = torch.optim.Adam([{'params': model.severity_embedding.parameters(), 'lr': 1e-5},
                                  {'params': model.attention_fusion.parameters(), 'lr': 1e-5},
                                  {'params': model.classifier.parameters(), 'lr': 1e-5},
                                  {'params': model.task_no.parameters(), 'lr': 1e-5},
                                  {'params': model.task_less.parameters(), 'lr': 1e-5},
                                  {'params': model.task_most.parameters(), 'lr': 1e-5},
                                  {'params': model.encoder.parameters(), 'lr': 1e-5},
                                  {'params': model.relation_head.parameters(), 'lr': 1e-5}])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_con = SupConLoss(temperature=args.temperature)
    model.to(args.device)
    pred_best_matrix = [0, 0, 0, 0, 0]
    patience = args.fine_tune_patience
    add_word = f'few-shot'
    save_path = f'{args.model_save_path}/{args.test_dataset}/{add_word}/model_state_dict_{times+1}.joblib'
    result_path = f'{args.result_path}/{args.test_dataset}/{add_word}/total_metrics_{times+1}.csv'
    record_path = f'{args.result_path}/{args.test_dataset}/{add_word}/'
    write_record(f'save_path: {save_path}, result_path: {result_path} \nFine tune starting ...', record_path, current_time)
    cnt = 0
    column = ['smiles', 'no_pred', 'less_pred', 'most_pred', 'no_prob', 'less_prob', 'most_prob', 'no_true', 'less_true', 'most_true']
    best_pre = pd.DataFrame(columns=column)
    best_pre['smiles'] = query_smiles
    columns = {0: ['no_pred', 'no_prob', 'no_true'], 1: ['less_pred', 'less_prob', 'less_true'], 2: ['most_pred', 'most_prob', 'most_true']}

    for epoch in range(args.fine_tune_epochs):
        print(f'Fine-tune epoch: {epoch + 1}')
        write_record(f'Fine-tune epoch: {epoch + 1}', record_path, current_time)
        total_loss = 0
        model.train()
        auc_epoch = 0
        epoch_pre = pd.DataFrame(columns=column)
        epoch_pre['smiles'] = query_smiles
        epoch_matrix = pd.DataFrame(columns=['acc', 'auc', 'f1', 'pre', 'recall'])
        for task_id in test_task_id_list:
            t_data = test_data_batches[task_id]
            s_loader = t_data['s_data']
            for s_data in s_loader:
                try:
                    s_data = s_data.to(args.device)
                except RuntimeError as e:
                    print(f"Error moving data to {args.device}: {e}")
                s_representation, _ = model(s_data.x.float(), s_data.edge_index, s_data.fp, s_data.batch, s_data.sequence, s_data.severity, 'multi_modal', in_con, task_id)
                s_labels = s_data.y
                loss_cls = criterion_cls(s_representation, s_labels)
                s_data_graph_aug = graph_augment(s_data)
                s_data_fp_aug = fp_augment(s_data.fp)
                s_data_smiles_aug = smiles_augment(s_data.sequence, args.vocab_size, args.device)

                rep1, _ = model(s_data.x.float(), s_data.edge_index, s_data.fp, s_data.batch, s_data.sequence,
                             s_data.severity, 'multi_modal', in_con, task_id)
                rep2, _ = model(s_data_graph_aug.x.float(), s_data_graph_aug.edge_index, s_data_fp_aug,
                             s_data_graph_aug.batch, s_data_smiles_aug, s_data.severity, 'multi_modal', in_con, task_id)
                features = torch.stack([rep1, rep2], dim=1)
                loss_con = criterion_con(features, labels=s_data.y)
                loss = loss_cls + args.contrastive_weight * loss_con
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            model.eval()
            preds_, probs, label, smiles = diliprediction(model, t_data, args, in_con, task_id)
            col = columns[task_id]
            for i in range(len(smiles)):
                epoch_pre.loc[epoch_pre['smiles'] == smiles[i], col[0]] = preds_[i]
                epoch_pre.loc[epoch_pre['smiles'] == smiles[i], col[1]] = probs[i]
                epoch_pre.loc[epoch_pre['smiles'] == smiles[i], col[2]] = label[i]
            evalu_matrix = evaluate(preds_, probs, label)
            acc_, auc_, f1_, pre_, recall_ = evalu_matrix['acc'], evalu_matrix['auc'], evalu_matrix['f1'], evalu_matrix['precision'], evalu_matrix['recall']
            epoch_matrix.loc[task_id] = [acc_, auc_, f1_, pre_, recall_]
            auc_epoch += auc_
            word = (f' task: {task_id}, Acc: {acc_:.4f}, AUC: {auc_:.4f}, F1: {f1_:.4f}, Pre: {pre_:.4f}, Recall: {recall_:.4f}')
            write_record(word, record_path, current_time)
        total_results = get_results(epoch_pre, [['no_prob', 'less_prob', 'most_prob'], ['no_true', 'less_true', 'most_true']])
        total_evalu_matrix = evaluate_multi(total_results[1], total_results[2], total_results[3])
        acc, auc, f1, pre, recall = (total_evalu_matrix['acc'], total_evalu_matrix['auc'], total_evalu_matrix['f1'],
                                     total_evalu_matrix['precision'], total_evalu_matrix['recall'])
        scheduler.step(auc)
        if auc > pred_best_matrix[1]:
            best_pre = epoch_pre
            best_pre.to_csv(result_path, index=False)
            best_model = model.state_dict()
            torch.save(best_model, save_path)
            pred_best_matrix = [acc, auc, f1, pre, recall]
            cnt = 0
        elif pred_best_matrix[1] < 0.5:
            cnt = 0
        else:
            cnt += 1
        word = (f' Mean Acc: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, Pre: {pre:.4f}, Recall: {recall:.4f}, best_auc: {pred_best_matrix[1]:.4f}, cnt: {cnt}')
        print(word)
        write_record(word, record_path, current_time)
        if cnt == patience:
            word_ = f'Fine-tune early stop at epoch: {epoch + 1}, best_acc: {pred_best_matrix[0]:.4f}, best_auc: {pred_best_matrix[1]:.4f}, best_f1: {pred_best_matrix[2]:.4f}, best_pre: {pred_best_matrix[3]:.4f}, best_recall: {pred_best_matrix[4]:.4f}'
            print(word_)
            write_record(word_, record_path, current_time)
            break
    return best_pre, [['no_prob', 'less_prob', 'most_prob'], ['no_true', 'less_true', 'most_true']]


def diliprediction(encoder, test_data, args, in_con, task_id):
    s_dataloader, q_dataloader = test_data['s_data'], test_data['q_data']
    preds_, probs_, labels_, smiles = [], [], [], []
    for q_data in q_dataloader:
        q_data = q_data.to(args.device)
        q_representation, _ = encoder(q_data.x.float(), q_data.edge_index, q_data.fp, q_data.batch, q_data.sequence, q_data.severity, 'multi_modal', in_con, task_id)
        q_labels = q_data.y.float().cpu().detach().numpy()
        probs = torch.softmax(q_representation, dim=1)
        preds = probs.argmax(dim=1)
        preds_.extend(preds.cpu().detach().numpy())
        probs_.extend(probs.cpu().detach().numpy()[:, 1])
        labels_.extend(q_labels)
        smiles.extend(q_data.smiles)
        del q_data, q_representation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return preds_, probs_, labels_, smiles
