import torch
import torch.nn as nn
from egnn_pytorch import EGNN
from torch.nn.utils.rnn import pad_sequence


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BiLSTMSequenceEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, num_layers=2, out_dim=128, dropout=0.1):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        lengths = None
        if mask is not None:
            lengths = mask.sum(dim=1).to(torch.long).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.bilstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=x.size(1)
            )
        else:
            lstm_out, _ = self.bilstm(x)

        lstm_out = self.output_projection(lstm_out)
        return self.dropout(lstm_out)


class GlobalContextEncoder(nn.Module):
    def __init__(self, dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        key_padding_mask = None if mask is None else ~mask.bool()
        out = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.norm(out)


class AttentionFusion(nn.Module):
    def __init__(self, feature_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1),
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def _stack_feats(self, feats):
        if len(feats) == 0:
            raise ValueError('AttentionFusion requires at least one feature tensor')
        if len(feats) == 1:
            return feats[0], None
        stacked = torch.stack(feats, dim=2)
        mean_feat = stacked.mean(dim=2)
        max_feat = stacked.max(dim=2).values
        return stacked, torch.cat([mean_feat, max_feat], dim=-1)

    def forward(self, cnn_feat=None, lstm_feat=None, global_feat=None, mask=None, use_gate=True):
        feats = [feat for feat in (cnn_feat, lstm_feat, global_feat) if feat is not None]
        if len(feats) == 1:
            return self.out_norm(feats[0])

        stacked, summary = self._stack_feats(feats)
        if stacked.dim() != 4:
            raise RuntimeError('Internal fusion error: stacked feature tensor must be 4D')

        if use_gate:
            mean_feat = stacked.mean(dim=2)
            max_feat = stacked.max(dim=2).values
            scores = []
            for idx in range(stacked.size(2)):
                feat_i = stacked[:, :, idx, :]
                score_in = torch.cat([feat_i, mean_feat, max_feat], dim=-1)
                scores.append(self.score_mlp(score_in))
            gate = torch.softmax(torch.cat(scores, dim=-1), dim=-1).unsqueeze(-1)
            fused = torch.sum(stacked * gate, dim=2)
        else:
            fused = stacked.mean(dim=2)

        if summary is not None:
            fused = self.context_mlp(summary) + fused

        key_padding_mask = None if mask is None else ~mask.bool()
        kv = torch.cat(feats, dim=1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.repeat(1, len(feats))

        attn_out, _ = self.cross_attn(
            query=fused,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.out_norm(fused + self.dropout(attn_out))


class SequenceConv(nn.Module):
    def __init__(self, MAX_LENGTH, node_dim=26, sequence_mode='dual', use_cnn=True, use_bilstm=True, use_global=True, use_gate=True):
        super().__init__()
        if sequence_mode not in ('dual', 'cnn', 'lstm'):
            raise ValueError(f"Unsupported sequence_mode: {sequence_mode}")
        self.dim = 128
        self.seq_mode = sequence_mode
        self.max_length = MAX_LENGTH
        self.use_cnn = use_cnn
        self.use_bilstm = use_bilstm
        self.use_global = use_global
        self.use_gate = use_gate

        if not (self.use_cnn or self.use_bilstm or self.use_global):
            raise ValueError('At least one of use_cnn, use_bilstm, use_global must be True')

        self.protein_embed = Embeddings(node_dim, self.dim, self.max_length, 0.1)
        self.cnn = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.dim, 64, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Conv1d(128, 128, kernel_size=7, padding=3),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv1d(self.dim, 64, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Conv1d(128, 128, kernel_size=5, padding=2),
                nn.GELU(),
            ),
        ])
        self.cnn_fuse = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.GELU(),
            nn.LayerNorm(self.dim),
        )
        self.cnn_residual = nn.Linear(self.dim, self.dim)
        self.cnn_norm = nn.LayerNorm(self.dim)
        self.bilstm_encoder = BiLSTMSequenceEncoder(
            input_dim=self.dim, hidden_dim=96, num_layers=2, out_dim=self.dim, dropout=0.15
        )
        self.global_encoder = GlobalContextEncoder(dim=self.dim, num_heads=4, dropout=0.1)
        self.fusion = AttentionFusion(feature_dim=self.dim, num_heads=4, dropout=0.1)
        self.output_norm = nn.LayerNorm(self.dim)
        self.dropout = nn.Dropout(0.1)

    def _make_mask(self, protein):
        return protein.ne(0)

    def forward(self, protein):
        protein_embed = self.protein_embed(protein)
        mask = self._make_mask(protein)
        mask_f = mask.unsqueeze(-1).float()
        protein_embed = protein_embed * mask_f

        cnn_feat = None
        lstm_feat = None
        global_feat = None

        if self.use_cnn:
            cnn_in = protein_embed.permute(0, 2, 1)
            cnn_views = [branch(cnn_in).permute(0, 2, 1) for branch in self.cnn]
            cnn_feat = self.cnn_fuse(torch.cat(cnn_views, dim=-1))
            cnn_feat = self.cnn_norm(cnn_feat + self.cnn_residual(protein_embed))
            cnn_feat = cnn_feat * mask_f

        if self.use_bilstm:
            lstm_feat = self.bilstm_encoder(protein_embed, mask=mask) * mask_f

        if self.use_global:
            global_feat = self.global_encoder(protein_embed, mask=mask) * mask_f

        active = [feat for feat in (cnn_feat, lstm_feat, global_feat) if feat is not None]
        if self.seq_mode == 'cnn' and cnn_feat is not None:
            return self.output_norm(self.dropout(cnn_feat))
        if self.seq_mode == 'lstm' and lstm_feat is not None:
            return self.output_norm(self.dropout(lstm_feat))
        if len(active) == 1:
            return self.output_norm(self.dropout(active[0]))

        fused_feat = self.fusion(
            cnn_feat=cnn_feat,
            lstm_feat=lstm_feat,
            global_feat=global_feat,
            mask=mask,
            use_gate=self.use_gate,
        )
        return self.output_norm(self.dropout(fused_feat))


class EquivariantGraph(nn.Module):
    def __init__(self, num_embeddings, n_layers=3, dim=128):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, embedding_dim=dim)
        self.egnn_layers = nn.ModuleList([
            EGNN(dim=dim, m_dim=dim * 2, edge_dim=1) for _ in range(n_layers)
        ])
        self.elu = nn.ELU()
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(0.1)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, data):
        x, pos, edge, batch = data.x, data.pos, data.edge, data.batch
        device = x.device

        if hasattr(data, 'ptr'):
            counts = (data.ptr[1:] - data.ptr[:-1]).tolist()
        else:
            counts = torch.bincount(batch).tolist()

        edge = edge.view(-1)
        node_offset = 0
        edge_offset = 0
        graph_outputs = []

        for node_num in counts:
            x_i = x[node_offset:node_offset + node_num]
            pos_i = pos[node_offset:node_offset + node_num]

            edge_num = node_num * node_num
            edge_i = edge[edge_offset:edge_offset + edge_num]
            if edge_i.numel() != edge_num:
                edge_i = torch.zeros(edge_num, device=device, dtype=pos_i.dtype)

            x_i = self.embed(x_i).unsqueeze(0)
            pos_i = pos_i.unsqueeze(0)

            pos_i = pos_i - pos_i.mean(dim=1, keepdim=True)

            edge_i = edge_i.reshape((node_num, node_num, -1)).unsqueeze(0)

            x_in = x_i
            for i, egnn in enumerate(self.egnn_layers):
                x_res = x_i
                x_i, pos_i = egnn(x_i, pos_i, edge_i)
                x_i = self.layer_norms[i](x_i + x_res)
                if i < len(self.egnn_layers) - 1:
                    x_i = self.dropout(self.elu(x_i))
                pos_i = pos_i.clamp(-100, 100)


            x_i = x_i + x_in
            x_i = self.final_norm(x_i)
            graph_outputs.append(x_i.squeeze(0))

            node_offset += node_num
            edge_offset += edge_num

        x_out = pad_sequence(graph_outputs, batch_first=True)
        return x_out


class PLMAdapter(nn.Module):
    def __init__(self, input_dim=1280, output_dim=128, drop_out=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.proj(x)


class LinkerConfEncoder(nn.Module):
    def __init__(self, input_dim=64, output_dim=128, drop_out=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class SmilesEncoder(nn.Module):
    def __init__(self, input_dim=384, output_dim=128, drop_out=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class MyModel(nn.Module):
    def __init__(self,
                 ligase_ligand_model,
                 ligase_model,
                 target_ligand_model,
                 target_model,
                 linker_model,
                 dim=128,
                 drop_out=0.2,
                 plm_dim=1280,
                 linker_conf_dim=64,
                 smiles_dim=384,
                 use_cross_attention=True
                 ):
        super().__init__()
        self.ligase_ligand_model = ligase_ligand_model
        self.ligase_model = ligase_model
        self.target_ligand_model = target_ligand_model
        self.target_model = target_model
        self.linker_model = linker_model
        self.use_cross_attention = use_cross_attention

        self.target_plm_adapter = PLMAdapter(input_dim=plm_dim, output_dim=dim, drop_out=drop_out)
        self.e3_plm_adapter = PLMAdapter(input_dim=plm_dim, output_dim=dim, drop_out=drop_out)
        self.linker_conf_encoder = LinkerConfEncoder(input_dim=linker_conf_dim, output_dim=dim, drop_out=drop_out)
        self.smiles_encoder = SmilesEncoder(input_dim=smiles_dim, output_dim=dim, drop_out=drop_out)
        self.fusion_dim = dim

        self.cross_attn_t2l = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=4,
            dropout=drop_out,
            batch_first=True
        )
        self.cross_attn_l2t = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=4,
            dropout=drop_out,
            batch_first=True
        )
        self.cross_ln_t = nn.LayerNorm(dim)
        self.cross_ln_l = nn.LayerNorm(dim)

        self.relu = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(drop_out)
        self.dropout2 = nn.Dropout(drop_out)
        self.dropout3 = nn.Dropout(drop_out)
        self.attn_dropout = nn.Dropout(drop_out)


        self.fc1 = nn.Linear(dim * 5, 1024)
        self.ln1 = nn.LayerNorm(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.ln2 = nn.LayerNorm(512)
        self.fc3 = nn.Linear(512, 256)
        self.ln3 = nn.LayerNorm(256)
        self.out = nn.Linear(256, 2)

    def forward(self,
                ligase_ligand,
                ligase,
                target_ligand,
                target,
                linker,
                target_plm=None,
                e3_plm=None,
                linker_conf=None,
                smiles_chemberta=None):
        v_0 = self.ligase_ligand_model(ligase_ligand)
        v_1 = self.target_ligand_model(target_ligand)
        v_2 = self.linker_model(linker)
        v_t = self.target_model(target)
        v_l = self.ligase_model(ligase)

        def align_plm(plm_feat, target_len, batch_size, adapter):
            if plm_feat is None:
                return torch.zeros((batch_size, target_len, 128), device=v_t.device, dtype=v_t.dtype)
            if plm_feat.size(1) < target_len:
                pad_len = target_len - plm_feat.size(1)
                plm_feat = torch.nn.functional.pad(plm_feat, (0, 0, 0, pad_len))
            elif plm_feat.size(1) > target_len:
                plm_feat = plm_feat[:, :target_len, :]
            return adapter(plm_feat)

        batch_size = v_t.size(0)
        target_plm_proj = align_plm(target_plm, v_t.size(1), batch_size, self.target_plm_adapter)
        e3_plm_proj = align_plm(e3_plm, v_l.size(1), batch_size, self.e3_plm_adapter)
        v_t = v_t + target_plm_proj
        v_l = v_l + e3_plm_proj

        if linker_conf is None:
            linker_conf_emb = torch.zeros((batch_size, self.fusion_dim), device=v_t.device, dtype=v_t.dtype)
        else:
            linker_conf_emb = self.linker_conf_encoder(linker_conf)

        if smiles_chemberta is None:
            smiles_emb = torch.zeros((batch_size, self.fusion_dim), device=v_t.device, dtype=v_t.dtype)
        else:
            smiles_emb = self.smiles_encoder(smiles_chemberta)

        target_seq_len = v_t.size(1)
        ligase_seq_len = v_l.size(1)
        target_ligand_len = v_1.size(1)
        linker_len = v_2.size(1)
        ligase_ligand_len = v_0.size(1)

        v_d = torch.cat([v_1, v_2, v_0], dim=1)

        v_t_d = torch.cat([v_t, v_d], dim=1)
        v_l_d = torch.cat([v_l, v_d], dim=1)

        if self.use_cross_attention:
            t_context, t2l_attn_weights = self.cross_attn_t2l(
                query=v_t_d,
                key=v_l_d,
                value=v_l_d,
                need_weights=True,
                average_attn_weights=False
            )
            l_context, l2t_attn_weights = self.cross_attn_l2t(
                query=v_l_d,
                key=v_t_d,
                value=v_t_d,
                need_weights=True,
                average_attn_weights=False
            )


            v_t_att = self.cross_ln_t(v_t_d + self.attn_dropout(t_context))
            v_l_att = self.cross_ln_l(v_l_d + self.attn_dropout(l_context))

            v_t_fused = v_t_att
            v_l_fused = v_l_att
        else:
            t2l_attn_weights = None
            l2t_attn_weights = None
            v_t_fused = v_t_d
            v_l_fused = v_l_d


        t_pool = torch.mean(v_t_fused, dim=1)
        l_pool = torch.mean(v_l_fused, dim=1)
        pair_mul = t_pool * l_pool
        pair_diff = torch.abs(t_pool - l_pool)

        global_fused = 0.5 * (t_pool + l_pool)

        fusion_feature = torch.cat([global_fused, pair_mul, pair_diff, linker_conf_emb, smiles_emb], dim=1)

        fully1 = self.relu(self.ln1(self.fc1(fusion_feature)))
        fully1 = self.dropout1(fully1)
        fully2 = self.relu(self.ln2(self.fc2(fully1)))
        fully2 = self.dropout2(fully2)
        fully3 = self.relu(self.ln3(self.fc3(fully2)))
        fully3 = self.dropout3(fully3)
        predict = self.out(fully3)

        target_drug_start = target_seq_len
        target_drug_end = target_drug_start + target_ligand_len
        target_linker_end = target_drug_end + linker_len
        target_e3_ligand_end = target_linker_end + ligase_ligand_len

        ligase_drug_start = ligase_seq_len
        ligase_drug_end = ligase_drug_start + target_ligand_len
        ligase_linker_end = ligase_drug_end + linker_len
        ligase_e3_ligand_end = ligase_linker_end + ligase_ligand_len

        attention_maps = {
            "t2l_attn": t2l_attn_weights,
            "l2t_attn": l2t_attn_weights,
        }

        token_index_map = {
            "t2l": {
                "query": {
                    "target_sequence": (0, target_seq_len),
                    "target_ligand": (target_drug_start, target_drug_end),
                    "linker": (target_drug_end, target_linker_end),
                    "e3_ligand": (target_linker_end, target_e3_ligand_end),
                },
                "key": {
                    "ligase_sequence": (0, ligase_seq_len),
                    "target_ligand": (ligase_drug_start, ligase_drug_end),
                    "linker": (ligase_drug_end, ligase_linker_end),
                    "e3_ligand": (ligase_linker_end, ligase_e3_ligand_end),
                },
            },
            "l2t": {
                "query": {
                    "ligase_sequence": (0, ligase_seq_len),
                    "target_ligand": (ligase_drug_start, ligase_drug_end),
                    "linker": (ligase_drug_end, ligase_linker_end),
                    "e3_ligand": (ligase_linker_end, ligase_e3_ligand_end),
                },
                "key": {
                    "target_sequence": (0, target_seq_len),
                    "target_ligand": (target_drug_start, target_drug_end),
                    "linker": (target_drug_end, target_linker_end),
                    "e3_ligand": (target_linker_end, target_e3_ligand_end),
                },
            },
            "drug_token_order": ["target_ligand", "linker", "e3_ligand"],
        }


        predict = predict.clamp(-20, 20)
        return predict, attention_maps, token_index_map
