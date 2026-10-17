"""
Retrieval engine for IDeal demo.
Wraps the VSE model + BERT text encoder to compute embeddings and
perform text-to-scene retrieval. Does NOT depend on the original
dataset classes (which use hardcoded relative paths).
"""
import sys
import os
import json
import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from transformers import BertTokenizer
import arguments
from lib.vse_bert import VSEModel


class RetrievalEngine:
    """Loads model once, provides text encoding and scene retrieval."""

    MODEL_PATH = os.path.join(_PROJECT_ROOT, "runs", "scanrefer_train_butd_ESAregion_bigru", "transfer.pth")
    BERT_PATH = os.path.join(_PROJECT_ROOT, "bert-base-uncased")
    DATA_DIR = os.path.join(_PROJECT_ROOT, "lib", "data")
    SPLIT_DIR = os.path.join(_PROJECT_ROOT, "lib", "data", "split")
    TEXT_DIR = os.path.join(_PROJECT_ROOT, "lib", "data", "text", "scanrefer")

    def __init__(self, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Load BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.BERT_PATH)

        # Load model args (override defaults for scanrefer)
        parser = arguments.get_argument_parser()
        self.opt = parser.parse_args([
            "--data_path", self.DATA_DIR,
            "--data_name", "scanrefer",
        ])
        self.opt.vocab_size = len(self.tokenizer.vocab)

        # Load VSE model (VSEModel moves its own encoders to CUDA in __init__)
        checkpoint = torch.load(self.MODEL_PATH, map_location="cpu")
        self.model = VSEModel(self.opt)
        self.model.load_state_dict(checkpoint["model"])
        self.model.val_start()
        self.txt_encoder = self.model.txt_enc

        # Pre-compute scene embeddings and load metadata
        self._load_all_data()

    def _load_all_data(self):
        """Load point cloud features, val scene list, captions, and encode all scenes."""
        # 1. Load point cloud features (141 scenes x 200 regions x 1024)
        pt_features = np.load(os.path.join(self.DATA_DIR, "pt2vec_200_random_val.npy"))
        pt_pos = np.load(os.path.join(self.DATA_DIR, "pt2vec_200_random_pos_val.npy"))

        # Normalize positions
        x_coords = pt_pos[:, :, 0]
        y_coords = pt_pos[:, :, 1]
        x_min, x_max = x_coords.min(axis=1, keepdims=True), x_coords.max(axis=1, keepdims=True)
        y_min, y_max = y_coords.min(axis=1, keepdims=True), y_coords.max(axis=1, keepdims=True)
        pos_normalized = pt_pos.copy()
        pos_normalized[:, :, 0] = (x_coords - x_min) / (x_max - x_min + 1e-9)
        pos_normalized[:, :, 1] = (y_coords - y_min) / (y_max - y_min + 1e-9)

        # 2. Load val scene list
        with open(os.path.join(self.SPLIT_DIR, "ScanRefer_filtered_val.txt"), "r", encoding="utf-8") as f:
            val_scenes = [line.strip() for line in f if line.strip()]

        # 3. Load captions and memory from JSONL (matching precomputed features ordering)
        # The precomputed features (bert_interactive_features_5.npy) were computed from
        # bert_interactive_text.jsonl, so we must use the same descriptions for consistency.
        jsonl_path = os.path.join(_PROJECT_ROOT, "temp_results", "8_9", "self_scanrefer",
                                   "bert_interactive_text.jsonl")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            jsonl_records = [json.loads(line) for line in f]

        # Build per-scene caption list: one description per record (1410 total)
        # This matches the ordering of bert_interactive_features_5.npy exactly
        self.scene_ids = []
        self.scene_texts = []
        self.scene_memory = []
        self.images_list = []
        self.pos_list = []

        for rec in jsonl_records:
            sid = rec["scene_id"]
            # Find the scene's position index in val_scenes for image features
            scene_idx = val_scenes.index(sid)
            self.scene_ids.append(sid)
            self.scene_texts.append(rec.get("description", ""))
            self.scene_memory.append("")
            self.images_list.append(pt_features[scene_idx])
            self.pos_list.append(pos_normalized[scene_idx])

        n_samples = len(self.scene_ids)
        print(f"Loaded {n_samples} captions across {len(val_scenes)} scenes")

        # 4. Encode all scenes through the model
        # Batch process to build cap_embs
        div = 10  # downsample: one image per 10 captions
        batch_size = 64

        img_embs_all = np.zeros((len(val_scenes), 2, 1024))
        cap_embs_all = np.zeros((n_samples, 2, 1024))

        with torch.no_grad():
            # Encode images through img_enc directly (bypass forward_emb which needs captions)
            for start in range(0, len(val_scenes), batch_size):
                end = min(start + batch_size, len(val_scenes))
                batch_imgs = torch.tensor(
                    np.stack([self.images_list[i * div] for i in range(start, end)]),
                    dtype=torch.float32
                ).to(self.device)
                batch_pos = torch.tensor(
                    np.stack([self.pos_list[i * div] for i in range(start, end)]),
                    dtype=torch.float32
                ).to(self.device)
                img_lens = torch.tensor([batch_imgs.shape[1]] * (end - start)).to(self.device)

                img_emb1, img_emb2 = self.model.img_enc(batch_imgs, img_lens, batch_pos)
                # In eval mode, img_enc returns (N, 1024) not (N, 2, 1024)
                emb2 = img_emb2 if img_emb2.dim() == 2 else img_emb2[:, 1, :]
                img_embs_all[start:end, 1, :] = emb2.cpu().numpy()

            # Encode captions in batches using batched BERT tokenization
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_texts = self.scene_texts[start:end]
                batch_ids = list(range(start, end))

                # Tokenize and pad to max length (matching collate_fn_test: fixed 500)
                encodings = self.tokenizer(
                    batch_texts, padding='max_length', truncation=True, max_length=500,
                    return_tensors='pt'
                )
                caption_tensor = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                cap_lens = attention_mask.sum(dim=1)

                # Use the text encoder's forward method directly (eval mode)
                with torch.no_grad():
                    feature_cap, _ = self.txt_encoder(caption_tensor, cap_lens, None)

                cap_embs_all[batch_ids, 1, :] = feature_cap.cpu().numpy()

                if (start // batch_size) % 10 == 0:
                    print(f"  Encoded {end}/{n_samples} captions...")

        self.img_embs = img_embs_all
        self.cap_embs = cap_embs_all
        print(f"RetrievalEngine: encoded {n_samples} captions")

    def encode_text(self, text):
        """Encode text to 1024-dim embedding via BERT + text encoder."""
        caption_tokens = self.tokenizer.basic_tokenizer.tokenize(text)
        target = self._process_caption(caption_tokens)
        target = target.cuda() if torch.cuda.is_available() else target
        length = torch.tensor([target.size(0)]).cuda() if torch.cuda.is_available() else torch.tensor([target.size(0)])
        tensor = torch.zeros((1, 500))
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        tensor[0, :length] = target

        with torch.no_grad():
            _, feat = self.txt_encoder(tensor.long(), length, None)
        return feat.cpu().numpy().squeeze()

    def _process_caption(self, tokens):
        """Tokenize and prepare caption tensor (matches original process_caption)."""
        output_tokens = []
        for token in tokens:
            sub_tokens = self.tokenizer.wordpiece_tokenizer.tokenize(token)
            output_tokens.extend(sub_tokens)
        output_tokens = ['[CLS]'] + output_tokens + ['[SEP]']
        target = self.tokenizer.convert_tokens_to_ids(output_tokens)
        target = torch.Tensor(target)
        if len(target) > 500:
            target = target[:500]
        return target

    def retrieve_topk(self, query_text, k=5):
        """Retrieve top-K scenes using dot-product similarity."""
        query_emb = self.encode_text(query_text)
        query_emb = query_emb.reshape(1, -1)
        scene_embs = self.cap_embs[:, 1, :]
        sims = (query_emb @ scene_embs.T).squeeze()
        top_indices = np.argsort(sims)[::-1][:k]
        return [
            {"scene_id": self.scene_ids[idx], "text": self.scene_texts[idx],
             "score": float(sims[idx]), "rank": i+1}
            for i, idx in enumerate(top_indices)
        ]

    def compute_fusion_result(self, original_emb, answer_embs, k=5):
        """
        Compute retrieval with feature fusion.
        original_emb: (1024,) — encoded original description
        answer_embs: list of (1024,) — encoded interactive answers
        Returns top-K results with weighted fusion.
        """
        scene_embs = self.cap_embs[:, 1, :]
        ori_sims = (original_emb.reshape(1, -1) @ scene_embs.T).squeeze()

        if len(answer_embs) >= 2:
            valid_embs = np.array(answer_embs)
            valid_embs_norm = valid_embs / (np.linalg.norm(valid_embs, axis=1, keepdims=True) + 1e-8)
            fused = valid_embs_norm.mean(axis=0)
            fused = fused / (np.linalg.norm(fused) + 1e-8)
            ans_sims = (fused.reshape(1, -1) @ scene_embs.T).squeeze()
        else:
            ans_sims = ori_sims

        # Coarse-grained fusion weights (matching I_retr_Qwen_bert.py: 0.55/0.35/0.10)
        final_sims = 0.55 * ori_sims + 0.35 * ans_sims

        top_indices = np.argsort(final_sims)[::-1][:k]
        return [
            {"scene_id": self.scene_ids[idx], "text": self.scene_texts[idx],
             "score": float(final_sims[idx]), "rank": i+1}
            for i, idx in enumerate(top_indices)
        ]


# Singleton pattern
_engine = None


def get_engine(device=None):
    """Get or create the global RetrievalEngine instance."""
    global _engine
    if _engine is None:
        _engine = RetrievalEngine(device=device)
    return _engine
