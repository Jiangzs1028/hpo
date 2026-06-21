from collections import defaultdict
import numpy as np
import torch

def filter_encoder_inputs_by_uid(
    encoder_inputs: dict,
    step_index_encoder,
    step_texts,
    uids,
    sample_sign,
    eligible_uids: set,
):
    sel2orig = [i for i, uid in enumerate(uids) if uid in eligible_uids]
    if len(sel2orig) == 0:
        return None, None, None, None, None, None

    input_ids = encoder_inputs["input_ids"]
    attn_mask = encoder_inputs["attention_mask"]

    enc_inputs_sel = {
        "input_ids": input_ids[sel2orig],
        "attention_mask": attn_mask[sel2orig],
    }
    step_index_sel = [step_index_encoder[i] for i in sel2orig]
    step_texts_sel = [step_texts[i] for i in sel2orig]
    uids_sel = [uids[i] for i in sel2orig]
    sign_sel = [sample_sign[i] for i in sel2orig]
    return enc_inputs_sel, step_index_sel, step_texts_sel, uids_sel, sign_sel, sel2orig


def get_steps_text(input_ids, step_index, tokenizer, encoder_tokenizer):
    B, T = input_ids.shape

    step_texts = []
    for i in range(B):
        token_ids = input_ids[i]
        step_ids = step_index[i]
        valid_mask = step_ids != -1

        token_ids = token_ids[valid_mask]
        step_ids = step_ids[valid_mask]

        steps = []
        for s in torch.unique_consecutive(step_ids):
            if s < 0:
                continue
            step_mask = step_ids == s
            step_tokens = token_ids[step_mask]
            text = tokenizer.decode(
                step_tokens.tolist(),
                skip_special_tokens=True,
            )
            steps.append(text)
        step_texts.append(steps)

    step_index_encoder = []
    encoder_input_ids_list = []

    for steps in step_texts:
        cur_ids = []
        cur_step_end = []
        cur_len = 0

        for step_text in steps:
            step_ids_enc = encoder_tokenizer(
                step_text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]

            if len(step_ids_enc) == 0:
                continue

            cur_ids.extend(step_ids_enc)
            cur_len += len(step_ids_enc)
            cur_step_end.append(cur_len - 1)

        encoder_input_ids_list.append(cur_ids)
        step_index_encoder.append(cur_step_end)

    max_len = max((len(x) for x in encoder_input_ids_list), default=0)
    model_max_len = getattr(encoder_tokenizer, "model_max_length", None)
    target_len = min(max_len, model_max_len)

    input_ids_enc = torch.full((B, target_len), encoder_tokenizer.pad_token_id, dtype=torch.long)
    attn_mask_enc = torch.zeros((B, target_len), dtype=torch.long)

    for i, ids in enumerate(encoder_input_ids_list):
        ids = ids[:target_len]
        if len(ids) > 0:
            input_ids_enc[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn_mask_enc[i, :len(ids)] = 1

        if len(step_index_encoder[i]) > 0:
            step_index_encoder[i] = [idx for idx in step_index_encoder[i] if idx < target_len]

    encoder_inputs = {
        "input_ids": input_ids_enc,
        "attention_mask": attn_mask_enc,
    }

    return step_index_encoder, encoder_inputs, step_texts


def get_steps_text_off(step_texts, encoder_tokenizer):
    step_index_encoder = []
    encoder_input_ids_list = []
    B = len(step_texts)

    for steps in step_texts:
        cur_ids = []
        cur_step_end = []
        cur_len = 0

        for step_text in steps:
            step_ids_enc = encoder_tokenizer(
                step_text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]

            if len(step_ids_enc) == 0:
                continue

            cur_ids.extend(step_ids_enc)
            cur_len += len(step_ids_enc)
            cur_step_end.append(cur_len - 1)

        encoder_input_ids_list.append(cur_ids)
        step_index_encoder.append(cur_step_end)

    max_len = max((len(x) for x in encoder_input_ids_list), default=0)
    model_max_len = getattr(encoder_tokenizer, "model_max_length", None)
    target_len = min(max_len, model_max_len)

    input_ids_enc = torch.full((B, target_len), encoder_tokenizer.pad_token_id, dtype=torch.long)
    attn_mask_enc = torch.zeros((B, target_len), dtype=torch.long)

    for i, ids in enumerate(encoder_input_ids_list):
        ids = ids[:target_len]
        if len(ids) > 0:
            input_ids_enc[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn_mask_enc[i, :len(ids)] = 1

        if len(step_index_encoder[i]) > 0:
            step_index_encoder[i] = [idx for idx in step_index_encoder[i] if idx < target_len]

    encoder_inputs = {
        "input_ids": input_ids_enc,
        "attention_mask": attn_mask_enc,
    }

    return step_index_encoder, encoder_inputs, step_texts  


@torch.no_grad()
def _gather_step_embs_flat_and_offsets(
    hs: torch.Tensor,
    step_index_encoder: list,
):
    N, Tenc, H = hs.shape
    chunks = []
    offsets = [0]

    for n in range(N):
        ends = step_index_encoder[n]
        if not ends:
            offsets.append(offsets[-1])
            continue

        idx = torch.as_tensor(ends, device=hs.device, dtype=torch.long)
        idx = idx[(idx >= 0) & (idx < Tenc)]
        if idx.numel() == 0:
            offsets.append(offsets[-1])
            continue

        chunks.append(hs[n].index_select(0, idx))
        offsets.append(offsets[-1] + idx.numel())

    flat = torch.cat(chunks, dim=0) if len(chunks) > 0 else hs.new_zeros((0, H))
    offsets = torch.tensor(offsets, dtype=torch.long, device="cpu")
    return flat, offsets


def build_flat_embs_and_index_off(
    hs_on: torch.Tensor,
    hs_off: torch.Tensor,
    step_index_encoder_on: list,
    step_index_encoder_off: list,
    uids: list,
    uid2offrow: dict,
):
    emb_on_t, offsets_on = _gather_step_embs_flat_and_offsets(hs_on, step_index_encoder_on)
    emb_off_t, offsets_off = _gather_step_embs_flat_and_offsets(hs_off, step_index_encoder_off)

    uid2on = defaultdict(list)
    flat2info = []

    B = len(uids)
    for b in range(B):
        st = int(offsets_on[b].item())
        ed = int(offsets_on[b+1].item())
        S = ed - st
        if S <= 0:
            continue
        flat2info.extend((b, j) for j in range(S))
        uid2on[uids[b]].extend(range(st, ed))

    uid2off = defaultdict(list)
    for uid, off_row in uid2offrow.items():
        uid_st, uid_ed = off_row
        st = int(offsets_off[uid_st].item())
        ed = int(offsets_off[uid_ed].item())
        if ed > st:
            uid2off[uid].extend(range(st, ed))

    uid2flatidx = {}
    all_uids = set(uid2on.keys()) | set(uid2off.keys())
    for uid in all_uids:
        uid2flatidx[uid] = {
            "on": uid2on.get(uid, []),
            "off": uid2off.get(uid, []),
        }

    num_on_indices = sum(len(v["on"]) for v in uid2flatidx.values())

    assert emb_on_t.shape[0] == num_on_indices, (
        f"emb_on row mismatch: emb_on={emb_on_t.shape[0]}, "
        f"sum(on_indices)={num_on_indices}"
    )
    
    num_off_indices = sum(len(v["off"]) for v in uid2flatidx.values())

    assert emb_off_t.shape[0] == num_off_indices, (
        f"emb_off row mismatch: emb_off={emb_off_t.shape[0]}, "
        f"sum(off_indices)={num_off_indices}"
    )

    emb_on = emb_on_t.detach().float().cpu().numpy().astype(np.float32, copy=False)
    emb_off = emb_off_t.detach().float().cpu().numpy().astype(np.float32, copy=False)
    return emb_on, emb_off, uid2flatidx, flat2info


@torch.no_grad()
def build_flat_embs_and_index(
    hs: torch.Tensor,
    uids_sel,
    step_index_encoder_sel,
    sample_sign_sel,
    sel2orig,
    device=None,
):
    if device is None:
        device = hs.device
    hs = hs.to(device)

    uid2flatidx = defaultdict(lambda: {"pos": [], "all": []})
    flat2info = []
    flat_emb_list = []

    for sel_b, (uid, pos_list, sign) in enumerate(zip(uids_sel, step_index_encoder_sel, sample_sign_sel)):
        orig_b = sel2orig[sel_b]

        for step_id, tok_idx in enumerate(pos_list):
            emb = hs[sel_b, tok_idx, :]
            emb = emb.detach().float().cpu().numpy()
            flat_idx = len(flat_emb_list)

            flat_emb_list.append(emb)
            uid2flatidx[uid]["all"].append(flat_idx)
            if sign == "pos":
                uid2flatidx[uid]["pos"].append(flat_idx)

            flat2info.append((orig_b, step_id, tok_idx, sign))

    emb_flat = np.stack(flat_emb_list, axis=0) if len(flat_emb_list) > 0 else np.zeros((0, hs.size(-1)), dtype=np.float32)
    return emb_flat, uid2flatidx, flat2info
