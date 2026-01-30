from torch.utils.data import Dataset, DataLoader

class TensorDataset(Dataset):
    """
    너가 원하는 형태:
    - case A: input_ids (B,L) + attention_mask (B,L)
    - case B: inputs_embeds (B,L,H) + attention_mask (B,L)
    """
    def __init__(self, x: torch.Tensor, attention_mask: torch.Tensor, y: torch.Tensor, use_embeds: bool):
        self.x = x
        self.attention_mask = attention_mask
        self.y = y
        self.use_embeds = use_embeds

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        item = {
            "attention_mask": self.attention_mask[idx],
            "labels": self.y[idx],
        }
        if self.use_embeds:
            item["inputs_embeds"] = self.x[idx]
        else:
            item["input_ids"] = self.x[idx]
        return item


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total += loss.item()
    return total / max(1, len(loader))


@torch.no_grad()
def eval_loop(model, loader, device):
    model.eval()
    total = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total += out["loss"].item()
    return total / max(1, len(loader))


def main_train():
    cfg = DownstreamConfig(
        model_id="meta-llama/Meta-Llama-3.1-8B",
        task="classification",
        num_labels=2,
        pooling="last",
        dora_r=8,
        dora_alpha=16,
    )

    model = LlamaForDownstream(cfg)
    device = next(model.parameters()).device

    # -----------------------
    # ✅ 여기에 네 데이터 텐서 넣기
    # 예시 shape:
    # input_ids: (B,L) long
    # inputs_embeds: (B,L,H) float
    # attention_mask: (B,L) long/bool
    # labels: (B,) long (cls) or float (reg)
    # -----------------------
    B, L, H = 8, 128, model.head.in_features
    use_embeds = False  # True면 inputs_embeds 경로 사용

    if use_embeds:
        x = torch.randn(B, L, H)
    else:
        x = torch.randint(0, 32000, (B, L))

    attention_mask = torch.ones(B, L)
    labels = torch.randint(0, cfg.num_labels, (B,))

    ds = TensorDataset(x, attention_mask, labels, use_embeds=use_embeds)
    dl = DataLoader(ds, batch_size=2, shuffle=True)

    # ✅ optimizer는 trainable 파라미터(=DoRA + head)만
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=2e-4)

    for epoch in range(3):
        tr = train_one_epoch(model, dl, optimizer, device)
        ev = eval_loop(model, dl, device)
        print(f"epoch={epoch} train_loss={tr:.4f} eval_loss={ev:.4f}")

    return model


if __name__ == "__main__":
    main_train()