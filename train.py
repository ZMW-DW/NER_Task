import os
import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from baseline_method import (
    DATASETS_DIR,
    BaseLineDataset,
    collate_fn_factory,
    BaseLineModelConfig,
    BaseLineModel
)

def main():
    config = BaseLineModelConfig(is_train=True, batch_size=2)
    model: BaseLineModel = BaseLineModel(config=config)
    model.to(config.device)
    model.train()
    os.makedirs("checkpoints", exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(str(config.backbone_model))
    train_path = DATASETS_DIR / "train.json"

    collate_fn = collate_fn_factory(tokenizer)

    train_dataloader = DataLoader(
        BaseLineDataset(train_path),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = AdamW(model.parameters(), lr=config.lr)

    for epoch in range(config.epochs):
        total_train_loss = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1} / {config.epochs} [Train]")

        for batch_data in pbar:
            outputs = model(
                input_ids=batch_data['input_ids'],
                attention_mask=batch_data['attention_mask'],
                labels=batch_data['labels']
            )

            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()
            pbar.set_postfix_str(f"current loss: {loss.item()}")
        
        avg_loss = total_train_loss / len(train_dataloader)
        print(f"The average loss is {avg_loss}\n")
        save_path = f"checkpoints/BaselineModel_{avg_loss:.3f}.bin"
        torch.save(model.state_dict(), save_path)
        print(f"Model had saved to {save_path}\n")

if __name__ == "__main__":
    main()
        