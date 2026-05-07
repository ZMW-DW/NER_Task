import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from baseline_method import (
    DATASETS_DIR,
    BaseLineDataset,
    collate_fn_factory,
    BaseLineModel,
    BaseLineModelConfig,
    id2entropy
)

def main():

    # =========================
    # config
    # =========================
    config = BaseLineModelConfig(
        is_train=False,
        batch_size=16
    )

    # =========================
    # tokenizer
    # =========================
    tokenizer = BertTokenizer.from_pretrained(
        str(config.backbone_model)
    )

    # =========================
    # dataset
    # =========================
    test_path = DATASETS_DIR / "test.json"

    test_loader = DataLoader(
        BaseLineDataset(test_path),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn_factory(tokenizer)
    )

    # =========================
    # model
    # =========================
    model = BaseLineModel(config)

    ckpt = "checkpoints/your_model.bin"
    model.load_state_dict(
        torch.load(ckpt, map_location=config.device)
    )

    model.to(config.device)
    model.eval()

    # =========================
    # evaluation cache
    # =========================
    all_preds = []
    all_labels = []

    # =========================
    # evaluation
    # =========================
    with torch.no_grad():

        pbar = tqdm(test_loader)

        for batch in pbar:

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )

            logits = outputs.logits

            pred_ids = torch.argmax(logits, dim=-1)

            labels = batch["labels"]

            # ========= 对齐 =========
            for pred_seq, label_seq in zip(
                pred_ids.cpu().tolist(),
                labels.cpu().tolist()
            ):

                true_labels = []
                pred_labels = []

                for pred_id, label_id in zip(pred_seq, label_seq):

                    # 忽略 CLS/SEP/PAD/subword
                    if label_id == -100:
                        continue

                    true_labels.append(id2entropy[label_id])
                    pred_labels.append(id2entropy[pred_id])

                all_labels.append(true_labels)
                all_preds.append(pred_labels)

    # =========================
    # metrics
    # =========================
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print("\n========== Evaluation ==========")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\n========== Detail Report ==========")
    print(classification_report(all_labels, all_preds))


if __name__ == "__main__":
    main()