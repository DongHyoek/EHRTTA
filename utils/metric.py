import evaluate
import torch

class MetricManager:
    """
    Usage per epoch:
      mm = MetricManager(MetricsConfig(task="classification"))
      mm.reset()
      mm.update_classification(logits, labels)  # OR mm.update_regression(preds, labels)
      results = mm.compute()
    """
    def __init__(self, args):
        
        self.args = args

        if args.task == "classification":
            self.acc = evaluate.load("accuracy")
            self.f1 = evaluate.load("f1")
            self.prec = evaluate.load("precision")
            self.rec = evaluate.load("recall")
            self.auroc = evaluate.load("roc_auc")
            self.cm = evaluate.load("confusion_matrix")
            # AUPRC 직접 계산용 버퍼
            self._all_pos_scores = []
            self._all_labels = []

        elif args.task == "regression":
            self.mse = evaluate.load("mse")
            self.mae = evaluate.load("mae")
            # MAPE는 evaluate에 없거나 환경에 따라 다를 수 있어 직접 구현(안전)
            self._mape_sum = 0.0
            self._mape_count = 0
        else:
            raise ValueError("task must be 'classification' or 'regression'")

    def reset(self):
        if self.args.task == "classification":
            self.acc = evaluate.load("accuracy")
            self.f1 = evaluate.load("f1")
            self.prec = evaluate.load("precision")
            self.rec = evaluate.load("recall")
            self.auroc = evaluate.load("roc_auc")
            self.cm = evaluate.load("confusion_matrix")
            # AUPRC 직접 계산용 버퍼
            self._all_pos_scores = []
            self._all_labels = []

        elif self.args.task == "regression":
            self.mse = evaluate.load("mse")
            self.mae = evaluate.load("mae")
            # MAPE는 evaluate에 없거나 환경에 따라 다를 수 있어 직접 구현(안전)
            self._mape_sum = 0.0
            self._mape_count = 0

    @torch.no_grad()
    def update_classification(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        logits: (B, C) 
        labels: (B,)
        """
        logits = logits.detach().cpu()
        labels = labels.detach().cpu()

        preds = logits.argmax(dim=-1)  # (B,)
        probs = torch.softmax(logits, dim=-1)  # (B,C)
        pos_scores = probs[:, 1]  # (B,)

        # label-based
        self.acc.add_batch(predictions=preds, references=labels)
        self.f1.add_batch(predictions=preds, references=labels)
        self.prec.add_batch(predictions=preds, references=labels)
        self.rec.add_batch(predictions=preds, references=labels)

        # score-based (binary)
        self.auroc.add_batch(prediction_scores=pos_scores, references=labels)
        # AUPRC용 누적
        self._all_pos_scores.append(pos_scores)
        self._all_labels.append(labels)

        # confusion matrix
        # evaluate의 confusion_matrix는 label 리스트가 있으면 더 안전
        self.cm.add_batch(predictions=preds, references=labels)

    @torch.no_grad()
    def update_regression(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        preds: (B,) or (B,1)
        labels: (B,) or (B,1)
        """
        preds = preds.detach().cpu().view(-1)
        labels = labels.detach().cpu().view(-1)

        self.mse.add_batch(predictions=preds, references=labels)
        self.mae.add_batch(predictions=preds, references=labels)

        # MAPE (percent)
        denom = labels.abs().clamp_min(self.args.mape_eps)
        mape_batch = (preds - labels).abs() / denom  # (B,)
        self._mape_sum += float(mape_batch.sum().item())
        self._mape_count += int(mape_batch.numel())

    def _average_precision_binary(self, scores: torch.Tensor, labels: torch.Tensor) -> float:
        """
        scores: (N,) probability or score for positive class
        labels: (N,) in {0,1}
        """
        scores = scores.float()
        labels = labels.long()

        # positives count
        P = int(labels.sum().item())
        if P == 0:
            return float("nan")  # or 0.0 (원하는 정의에 맞게)
        
        # sort by score descending
        idx = torch.argsort(scores, descending=True)
        y = labels[idx]

        # precision@k
        tp = torch.cumsum(y, dim=0)
        k = torch.arange(1, y.numel() + 1, device=y.device)
        precision_at_k = tp / k

        # AP = mean precision at each true positive
        ap = (precision_at_k * y).sum().item() / P
        return float(ap)

    def compute(self):
        if self.args.task == "classification":
            # average 옵션 반영 (binary/macro 등)
            f1 = self.f1.compute(average='macro')["f1"]
            precision = self.prec.compute(average='macro')["precision"]
            recall = self.rec.compute(average='macro')["recall"]

            scores = torch.cat(self._all_pos_scores, dim=0) if len(self._all_pos_scores) else torch.tensor([])
            labels = torch.cat(self._all_labels, dim=0) if len(self._all_labels) else torch.tensor([])
            
            auprc = self._average_precision_binary(scores, labels) if scores.numel() else float("nan")

            out = {
                "accuracy": self.acc.compute()["accuracy"],
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "auroc": self.auroc.compute()["roc_auc"],
                "auprc": auprc,
                "confusion_matrix": self.cm.compute()["confusion_matrix"],
            }
            return out

        else:
            mape = (self._mape_sum / max(self._mape_count, 1)) * 100.0
            return {
                "mse": self.mse.compute()["mse"],
                "mae": self.mae.compute()["mae"],
                "mape": mape,
            }