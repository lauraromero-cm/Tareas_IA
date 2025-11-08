# parallel_trainer.py

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from models import GameDataset, LogisticRegressionTorch, LinearSVMTorch


def make_dataloader(X, y, batch_size, shuffle=True):
    ds = GameDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model, optimizer, loss_fn, train_loader, device):
    """
    Entrena UNA época y devuelve (loss_promedio, acc_train)
    """
    model.train()
    total_loss = 0.0

    all_preds = []
    all_true = []

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)

        preds = torch.argmax(logits.detach(), dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(yb.detach().cpu().numpy())

    avg_loss = total_loss / len(train_loader.dataset)
    acc = accuracy_score(all_true, all_preds)
    return avg_loss, acc


def eval_accuracy(model, data_loader, device):
    """
    Accuracy en un loader dado
    """
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(yb.cpu().numpy())

    return accuracy_score(all_true, all_preds)


def run_parallel_training(
    model_type,
    configs,
    X_train,
    y_train,
    input_dim,
    num_classes,
    device="cpu"
):
    """
    model_type: "logreg" o "svm"
    configs: lista de dicts con hiperparámetros cargados del YAML

    Devuelve:
        survivors: lista con las 2 mejores configs (cada una con modelo entrenado, historia, etc.)
        global_history: estado de entrenamiento por época global
    """

    alive = []
    for cfg in configs:
        if model_type == "logreg":
            model = LogisticRegressionTorch(input_dim, num_classes).to(device)
            loss_fn = torch.nn.CrossEntropyLoss()

        elif model_type == "svm":
            model = LinearSVMTorch(input_dim, num_classes).to(device)

            margin = cfg.get("margin", 1.0)
            C = cfg.get("C", 1.0)

            def svm_loss_fn(logits, y, margin=margin, C=C, model=model):
                hinge = model.multiclass_hinge_loss(logits, y, margin=margin)
                return C * hinge

            loss_fn = svm_loss_fn

        else:
            raise ValueError("model_type inválido")

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"]
        )

        dataloader = make_dataloader(
            X_train,
            y_train,
            batch_size=cfg["batch_size"],
            shuffle=True
        )

        alive.append({
            "cfg": cfg,
            "model": model,
            "optimizer": optimizer,
            "loss_fn": loss_fn,
            "dataloader": dataloader,
            "epoch": 0,
            "history": []  # lista de dicts {epoch, loss, acc}
        })

    global_history = []
    max_max_epochs = max([c["max_epochs"] for c in configs])

    for ep in range(max_max_epochs):
        # entrenar todas las configs vivas (si no pasaron su max_epochs interno)
        for slot in alive:
            max_epochs_cfg = slot["cfg"]["max_epochs"]
            if slot["epoch"] >= max_epochs_cfg:
                continue

            loss, acc = train_one_epoch(
                slot["model"],
                slot["optimizer"],
                slot["loss_fn"],
                slot["dataloader"],
                device
            )

            slot["epoch"] += 1
            slot["history"].append({
                "epoch": slot["epoch"],
                "loss": loss,
                "acc": acc
            })

        # logueamos estado global
        snapshot = {
            "epoch_global": ep + 1,
            "alive_status": [
                {
                    "name": s["cfg"]["name"],
                    "epoch": s["epoch"],
                    "last_acc": (s["history"][-1]["acc"] if len(s["history"]) else None)
                }
                for s in alive
            ]
        }
        global_history.append(snapshot)

        # cada 5 épocas globales sacamos la peor accuracy si hay >2
        if (ep + 1) % 5 == 0 and len(alive) > 2:
            ranked = []
            for s in alive:
                if len(s["history"]) == 0:
                    score = -1.0
                else:
                    score = s["history"][-1]["acc"]
                ranked.append((s, score))

            ranked.sort(key=lambda x: x[1])  # peor primero
            worst_slot, worst_acc = ranked[0]

            print(f"[{model_type}] Eliminando config '{worst_slot['cfg']['name']}' "
                  f"en epoch_global={ep+1} (acc={worst_acc:.4f})")

            alive.remove(worst_slot)

        # si quedan 2 o menos, ya terminamos 
        if len(alive) <= 2:
            break

    # rankeamos final por última accuracy y devolvemos top2
    final_rank = []
    for s in alive:
        if len(s["history"]) == 0:
            score = -1.0
        else:
            score = s["history"][-1]["acc"]
        final_rank.append((s, score))

    final_rank.sort(key=lambda x: x[1], reverse=True)
    survivors = [r[0] for r in final_rank[:2]]

    return survivors, global_history

