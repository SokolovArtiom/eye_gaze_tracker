import os

import timm
import torch
import tqdm
from src.Dataset import EyeGazeDataset

torch.backends.cudnn.benchmark = True


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.out_dim = cfg["out_dims"]
        self.device = cfg["device"]

        os.makedirs(self.cfg["output_directory"], exist_ok=True)

        self.cur_train_loss = 0
        self.cur_val_loss = 0

    def train(self):
        model = self._get_model(self.cfg["model_name"]).to(self.cfg["device"])
        optimizer = self._get_optimizer(
            model,
            self.cfg["optimizer"],
            self.cfg["lr"],
            self.cfg["weight_decay"],
        )
        scheduler = self._get_scheduler(
            optimizer,
            self.cfg["scheduler"],
            self.cfg["iters"],
            self.cfg["warmup"],
            self.cfg["cycles"],
            self.cfg["milestones"],
            self.cfg["sch_gamma"],
            self.cfg["patience"],
        )
        criterion = self._get_criterion(self.cfg["criterion"])

        train_dl, val_dl = self._get_dataloader(
            self.cfg["batch_size"], self.cfg["num_workers"]
        )

        os.makedirs(
            os.path.join(
                self.cfg["output_directory"],
                "{}/".format(self.cfg["model_name"]),
            ),
            exist_ok=True,
        )

        best_val_loss = float("inf")
        best_epoch = 0

        for epoch in range(1, self.cfg["epochs"] + 1):
            print("Epoch : {}".format(epoch))
            self._train_one_epoch(
                epoch, model, train_dl, optimizer, criterion, scheduler
            )

            self._validate(epoch, model, val_dl, criterion, scheduler)

            if self.cur_val_loss < best_val_loss:
                torch.save(
                    model.state_dict(),
                    "{}/{}/best.pth".format(
                        self.cfg["output_directory"], self.cfg["model_name"]
                    ),
                )

                best_val_loss = self.cur_val_loss
                best_epoch = epoch

        print(
            f"Training finished \n Best model's MSE = {best_val_loss} on epoch {best_epoch}."
        )

    def _train_one_epoch(self, epoch, model, train_dl, optimizer, criterion, scheduler):
        model.train()

        train_loss = 0.0

        for step, (data, labels) in enumerate(tqdm.tqdm(train_dl)):
            optimizer.zero_grad()

            data = data.to(self.device)
            labels = torch.squeeze(labels).to(self.device)
            predictions = torch.tanh(model(data))
            loss = criterion(predictions, labels)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                train_loss += loss.item()

        self.cur_train_loss = train_loss / len(train_dl)
        print(len(train_dl))
        print("train MSE: ", self.cur_train_loss)

    def _validate(self, epoch, model, val_dl, criterion, scheduler):
        model.eval()
        val_loss = 0

        for step, (data, labels) in enumerate(tqdm.tqdm(val_dl)):
            with torch.no_grad():
                data = data.to(self.device)
                labels = torch.squeeze(labels).to(self.device)
                predictions = torch.tanh(model(data))
                loss = criterion(predictions, labels)

                val_loss += loss.item()

        self.cur_val_loss = val_loss / len(val_dl)
        print("val MSE: ", self.cur_val_loss)

        if self.cfg["scheduler"] == "ReduceLROnPlateau":
            scheduler.step(self.cur_val_loss)

    def _get_model(self, model_name):
        print("Loading {}...".format(model_name))

        if model_name == "MobileNetV3":
            model = timm.create_model(
                "mobilenetv3_small_100", pretrained=True, num_classes=2
            )
        else:
            raise ValueError("{} model isn't supported by now".format(model_name))

        return model

    def _get_optimizer(self, model, optimizer_name, lr, weight_decay):
        if optimizer_name == "Adam":
            return torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "AdamW":
            return torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "SGD":
            return torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(
                "`{}` optimizer isn't supported by now".format(optimizer_name)
            )

    def _get_criterion(self, criterion_name):
        if criterion_name == "MSE":
            return torch.nn.MSELoss()
        else:
            raise ValueError("{} isn't supported yet".format(criterion_name))

    def _get_scheduler(
        self,
        optimizer,
        scheduler,
        iters,
        warmup,
        cycles,
        milestones,
        sch_gamma,
        patience,
    ):
        if scheduler == "ReduceLROnPlateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=patience
            )
        elif scheduler == "MultiStep":
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones, gamma=sch_gamma
            )
        elif scheduler == "None":
            return None

    def _get_dataloader(self, batch_size, num_workers):
        train_ds = EyeGazeDataset(self.cfg["data_path"], is_train=True)
        val_ds = EyeGazeDataset(self.cfg["data_path"], is_train=False)

        train_dl = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
        )

        val_dl = torch.utils.data.DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

        return train_dl, val_dl
