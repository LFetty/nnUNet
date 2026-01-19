import torch

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerRegression_mae_deep import nnUNetTrainerRegression_mae_deep
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler

class nnUNetTrainerRegression_20epochs(nnUNetTrainerRegression_mae_deep):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50
        self.initial_lr = 1e-3


class nnUNetTrainerRegression_encoder_freeze(nnUNetTrainerRegression_20epochs):
    def configure_optimizers(self):
        # Freeze encoder layers
        for param in self.network.encoder.parameters():
            param.requires_grad = False

        # Only pass trainable parameters to optimizer
        trainable_params = [p for p in self.network.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(trainable_params, self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler
