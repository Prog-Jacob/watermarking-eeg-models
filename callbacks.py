from pytorch_lightning.callbacks import Callback


class MultiplyLRScheduler(Callback):
    def __init__(self, x, n, epsilon):
        super().__init__()
        self.x = x
        self.n = n
        self.epsilon = epsilon

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch % self.n == 0 and trainer.current_epoch > 0:
            optimizer = trainer.optimizers[0]

            for param_group in optimizer.param_groups:
                new_lr = param_group["lr"] * self.x

                if new_lr < self.epsilon:
                    new_lr = self.epsilon
                elif new_lr > 1:
                    new_lr = 1

                param_group["lr"] = new_lr
