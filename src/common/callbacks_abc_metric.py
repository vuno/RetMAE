from abc import abstractclassmethod

from lightning.pytorch import Callback


class BaseMetricCallback(Callback):
    def __init__(self, metric_train=True, metric_val=True, metric_test=True) -> None:
        super().__init__()
        
        self.metric_train = metric_train
        self.metric_val = metric_val
        self.metric_test = metric_test
    
    ################################################################################################
    def on_fit_start(self, trainer, pl_module) -> None:
        if self.metric_train:
            self.metric_train.to(pl_module.device)
        if self.metric_val:
            self.metric_val.to(pl_module.device)
            
    def on_test_start(self, trainer, pl_module) -> None:
        if self.metric_test:
            self.metric_test.to(pl_module.device)
        
    ################################################################################################
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        if self.metric_train:
            self.update_metrics(outputs['preds'], outputs['target'], split='train')
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        if self.metric_val:
            self.update_metrics(outputs['preds'], outputs['target'], split='val')
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        if self.metric_test:
            self.update_metrics(outputs['preds'], outputs['target'], split='test')
    
    ################################################################################################
    
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self.metric_train:
            self.compute_metrics(pl_module, 'train')
    
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self.metric_val:
            self.compute_metrics(pl_module, 'val')
    
    def on_test_epoch_end(self, trainer, pl_module) -> None:
        if self.metric_test:
            self.compute_metrics(pl_module, 'test')
    
    ################################################################################################
    
    @abstractclassmethod
    def update_metrics(self, preds, target, split):
        pass
    
    @abstractclassmethod
    def compute_metrics(self, pl_module, split):
        pass