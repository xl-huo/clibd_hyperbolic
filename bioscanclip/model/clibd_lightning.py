import torch
import hydra
import lightning.pytorch as pl
from bioscanclip.model.simple_clip import load_clip_model
from bioscanclip.model.loss_func import ClipLoss, ContrastiveLoss
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from bioscanclip.util.util import scale_learning_rate, eval_phase, convert_acc_dict_to_wandb_dict, compute_overall_acc
import torch.distributed as dist
import os
import datetime
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.callbacks import ProgressBarBase


class MyProgressBar(ProgressBarBase):
    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items['loss'] = trainer.logged_metrics.get('train_loss', 0)
        return items


class CLIBDLightning(pl.LightningModule):

    def __init__(self, args, len_train_dataloader=None, all_keys_dataloader=None, seen_val_dataloader=None,
                 unseen_val_dataloader=None, k_list=None):
        super().__init__()
        self.args = args
        self.define_config()
        self.length_of_pre_train_dataloader = len_train_dataloader
        # initialize the network
        self.model = load_clip_model(self.args)
        self.setup_criterion()

        self.best_epoch = None
        self.best_overall_acc = None

        # Save dataloaders
        self.all_keys_dataloader = all_keys_dataloader
        self.seen_val_dataloader = seen_val_dataloader
        self.unseen_val_dataloader = unseen_val_dataloader

    def setup_criterion(self):
        if self.all_gather:
            self.criterion = ClipLoss(local_loss=self.args.model_config.loss_setup.local_loss,
                                      gather_with_grad=self.args.model_config.loss_setup.gather_with_grad,
                                      use_horovod=self.args.model_config.loss_setup.use_horovod,
                                      criterion=nn.CrossEntropyLoss(), bind_to=self.bind_to,
                                      no_image_text_loss=self.no_image_text_loss)
        else:
            self.criterion = ContrastiveLoss(criterion=nn.CrossEntropyLoss(), logit_scale=1 / 0.07)

    def define_config(self):
        self.for_open_clip = getattr(self.args.model_config, 'for_open_clip', False)
        self.all_gather = getattr(self.args.model_config, 'all_gather', False)
        self.fix_temperature = getattr(self.args.model_config, 'fix_temperature', None)
        self.enable_amp = getattr(self.args.model_config, 'amp', False)
        self.eval_skip_epoch = getattr(self.args.model_config, 'eval_skip_epoch', -1)
        self.bind_to = getattr(self.args.model_config, 'bind_to', None)
        self.batch_size = getattr(self.args.model_config, 'batch_size', 500)
        self.no_image_text_loss = getattr(self.args.model_config, 'no_image_text_loss', False)

        self.lr = getattr(self.args.model_config.lr_config, 'lr', 0.001)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.lr = scale_learning_rate(lr=self.lr, batch_size=world_size)
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H%M%S")

        self.folder_path = os.path.join(self.args.project_root_path, self.args.model_output_dir,
                                        self.args.model_config.model_output_name, formatted_datetime)
        os.makedirs(self.folder_path, exist_ok=True)

        OmegaConf.save(self.args, os.path.join(self.folder_path, 'config.yaml'))

    def forward(self, image_input_batch, dna_input_batch, language_input):
        return self.model(image_input_batch, dna_input_batch, language_input)

    def training_step(self, batch, batch_idx):
        processid_batch, image_input_batch, dna_input_batch, input_ids, token_type_ids, attention_mask, label_for_train_batch = batch
        if self.for_open_clip:
            language_input = input_ids
        else:
            language_input = {'input_ids': input_ids, 'token_type_ids': token_type_ids,
                              'attention_mask': attention_mask}

        image_output, dna_output, language_output, logit_scale, logit_bias = self.model(image_input_batch,
                                                                                        dna_input_batch,
                                                                                        language_input)

        if self.fix_temperature is not None:
            logit_scale = 1 / 0.07
        loss = self.criterion(image_features=image_output, dna_features=dna_output, text_features=language_output,
                              labels=label_for_train_batch, logit_scale=logit_scale)

        self.log('train_loss', loss)

        return loss

    def on_train_epoch_end(self):
        device = next(self.model.parameters()).device

        acc_dict, pred_dict = eval_phase(
            self.model, device,
            self.all_keys_dataloader,
            self.seen_val_dataloader,
            self.unseen_val_dataloader,
            self.k_list,
            self.args
        )
        dict_for_wandb = convert_acc_dict_to_wandb_dict(acc_dict)
        dict_for_wandb['epoch'] = self.current_epoch
        overall_acc = compute_overall_acc(acc_dict)
        if self.best_overall_acc is None or self.best_overall_acc < overall_acc:
            self.best_epoch = self.current_epoch
            self.best_overall_acc = overall_acc
            if self.args.save_ckpt:
                best_ckpt_path = os.path.join(self.folder_path, f'best.pth')
                torch.save(self.model.state_dict(), best_ckpt_path)
        dict_for_wandb["overall_acc"] = overall_acc
        dict_for_wandb["best_epoch"] = self.current_epoch

        self.log_dict(dict_for_wandb)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        total_steps = self.length_of_pre_train_dataloader * self.args.model_config.trainer.max_epochs
        scheduler = None
        if hasattr(self.args.model_config, 'lr_scheduler'):
            if self.args.model_config.lr_scheduler == 'one_cycle':
                max_lr = 0.001
                if hasattr(self.args.model_config.lr_config, 'max_lr'):
                    max_lr = self.args.model_config.lr_config.max_lr
                max_lr = scale_learning_rate(lr=max_lr, batch_size=self.args.model_config.batch_size)
                scheduler = lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=max_lr,
                    total_steps=total_steps,
                    pct_start=0.3,
                    anneal_strategy='cos',
                    cycle_momentum=False,
                )
            elif self.args.model_config.lr_scheduler == 'exponential':
                scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
            elif self.args.model_config.lr_scheduler == 'step':
                scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            elif self.args.model_config.lr_scheduler == 'cosine':
                min_lr = 1e-9
                if hasattr(self.args.model_config.lr_config, 'min_lr'):
                    min_lr = self.args.model_config.lr_config.min_lr
                min_lr = scale_learning_rate(lr=min_lr, batch_size=self.args.model_config.batch_size)
                scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)
        return [optimizer], [scheduler]

    def load_from_checkpoint_with_path(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
        self.load_state_dict(checkpoint)
        return self.model
