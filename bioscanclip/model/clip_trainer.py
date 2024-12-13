import torch
import hydra
import lightning.pytorch as pl
from bioscanclip.model.simple_clip import load_clip_model
from bioscanclip.model.loss_func import ClipLoss, ContrastiveLoss
import torch.nn as nn


class CLIPTrainer(pl.LightningModule):

    def define_config(self):

        self.for_open_clip = False
        if hasattr(self.args.model_config, 'for_open_clip') and self.args.model_config.for_open_clip:
            self.for_open_clip = True
        self.all_gather = False
        if hasattr(self.args.model_config, 'all_gather') and self.args.model_config.all_gather:
            self.all_gather = True

        self.fix_temperature = None
        if hasattr(self.args.model_config, 'fix_temperature') and self.args.model_config.fix_temperature:
            self.fix_temperature = self.args.model_config.fix_temperature

        self.enable_amp = False
        if hasattr(self.args.model_config, 'amp') and self.args.model_config.amp:
            self.enable_amp = True

        self.eval_skip_epoch = -1
        if hasattr(self.args.model_config, 'eval_skip_epoch') and self.args.model_config.eval_skip_epoch:
            self.eval_skip_epoch = self.args.model_config.eval_skip_epoch

        self.bind_to = None
        if hasattr(self.args.model_config, 'bind_to'):
            self.bind_to = self.args.model_config.bind_to

        self.no_image_text_loss = False
        if hasattr(self.args.model_config, 'no_image_text_loss'):
            self.no_image_text_loss = self.args.model_config.no_image_text_loss

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.define_config()

        # initialize the network
        self.model = load_clip_model(self.args)
        if self.all_gather:

            criterion = ClipLoss(local_loss=args.model_config.loss_setup.local_loss,
                                     gather_with_grad=args.model_config.loss_setup.gather_with_grad,
                                     use_horovod=args.model_config.loss_setup.use_horovod,
                                     criterion=nn.CrossEntropyLoss(), bind_to=self.bind_to,
                                     no_image_text_loss=self.no_image_text_loss)
        else:
            criterion = ContrastiveLoss(criterion=nn.CrossEntropyLoss(), logit_scale=1 / 0.07)

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

        return loss
