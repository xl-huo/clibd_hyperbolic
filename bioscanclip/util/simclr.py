# modified from https://github.com/sthalles/SimCLR/blob/master/simclr.py
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb
import os
import shutil

import torch
import yaml

torch.manual_seed(42)


def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    ckpt_dir = os.path.join(args.project_root_path, "ckpt", "uni_model", "uni_model", "image",
                            args.model_config.model_output_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(state, os.path.join(ckpt_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(ckpt_dir, filename), 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.device = kwargs['device']
        self.model = kwargs['model'].to(self.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        log_dir = os.path.join(self.args.project_root_path, 'logs_for_SimCLR_training')
        os.makedirs(log_dir, exist_ok=True)

        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def info_nce_loss(self, features):

        labels = torch.cat(
            [torch.arange(self.args.model_config.batch_size) for i in range(self.args.model_config.n_views)], dim=0)

        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.args.model_config.temperature
        return logits, labels

    def train(self, train_loader):
        scaler = GradScaler(enabled=True)

        wandb.init(project="CLIBD-simclr", name=self.args.model_config.model_output_name)

        n_iter = 0
        print(f"Start SimCLR training for {self.args.model_config.epochs} epochs.")

        for epoch_counter in range(self.args.model_config.epochs):
            pbar = tqdm(train_loader, total=len(train_loader))
            epoch_loss = []
            for images_1, images_2 in pbar:
                images = torch.cat([images_1, images_2], dim=0)

                images = images.to(self.device)

                with autocast(enabled=True):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                epoch_loss.append(loss.item())

                if n_iter % self.args.model_config.log_every_n_steps == 0:
                    top1, _ = accuracy(logits, labels, topk=(1, 5))
                    wandb.log({"loss": loss, "acc/top1": top1[0], "learning_rate": self.scheduler.get_lr()[0],
                               "n_iter": n_iter})

                n_iter += 1

                pbar.set_description(f"Epoch: {epoch_counter}. Loss: {loss:.4f}.")

            epoch_loss_avg = sum(epoch_loss) / len(epoch_loss)
            wandb.log({"epoch_loss": epoch_loss_avg, "epoch": epoch_counter})

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            print(f"Epoch: {epoch_counter}\tLoss: {epoch_loss_avg:.4f}\tTop1 accuracy: {top1[0]}")

        print("Training has finished.")
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.model_config.epochs)
        save_checkpoint(
            self.args,
            {
                'epoch': self.args.model_config.epochs,
                'arch': self.args.model_config.arch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, is_best=False, filename=os.path.join(wandb.run.dir, checkpoint_name))
        print(f"Model checkpoint and metadata has been saved at {wandb.run.dir}.")

