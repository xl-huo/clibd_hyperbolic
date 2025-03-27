from tqdm import tqdm
import wandb
import psutil
import torch.distributed.nn
import torch.distributed as dist
from torch.cuda.amp import autocast


def train_epoch(activate_wandb, total_epochs, epoch, dataloader, model, optimizer, criterion, device, scaler, scheduler=None,
                for_open_clip=False, rank=None, fix_temperature=None, enable_autocast=False, train_hyperbolic=False):
    torch.autograd.set_detect_anomaly(True)
    if rank == 0:
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
    else:
        pbar = enumerate(dataloader)
    epoch_loss = 0.0
    total_step = len(dataloader)

    model.train()
    stop_flag = False
    for step, batch in pbar:
        processid_batch, image_input_batch, dna_input_batch, input_ids, token_type_ids, attention_mask, label_for_train_batch = batch
        if for_open_clip:
            language_input = input_ids
        else:
            language_input = {'input_ids': input_ids.to(device), 'token_type_ids': token_type_ids.to(device),
                              'attention_mask': attention_mask.to(device)}
        optimizer.zero_grad()
        image_input_batch = image_input_batch.to(device)
        dna_input_batch = dna_input_batch.to(device)

        if enable_autocast:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                image_output, dna_output, language_output, logit_scale, logit_bias_or_curv =\
                    model(image_input_batch, dna_input_batch, language_input)
        else:
            image_output, dna_output, language_output, logit_scale, logit_bias_or_curv =\
                model(image_input_batch, dna_input_batch, language_input)


        label_for_train_batch = label_for_train_batch.to(device)
        if fix_temperature is not None:
            logit_scale = 1 / 0.07

        if train_hyperbolic:
            losses = criterion(image_features=image_output, dna_features=dna_output, text_features=language_output,
                            labels=label_for_train_batch, logit_scale=logit_scale, curv=logit_bias_or_curv)
            loss = losses["loss"]
        else:
            loss = criterion(image_features=image_output, dna_features=dna_output, text_features=language_output,
                            labels=label_for_train_batch, logit_scale=logit_scale)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        epoch_loss = epoch_loss + loss.item()

        allocated_memory = torch.cuda.memory_allocated()
        memory_total = torch.cuda.get_device_properties(device).total_memory
        cached_memory = torch.cuda.memory_reserved(device)

        total_used_memory = allocated_memory + cached_memory

        current_lr = optimizer.param_groups[0]['lr']

        if rank == 0:
            if train_hyperbolic:
                pbar.set_description(
                    f'Epoch: {epoch} '
                    f'|| Step: {step}/{total_step} '
                    f'|| Loss: {loss.item()} '
                    f'|| Contrastive Loss: {losses["contrastive_loss"].item()} '
                    f'|| Entailment Loss: {losses["entailment_loss"].item()} '
                    f'|| Total Used CUDA Memory: {total_used_memory / (1024 ** 3):.2f} GB '
                    f'|| Total CUDA Memory: {memory_total / (1024 ** 3):.2f} GB '
                    f'|| Current LR: {current_lr} '
                    f'|| Curvature: {logit_bias_or_curv.item()}')
                
                if activate_wandb:
                    wandb.log({"loss": loss.item(),
                               "contrastive_loss": losses["contrastive_loss"].item(),
                               "entailment_loss": losses["entailment_loss"].item(),
                               "step": step + epoch * len(dataloader), 
                               "learning_rate": current_lr, "curvature": logit_bias_or_curv.item()})

            else:
                pbar.set_description(
                    f'Epoch: {epoch}||Step: {step}/{total_step}||Loss: {loss.item()} || Total Used CUDA Memory: {total_used_memory / (1024 ** 3):.2f} GB || Total CUDA Memory: {memory_total / (1024 ** 3):.2f} GB || Current LR: {current_lr}')

                if activate_wandb:
                    wandb.log({"loss": loss.item(), "step": step + epoch * len(dataloader), "learning_rate": current_lr})

    print(f'Epoch [{epoch}/{total_epochs}], Loss: {epoch_loss / len(dataloader)}')