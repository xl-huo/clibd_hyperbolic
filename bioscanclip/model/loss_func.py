import torch
import torch.nn as nn
from torch.nn import functional as F

from bioscanclip.util import lorentz as L

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def construct_label_metrix(labels):
    matrix = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    return matrix


class ContrastiveLoss(nn.Module):
    def __init__(self, criterion, logit_scale, local_loss=False, gather_with_grad=False, rank=0, world_size=1,
                 use_horovod=False):
        super(ContrastiveLoss, self).__init__()
        self.criterion = criterion
        self.logit_scale = logit_scale
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.criterion = criterion

        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, dna_features, text_features, labels, logit_scale):
        feature_list = [image_features, dna_features, text_features]
        feature_list = [item for item in feature_list if item is not None]
        label = construct_label_metrix(labels).to(labels.device)

        if len(feature_list) < 2:
            raise ValueError("Too less element for calculating the contrastive loss.")

        loss_list = []

        for idx_a, feature_a in enumerate(feature_list):
            for idx_b, feature_b in enumerate(feature_list):
                if idx_a == idx_b:
                    continue
                feature_a = F.normalize(feature_a, p=2, dim=1)
                feature_b = F.normalize(feature_b, p=2, dim=1)

                if logit_scale is not None:
                    sim_a_b = logit_scale * feature_a @ feature_b.T
                    sim_b_a = logit_scale * feature_b @ feature_a.T
                else:
                    sim_a_b = self.logit_scale * feature_a @ feature_b.T
                    sim_b_a = self.logit_scale * feature_b @ feature_a.T

                loss_a_b = self.criterion(sim_a_b, label)
                loss_b_a = self.criterion(sim_b_a, label)
                loss_list.append(loss_a_b)
                loss_list.append(loss_b_a)
        return sum(loss_list) * 1.0 / len(loss_list)


# Copied from the official OpenCLIP implementation
def gather_features(
        features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_features = hvd.allgather(features)
        else:
            with torch.no_grad():
                all_features = hvd.allgather(features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_features = list(all_features.chunk(world_size, dim=0))
                gathered_features[rank] = features
                all_features = torch.cat(gathered_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_features = torch.cat(torch.distributed.nn.all_gather(features), dim=0)
        else:
            gathered_features = [torch.zeros_like(features) for _ in range(world_size)]
            dist.all_gather(gathered_features, features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_features[rank] = features
            all_features = torch.cat(gathered_features, dim=0)

    return all_features


# Modified from the official OpenCLIP implementation
class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            criterion=nn.CrossEntropyLoss(),
            bind_to=None,
            no_image_text_loss=False
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.criterion = criterion

        # cache state
        self.prev_num_logits = 0
        self.labels = {}
        self.bind_to = bind_to
        self.no_image_text_loss = no_image_text_loss

    def forward(self, image_features, dna_features, text_features, labels, logit_scale, output_dict=False):
        device = image_features.device
        all_image_features = image_features
        all_dna_features = dna_features
        all_text_features = text_features
        all_labels = torch.cat(torch.distributed.nn.all_gather(labels), dim=0)
        all_labels = construct_label_metrix(all_labels).to(device)
        if self.world_size > 1:
            if image_features is not None:
                all_image_features = gather_features(
                    image_features,
                    self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            if dna_features is not None:
                all_dna_features = gather_features(
                    dna_features,
                    self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            if text_features is not None:
                all_text_features = gather_features(
                    text_features,
                    self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

        feature_list = [all_image_features, all_dna_features, all_text_features]
        feature_list = [item for item in feature_list if item is not None]

        if len(feature_list) < 2:
            raise ValueError("Too less element for calculating the contrastive loss.")

        loss_list = []
        bind_to_idx = None
        if self.bind_to is not None:
            if self.bind_to == "image":
                bind_to_idx = 0
            elif self.bind_to == "dna":
                bind_to_idx = 1
            elif self.bind_to == "text":
                bind_to_idx = 2


        for idx_a, feature_a in enumerate(feature_list):
            for idx_b, feature_b in enumerate(feature_list):
                if bind_to_idx is not None:
                    if idx_a != bind_to_idx and idx_b != bind_to_idx:
                        continue
                if idx_a == idx_b:
                    continue

                if self.no_image_text_loss and (idx_a == 0 or idx_b == 0) and (idx_a == 2 or idx_b == 2):
                    continue
                feature_a = F.normalize(feature_a, p=2, dim=1)
                feature_b = F.normalize(feature_b, p=2, dim=1)

                sim_a_b = logit_scale * feature_a @ feature_b.T
                sim_b_a = logit_scale * feature_b @ feature_a.T

                # sim_a_b = feature_a @ feature_b.T
                # sim_b_a = feature_b @ feature_a.T

                loss_a_b = self.criterion(sim_a_b, all_labels)
                loss_b_a = self.criterion(sim_b_a, all_labels)
                loss_list.append(loss_a_b)
                loss_list.append(loss_b_a)

        total_loss = sum(loss_list) * 1.0 / len(loss_list)
        return {"contrastive_loss": total_loss} if output_dict else total_loss


class ClipLoss_hyperbolic(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            criterion=nn.CrossEntropyLoss(),
            bind_to=None,
            no_image_text_loss=False,
            entail_weight=0.2,
            loss_type=None,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.criterion = criterion

        # cache state
        self.prev_num_logits = 0
        self.labels = {}
        self.bind_to = bind_to
        self.no_image_text_loss = no_image_text_loss

        self.entail_weight = entail_weight
        self.loss_type = loss_type

    def compute_entailment_loss(self, input_feature_a, input_feature_b, curv, exp=1e-6):
        _angle = L.oxy_angle(input_feature_a, input_feature_b, curv)
        _aperture = L.half_aperture(input_feature_a, curv, eps=exp)
        entailment_loss = torch.clamp(_angle - _aperture, min=0).mean()

        return entailment_loss


    def forward(self, image_features, dna_features, text_features, labels, logit_scale, curv):
        device = image_features.device

        # autocast to force a higher floating point precision.
        with torch.autocast(device.type, dtype=torch.float32):

            all_labels = torch.cat(torch.distributed.nn.all_gather(labels), dim=0)
            all_labels = construct_label_metrix(all_labels).to(device)
            if self.world_size > 1:
                if image_features is not None:
                    all_image_features = gather_features(
                        image_features,
                        self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
                else:
                    all_image_features = None

                if dna_features is not None:
                    all_dna_features = gather_features(
                        dna_features,
                        self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
                else:
                    all_dna_features = None

                if text_features is not None:
                    all_text_features = gather_features(
                        text_features,
                        self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
                else:
                    all_text_features = None
            else:
                all_image_features = image_features.clone() if image_features is not None else None
                all_dna_features = dna_features.clone() if dna_features is not None else None
                all_text_features = text_features.clone() if text_features is not None else None
                        
            input_features = [image_features, dna_features, text_features]
            input_features = [item for item in input_features if item is not None]
            feature_list = [all_image_features, all_dna_features, all_text_features]
            feature_list = [item for item in feature_list if item is not None]

            if len(feature_list) < 2:
                raise ValueError("Too less element for calculating the contrastive loss.")

            contrastive_loss_list = []
            entailment_loss_list = []
            bind_to_idx = None
            if self.bind_to is not None:
                if self.bind_to == "image":
                    bind_to_idx = 0
                elif self.bind_to == "dna":
                    bind_to_idx = 1
                elif self.bind_to == "text":
                    bind_to_idx = 2

            for idx_a, (feature_a, input_feature_a) in enumerate(zip(feature_list, input_features)):
                for idx_b, (feature_b, input_feature_b) in enumerate(zip(feature_list, input_features)):
                    if bind_to_idx is not None:
                        if idx_a != bind_to_idx and idx_b != bind_to_idx:
                            continue
                    if idx_a == idx_b:
                        continue

                    if self.no_image_text_loss and (idx_a == 0 or idx_b == 0) and (idx_a == 2 or idx_b == 2):
                        continue
                    # feature_a = F.normalize(feature_a, p=2, dim=1)
                    # feature_b = F.normalize(feature_b, p=2, dim=1)

                    # sim_a_b = logit_scale * feature_a @ feature_b.T
                    # sim_b_a = logit_scale * feature_b @ feature_a.T
                    sim_a_b = -L.pairwise_dist(feature_a, feature_b, curv)
                    sim_b_a = -L.pairwise_dist(feature_b, feature_a, curv)

                    loss_a_b = self.criterion(logit_scale * sim_a_b, all_labels)
                    loss_b_a = self.criterion(logit_scale * sim_b_a, all_labels)
                    contrastive_loss_list.append(loss_a_b)
                    contrastive_loss_list.append(loss_b_a)

                    # TODO: make more robust
                    if len(input_feature_a.shape) == 2: # image and text
                    # Hyperbolic entailment loss: text should entail matching image.
                        if idx_a == 1 and idx_b == 0:
                            entailment_loss_list.append(
                                self.compute_entailment_loss(input_feature_a, input_feature_b, curv, exp=1e-6)
                            )
                        elif idx_a == 0 and idx_b == 1:
                            entailment_loss_list.append(
                                self.compute_entailment_loss(input_feature_b, input_feature_a, curv, exp=1e-6)
                            )
                    else: # image, dna, text
                        if self.loss_type == 'A': # T->I + T->D
                            if idx_a == 2 and (idx_b == 0 or idx_b == 1): # a is text
                                entailment_loss_list.append(
                                    self.compute_entailment_loss(input_feature_a, input_feature_b, curv, exp=1e-6)
                                )
                            elif idx_b == 2 and (idx_a == 0 or idx_a == 1): # b is text
                                entailment_loss_list.append(
                                    self.compute_entailment_loss(input_feature_b, input_feature_a, curv, exp=1e-6)
                                )
                        elif self.loss_type == 'B': # T->D->I
                            if (idx_a == 1 and idx_b == 0) or (idx_a==2 and idx_b==1):
                                entailment_loss_list.append(
                                    self.compute_entailment_loss(input_feature_a, input_feature_b, curv, exp=1e-6)
                                )
                            elif (idx_b == 1 and idx_a == 0) or (idx_b==2 and idx_a==1):
                                entailment_loss_list.append(
                                    self.compute_entailment_loss(input_feature_b, input_feature_a, curv, exp=1e-6)
                                )
                        elif self.loss_type == 'C': # T->I->D + T->I
                            if (idx_a == 1 and idx_b == 0) or (idx_a==2 and idx_b==1) or (idx_a==2 and idx_b==0):
                                entailment_loss_list.append(
                                    self.compute_entailment_loss(input_feature_a, input_feature_b, curv, exp=1e-6)
                                )
                            elif (idx_b == 1 and idx_a == 0) or (idx_b==2 and idx_a==1) or (idx_b==2 and idx_a==0):
                                entailment_loss_list.append(
                                    self.compute_entailment_loss(input_feature_b, input_feature_a, curv, exp=1e-6)
                                )

            contrastive_total_loss = sum(contrastive_loss_list) * 1.0 / len(contrastive_loss_list)
            entailment_total_loss = sum(entailment_loss_list) * 1.0 / len(entailment_loss_list)

            total_loss = contrastive_total_loss
            if self.entail_weight > 0:
                total_loss += self.entail_weight * entailment_total_loss

        return {"loss": total_loss, "contrastive_loss": contrastive_total_loss, "entailment_loss": entailment_total_loss}