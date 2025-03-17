import math
import torch.nn.functional as F
import timm
import torch.nn as nn
from bioscanclip.model.mlp import MLPEncoder
from bioscanclip.model.image_encoder import LoRA_ViT_timm, LoRA_ViT_OpenCLIP
from bioscanclip.model.dna_encoder import load_pre_trained_bioscan_bert, LoRA_barcode_bert, Freeze_DNA_Encoder
from bioscanclip.model.language_encoder import load_pre_trained_bert, LoRA_bert, LoRA_bert_OpenCLIP
from bioscanclip.util.util import add_lora_layer_to_open_clip
import numpy as np
from typing import Optional
import torch
import open_clip
from bioscanclip.util.util import remove_module_from_state_dict
from bioscanclip.util import lorentz as L


class SimpleCLIP(nn.Module):
    def __init__(self, image_encoder, dna_encoder, language_encoder, open_clip_model=None, init_logit_scale: float = np.log(1 / 0.07),
                 init_logit_bias: Optional[float] = None, for_bio_clip=False):
        super(SimpleCLIP, self).__init__()
        self.image_encoder = image_encoder
        self.dna_encoder = dna_encoder
        self.language_encoder = language_encoder
        self.open_clip_model = open_clip_model
        self.tokenizer_for_open_clip = open_clip.get_tokenizer('ViT-B-32') if open_clip_model is not None else None
        if for_bio_clip:
            self.tokenizer_for_open_clip = open_clip.get_tokenizer('hf-hub:imageomics/bioclip') if open_clip_model is not None else None
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

    def forward(self, image_input, dna_input, language_input):
        image_output = None
        dna_output = None
        language_output = None


        if self.dna_encoder is not None:
            dna_output = F.normalize(self.dna_encoder(dna_input), p=2, dim=-1)

        if self.open_clip_model is not None:
            if image_input is not None:
                image_features = self.open_clip_model.encode_image(image_input)
                image_output = F.normalize(image_features, p=2, dim=-1)
            if language_input is not None:
                language_input = self.tokenizer_for_open_clip(language_input, context_length=77)
                language_input = language_input.to(image_input.device)
                text_features = self.open_clip_model.encode_text(language_input)
                language_output = F.normalize(text_features, p=2, dim=-1)
        else:
            if self.image_encoder is not None:
                image_output = F.normalize(self.image_encoder(image_input), p=2, dim=-1)
            if self.language_encoder is not None:
                language_output = F.normalize(self.language_encoder(language_input), p=2, dim=-1)
        return image_output, dna_output, language_output, self.logit_scale.exp(), self.logit_bias


class SimpleCLIP_hyperbolic(SimpleCLIP):
    def __init__(self, image_encoder, dna_encoder, language_encoder, open_clip_model=None, init_logit_scale: float = np.log(1 / 0.07),
                 init_logit_bias: Optional[float] = None, for_bio_clip=False, 
                 embed_dim: int = 768, curv_init: float = 1.0, learn_curv: bool = True):
        super().__init__(image_encoder, dna_encoder, language_encoder, open_clip_model, init_logit_scale, init_logit_bias, for_bio_clip)

        # Initialize curvature parameter. Hyperboloid curvature will be `-curv`.
        self.curv = nn.Parameter(
            torch.tensor(curv_init).log(), requires_grad=learn_curv
        )
        # When learning the curvature parameter, restrict it in this interval to
        # prevent training instability.
        self._curv_minmax = {
            "max": math.log(curv_init * 10),
            "min": math.log(curv_init / 10),
        }

        # Learnable scalars to ensure that image/text features have an expected
        # unit norm before exponential map (at initialization).
        self.visual_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
        self.textual_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
        self.dna_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())

    @property
    def device(self) -> torch.device:
        return self.logit_scale.device

    def forward(self, image_input, dna_input, language_input):

        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()

        # Clamp scaling factors such that they do not up-scale the feature norms.
        # Once `exp(scale) = 1`, they can simply be removed during inference.
        self.visual_alpha.data = torch.clamp(self.visual_alpha.data, max=0.0)
        self.textual_alpha.data = torch.clamp(self.textual_alpha.data, max=0.0)

        image_output = None
        dna_output = None
        language_output = None

        if self.dna_encoder is not None:
            dna_features = self.dna_encoder(dna_input) * self.dna_alpha.exp()
            # These features are space components of embeddings in the tangent
            # space of the Hyperboloid origin (which is Euclidean). Apply projection.
            with torch.autocast(self.device.type, dtype=torch.float32):
                dna_output = L.exp_map0(dna_features, curv=self.curv.exp())

        if self.open_clip_model is not None:
            if image_input is not None:
                image_features = self.open_clip_model.encode_image(image_input) * self.visual_alpha.exp()
                with torch.autocast(self.device.type, dtype=torch.float32):
                    image_output = L.exp_map0(image_features, curv=self.curv.exp())
            if language_input is not None:
                language_input = self.tokenizer_for_open_clip(language_input, context_length=77)
                language_input = language_input.to(image_input.device)
                text_features = self.open_clip_model.encode_text(language_input) * self.textual_alpha.exp()
                with torch.autocast(self.device.type, dtype=torch.float32):
                    language_output = L.exp_map0(text_features, curv=self.curv.exp())
        else:
            if self.image_encoder is not None:
                image_features = self.image_encoder(image_input) * self.visual_alpha.exp()
                with torch.autocast(self.device.type, dtype=torch.float32):
                    image_output = L.exp_map0(image_features, curv=self.curv.exp())
            if self.language_encoder is not None:
                language_features = self.language_encoder(language_input) * self.textual_alpha.exp()
                with torch.autocast(self.device.type, dtype=torch.float32):
                    language_output = L.exp_map0(language_features, curv=self.curv.exp())

        # Clamp temperature such that logits are not scaled more than 100x.
        # ln(100) = ~4.6052
        self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)

        return image_output, dna_output, language_output, self.logit_scale.exp(), _curv


def load_vit_for_simclr_training(args, device=None):
    torch.cuda.empty_cache()
    image_model_name = 'vit_base_patch16_224'
    if hasattr(args.model_config.image, 'pre_train_model'):
        image_model_name = args.model_config.image.pre_train_model
    pre_trained_timm_vit = timm.create_model(image_model_name, pretrained=True)
    for param in pre_trained_timm_vit.parameters():
        param.requires_grad = True
    return pre_trained_timm_vit


def wrap_vit_into_simple_clip(args, vit, device=None):
    torch.cuda.empty_cache()
    # Either we have the small models
    image_encoder = None
    dna_encoder = None
    language_encoder = None
    open_clip_model = None

    disable_lora = True

    image_encoder = vit

    model = SimpleCLIP(image_encoder=image_encoder, dna_encoder=dna_encoder,
                       language_encoder=language_encoder, open_clip_model=open_clip_model)

    if device is not None:
        model.to(device)

    if disable_lora:
        for param in model.parameters():
            param.requires_grad = True

    return model


def load_clip_model(args, device=None):
    torch.cuda.empty_cache()
    # Either we have the small models
    image_encoder = None
    dna_encoder = None
    language_encoder = None

    # Or when we use the OpenCLIP model
    open_clip_model = None

    disable_lora = False
    if hasattr(args.model_config, 'disable_lora'):
        disable_lora = args.model_config.disable_lora

    using_open_clip = False
    if hasattr(args.model_config, 'using_open_clip'):
        disable_lora = args.model_config.using_open_clip

    image_model = None
    if hasattr(args.model_config, 'image'):
        if hasattr(args.model_config.image, 'model'):
            image_model = args.model_config.image.model

    language_model = None
    if hasattr(args.model_config, 'language'):
        if hasattr(args.model_config.language, 'model'):
            language_model = args.model_config.language.model

    dna_model = "barcode_bert"
    if hasattr(args.model_config, 'dna'):
        if hasattr(args.model_config.dna, 'model'):
            dna_model = args.model_config.dna.model

    for_bio_clip = False
    if hasattr(args.model_config, 'for_bio_clip'):
        for_bio_clip = args.model_config.for_bio_clip

    if (using_open_clip or (image_model == "lora_clip_image" and language_model == "lora_clip_text")) and not for_bio_clip:
        open_clip_model, _, _ = open_clip.create_model_and_transforms('ViT-L/14', pretrained='commonpool_xl_laion_s13b_b90k')
        open_clip_model.to(device)
        if not disable_lora:
            open_clip_model = add_lora_layer_to_open_clip(open_clip_model, r=4, num_classes=args.model_config.output_dim)
    elif for_bio_clip:
        model, _, _ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
        tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
        open_clip_model = model
        print("Using BioCLIP model")
    else:
        # For image part
        if args.model_config.image.input_type == "image":
            image_model_name = 'vit_base_patch16_224'
            if hasattr(args.model_config.image, 'pre_train_model'):
                image_model_name = args.model_config.image.pre_train_model
            pre_trained_timm_vit = timm.create_model(image_model_name, pretrained=True)
            if hasattr(args.model_config.image, 'image_encoder_trained_with_simclr_style_ckpt_path'):
                image_encoder_trained_with_simclr_style_ckpt_path = args.model_config.image.image_encoder_trained_with_simclr_style_ckpt_path
                checkpoint = torch.load(image_encoder_trained_with_simclr_style_ckpt_path, map_location='cpu')
                state_dict = checkpoint['state_dict']
                state_dict = remove_module_from_state_dict(state_dict)
                pre_trained_timm_vit.load_state_dict(state_dict)

                # Free cuda memory for state_dict
                del state_dict
                del checkpoint
                torch.cuda.empty_cache()
                print("Loaded image encoder from %s" % image_encoder_trained_with_simclr_style_ckpt_path)
            if disable_lora:
                image_encoder = LoRA_ViT_timm(vit_model=pre_trained_timm_vit, r=4,
                                              num_classes=args.model_config.output_dim, lora_layer=[])
            else:
                image_encoder = LoRA_ViT_timm(vit_model=pre_trained_timm_vit, r=4,
                                              num_classes=args.model_config.output_dim)
        else:
            image_encoder = MLPEncoder(input_dim=args.model_config.image.input_dim,
                                       hidden_dim=args.model_config.image.hidden_dim,
                                       output_dim=args.model_config.output_dim)

        # For language
        if hasattr(args.model_config, 'language'):
            if args.model_config.language.input_type == "sequence":
                language_model_name = 'prajjwal1/bert-small'
                if hasattr(args.model_config.language, 'pre_train_model'):
                    language_model_name = args.model_config.language.pre_train_model
                _, pre_trained_bert = load_pre_trained_bert(language_model_name)
                if disable_lora:
                    language_encoder = LoRA_bert(model=pre_trained_bert, r=4, num_classes=args.model_config.output_dim,
                                                 lora_layer=[])
                else:
                    language_encoder = LoRA_bert(model=pre_trained_bert, r=4, num_classes=args.model_config.output_dim)
            else:
                raise TypeError(f"Using {args.model_config.language.input_type} as language input is not support yet.")


    # For DNA part
    if hasattr(args.model_config, 'dna'):
        if args.model_config.dna.input_type == "sequence":
            if dna_model == "barcode_bert" or dna_model == "lora_barcode_bert":
                pre_trained_barcode_bert = load_pre_trained_bioscan_bert(
                    bioscan_bert_checkpoint=args.bioscan_bert_checkpoint)
                if disable_lora:
                    dna_encoder = LoRA_barcode_bert(model=pre_trained_barcode_bert, r=4,
                                                    num_classes=args.model_config.output_dim, lora_layer=[])
                else:
                    dna_encoder = LoRA_barcode_bert(model=pre_trained_barcode_bert, r=4,
                                                    num_classes=args.model_config.output_dim)
        else:
            dna_encoder = MLPEncoder(input_dim=args.model_config.dna.input_dim,
                                     hidden_dim=args.model_config.dna.hidden_dim,
                                     output_dim=args.model_config.output_dim)

    # load hyperbolic hyperparameters
    train_hyperbolic = False
    if hasattr(args.model_config, 'hyperbolic_space'):
        curv_init = 1.0
        if hasattr(args.model_config.hyperbolic_space, 'curv_init'):
            curv_init = args.model_config.hyperbolic_space.curv_init

        learn_curv = True
        if hasattr(args.model_config.hyperbolic_space, 'learn_curv'):
            learn_curv = args.model_config.hyperbolic_space.learn_curv
        
        train_hyperbolic = True
        if hasattr(args.model_config.hyperbolic_space, 'train_hyperbolic'):
            train_hyperbolic = args.model_config.hyperbolic_space.train_hyperbolic

    if train_hyperbolic:
        model = SimpleCLIP_hyperbolic(image_encoder=image_encoder, dna_encoder=dna_encoder,
                                      language_encoder=language_encoder, open_clip_model=open_clip_model,
                                      for_bio_clip=for_bio_clip, embed_dim=args.model_config.output_dim, 
                                      curv_init=curv_init, learn_curv=learn_curv)
    else:
        model = SimpleCLIP(image_encoder=image_encoder, dna_encoder=dna_encoder,
                           language_encoder=language_encoder, open_clip_model=open_clip_model, 
                           for_bio_clip=for_bio_clip)

    if device is not None:
        model.to(device)

    if disable_lora:
        for param in model.parameters():
            param.requires_grad = True

    # if for_bio_clip:
    #     for param in model.open_clip_model.parameters():
    #         param.requires_grad = False

    # Freeze based on requirements
    if hasattr(args.model_config, 'image') and hasattr(args.model_config.image, 'freeze') and args.model_config.image.freeze:
        if model.image_encoder is not None:
            for param in model.image_encoder.parameters():
                param.requires_grad = False
        elif model.open_clip_model is not None:
            for param in model.open_clip_model.visual.parameters():
                param.requires_grad = False
    if hasattr(args.model_config, 'dna') and hasattr(args.model_config.dna, 'freeze') and args.model_config.dna.freeze:
        if model.dna_encoder is not None:
            for param in model.dna_encoder.parameters():
                param.requires_grad = False
    if hasattr(args.model_config, 'language') and hasattr(args.model_config.language, 'freeze') and args.model_config.language.freeze:
        if model.language_encoder is not None:
            for param in model.language_encoder.parameters():
                param.requires_grad = False
        elif model.open_clip_model is not None:
            for param in model.open_clip_model.transformer.parameters():
                param.requires_grad = False
    return model
