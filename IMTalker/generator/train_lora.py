import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import pi
from torch.nn import Module
from torch.utils import data
from torch import optim
from generator.dataset import AudioMotionSmirkGazeDataset
from generator.FM import FMGenerator
from generator.options.base_options import BaseOptions
from pytorch_lightning.loggers import TensorBoardLogger


def append_dims(t, ndims):
    return t.reshape(*t.shape, *((1,) * ndims))


class L1loss(Module):
    def forward(self, pred, target, **kwargs):
        return F.l1_loss(pred, target)


class LoRALinear(nn.Module):
    def __init__(self, base, rank=16, alpha=32, dropout=0.05):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / max(1, self.rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scale


def _get_child(root, path):
    obj = root
    for part in path.split('.'):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _set_child(root, path, value):
    parts = path.split('.')
    parent = _get_child(root, '.'.join(parts[:-1])) if len(parts) > 1 else root
    key = parts[-1]
    if key.isdigit():
        parent[int(key)] = value
    else:
        setattr(parent, key, value)


def lora_target_names(
    model,
    include_pose_lora=True,
    include_audio_lora=True,
    include_cam_lora=True,
    only_pose_lora=False,
):
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if only_pose_lora:
            if name == 'pose_projection.0':
                names.append(name)
            continue
        tier1 = (
            name.startswith('fmt.blocks.') and (
                name.endswith('.attn.qkv') or
                name.endswith('.attn.proj') or
                name.endswith('.mlp.fc1') or
                name.endswith('.mlp.fc2')
            )
        )
        tier2 = name.startswith('fmt.blocks.') and name.endswith('.adaLN_modulation.1')
        tier3_names = {'gaze_projection.0'}
        if include_cam_lora:
            tier3_names.add('cam_projection.0')
        if include_audio_lora:
            tier3_names.add('audio_projection.0')
        tier3 = name in tier3_names or (include_pose_lora and name == 'pose_projection.0')
        tier4 = name in {
            'fmt.decoder.linear',
            'fmt.decoder.adaLN_modulation.1',
        }
        if tier1 or tier2 or tier3 or tier4:
            names.append(name)
    return names

def apply_lora_to_model(
    model,
    rank=16,
    alpha=None,
    dropout=0.05,
    include_pose_lora=True,
    include_audio_lora=True,
    include_cam_lora=True,
    only_pose_lora=False,
):
    if alpha is None:
        alpha = 2 * int(rank)
    for p in model.parameters():
        p.requires_grad = False
    targets = lora_target_names(
        model,
        include_pose_lora=include_pose_lora,
        include_audio_lora=include_audio_lora,
        include_cam_lora=include_cam_lora,
        only_pose_lora=only_pose_lora,
    )
    for name in targets:
        base = _get_child(model, name)
        _set_child(model, name, LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[LoRA] targets={len(targets)} rank={rank} alpha={alpha} dropout={dropout} "
        f"include_pose_lora={include_pose_lora} include_audio_lora={include_audio_lora} "
        f"include_cam_lora={include_cam_lora} only_pose_lora={only_pose_lora}"
    )
    print('[LoRA] target modules:')
    for t in targets:
        print(f"  - {t}")
    print(f"[LoRA] trainable={trainable:,} total={total:,} ratio={trainable / max(1,total):.6f}")
    return targets


class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        self.shadow = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                old = self.shadow.get(name)
                if old is None:
                    self.shadow[name] = param.data.clone().detach()
                else:
                    self.shadow[name] = ((1.0 - self.decay) * param.data + self.decay * old.to(param.device)).clone()

    def apply_shadow(self):
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data
                param.data = self.shadow[name].to(param.device)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class System(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.model = FMGenerator(opt)
        if opt.base_ckpt and os.path.exists(opt.base_ckpt):
            self.load_base_ckpt(opt.base_ckpt)
        else:
            raise FileNotFoundError(f"Missing --base_ckpt: {opt.base_ckpt}")
        include_pose = not opt.no_lora_pose_projection
        include_audio = not opt.no_lora_audio_projection
        include_cam = not opt.no_lora_cam_projection
        if getattr(opt, 'cam_condition_mode', 'normal') in ('remove', 'zero_embedding'):
            include_cam = False
        apply_lora_to_model(
            self.model,
            rank=opt.lora_rank,
            alpha=opt.lora_alpha,
            dropout=opt.lora_dropout,
            include_pose_lora=include_pose,
            include_audio_lora=include_audio,
            include_cam_lora=include_cam,
            only_pose_lora=opt.only_lora_pose_projection,
        )
        self.loss_fn = L1loss()
        self.ema = EMA(self.model, decay=opt.ema_decay)
        self.save_hyperparameters(vars(opt))

    def load_base_ckpt(self, ckpt_path):
        print(f"[INFO] Loading frozen base generator from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt.get('ema_state_dict') or ckpt.get('state_dict', ckpt)
        if any(k.startswith('model.') for k in state_dict.keys()):
            state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Base loaded. missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print('[INFO] missing sample:', missing[:20])
        if unexpected:
            print('[INFO] unexpected sample:', unexpected[:20])

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.ema.update()

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema_state_dict'] = self.ema.shadow
        checkpoint['lora_config'] = {
            'rank': self.opt.lora_rank,
            'alpha': self.opt.lora_alpha,
            'dropout': self.opt.lora_dropout,
            'include_pose_lora': not self.opt.no_lora_pose_projection,
            'include_audio_lora': not self.opt.no_lora_audio_projection,
            'include_cam_lora': not self.opt.no_lora_cam_projection,
            'only_pose_lora': self.opt.only_lora_pose_projection,
            'cam_condition_mode': self.opt.cam_condition_mode,
        }

    def training_step(self, batch, batch_idx):
        m_now = batch['m_now']
        noise = torch.randn_like(m_now)
        times = torch.rand(m_now.size(0), device=self.device)
        t = append_dims(times, m_now.ndim - 1)
        batch['m_now'] = t * m_now + (1 - t) * noise
        gt_flow = m_now - noise
        pred_flow = self.model(batch, t=times)
        fm_loss = self.loss_fn(pred_flow, gt_flow)
        velocity_loss = self.loss_fn(pred_flow[:, 1:] - pred_flow[:, :-1], gt_flow[:, 1:] - gt_flow[:, :-1])
        loss = fm_loss + velocity_loss
        self.log('train_loss', loss, prog_bar=True)
        self.log('fm_loss', fm_loss, prog_bar=True)
        self.log('velocity_loss', velocity_loss, prog_bar=False)
        return loss

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = optim.AdamW(params, lr=self.opt.lr, betas=(0.9, 0.999), weight_decay=self.opt.weight_decay)
        return opt


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        parser.add_argument('--dataset_path', default=None, type=str)
        parser.add_argument('--list_path', default=None, type=str)
        parser.add_argument('--skip_dataset_filter', action='store_true')
        parser.add_argument('--base_ckpt', default='/workspace/IMTalker/checkpoints/generator.ckpt', type=str)
        parser.add_argument("--resume_ckpt", default=None, type=str)
        parser.add_argument('--lr', default=1e-4, type=float)
        parser.add_argument('--weight_decay', default=0.0, type=float)
        parser.add_argument('--batch_size', default=256, type=int)
        parser.add_argument('--iter', default=1000000, type=int)
        parser.add_argument('--max_minutes', default=30, type=int)
        parser.add_argument('--save_freq', type=int, default=200)
        parser.add_argument('--exp_path', type=str, default='/workspace/exps')
        parser.add_argument('--exp_name', type=str, default='lora_debug')
        parser.add_argument('--lora_rank', type=int, default=16)
        parser.add_argument('--lora_alpha', type=int, default=32)
        parser.add_argument('--lora_dropout', type=float, default=0.05)
        parser.add_argument('--no_lora_pose_projection', action='store_true')
        parser.add_argument('--no_lora_audio_projection', action='store_true')
        parser.add_argument('--no_lora_cam_projection', action='store_true')
        parser.add_argument('--only_lora_pose_projection', action='store_true')
        parser.add_argument('--cam_condition_mode', choices=['normal', 'remove', 'zero_embedding'], default='normal')
        parser.add_argument('--devices', type=int, default=1)
        parser.add_argument('--ema_decay', type=float, default=0.9999)
        parser.add_argument('--rank', type=str, default='cuda')
        return parser


class DataModule(pl.LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def setup(self, stage=None):
        self.train_dataset = AudioMotionSmirkGazeDataset(
            opt=self.opt,
            start=0,
            end=-100,
            list_path=getattr(self.opt, 'list_path', None),
        )

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, num_workers=8, batch_size=self.opt.batch_size, shuffle=True, pin_memory=True)


if __name__ == '__main__':
    opt = TrainOptions().parse()
    system = System(opt)
    dm = DataModule(opt)
    logger = TensorBoardLogger(save_dir=opt.exp_path, name=opt.exp_name)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(opt.exp_path, opt.exp_name, 'checkpoints'),
        filename='{step:06d}',
        every_n_train_steps=opt.save_freq,
        save_top_k=-1,
        save_last=True,
    )
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=opt.devices,
        strategy='ddp' if opt.devices > 1 else 'auto',
        max_steps=opt.iter,
        max_time={'minutes': opt.max_minutes},
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=None,
    )
    if opt.resume_ckpt:
        print("[INFO] Resuming LoRA training from {}".format(opt.resume_ckpt))
    trainer.fit(system, dm, ckpt_path=opt.resume_ckpt if opt.resume_ckpt else None)
