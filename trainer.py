# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter

from monai.data import decollate_batch


def train_epoch(model, loader, optimizer, prompt_model, tokenizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            if args.use_prompt:
                data, target, text_prompt = batch_data
            else:
                data, target = batch_data
        else:
            if args.use_prompt:
                data, target, text_prompt = batch_data["image"], batch_data["label"], batch_data["prompt"]
            else:
                data, target = batch_data["image"], batch_data["label"]
        data, target = data.cuda(), target.cuda()
        if args.use_prompt:
            text_tokens = tokenizer(text_prompt).cuda()
            with torch.no_grad(), torch.cuda.amp.autocast():
                text_features = prompt_model.encode_text(text_tokens)
            if not args.amp:
                text_features = text_features.float()
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            if args.use_prompt:
                logits = model(data, prompt_in=text_features)
            else:
                logits = model(data)
            loss = loss_func(logits, target)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        run_loss.update(loss.item(), n=args.batch_size)
        print(
            "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
            "loss: {:.4f}".format(run_loss.avg),
            "time {:.2f}s".format(time.time() - start_time),
        )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def val_epoch(model, loader, prompt_model, tokenizer, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None):
    model.eval()
    run_acc = AverageMeter()
    start_time = time.time()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                if args.use_prompt:
                    data, target, text_prompt = batch_data
                else:
                    data, target = batch_data
            else:
                if args.use_prompt:
                    data, target, text_prompt = batch_data["image"], batch_data["label"], batch_data["prompt"]
                else:
                    data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(), target.cuda()
            if args.use_prompt:
                text_tokens = tokenizer(text_prompt).cuda()
                text_features = prompt_model.encode_text(text_tokens)
                if not args.amp:
                    text_features = text_features.float()
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    if args.use_prompt:
                        logits = model_inferer(data, prompt_in=text_features)
                    else:
                        logits = model_inferer(data)
                else:
                    if args.use_prompt:
                        logits = model(data, prompt_in=text_features)
                    else:
                        logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc, not_nans = acc_func.aggregate()
            acc = acc.cuda()

            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            avg_acc = np.mean(run_acc.avg)
            print(
                "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "acc",
                avg_acc,
                "time {:.2f}s".format(time.time() - start_time),
            )
            start_time = time.time()
    return run_acc.avg


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    prompt_model=None,
    tokenizer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
):
    writer = SummaryWriter(log_dir=args.logdir)
    print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    for epoch in range(start_epoch, args.max_epochs):
        print("{}, Epoch: {}".format(time.ctime(), epoch))
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, prompt_model, tokenizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        print(
            "Final training  {}/{}".format(epoch, args.max_epochs - 1),
            "loss: {:.4f}".format(train_loss),
            "time {:.2f}s".format(time.time() - epoch_time),
        )
        if writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            epoch_time = time.time()
            val_avg_acc = val_epoch(
                model,
                val_loader,
                prompt_model, 
                tokenizer,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )

            val_avg_acc = np.mean(val_avg_acc)

            print(
                "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                "acc",
                val_avg_acc,
                "time {:.2f}s".format(time.time() - epoch_time),
            )
            if writer is not None:
                writer.add_scalar("val_acc", val_avg_acc, epoch)
            if val_avg_acc > val_acc_max:
                print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                val_acc_max = val_avg_acc
                b_new_best = True
                if args.logdir is not None and args.save_checkpoint:
                    save_checkpoint(
                        model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                    )
            if args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

        if scheduler is not None:
            scheduler.step()

    if args.logdir is not None and args.save_checkpoint:
        save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_last_epoch.pt")
    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max
