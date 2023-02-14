import logging
import os
import pickle
from datetime import datetime

import torch
import transformers
from torch.utils.data import DataLoader

from utils.function_module import EarlyStopping
from utils.train_template import MyDataset, collate_fn, calculate_acc


class Train:
    def __init__(self, args):
        self.args = args
        self.logger = self.create_logger()
        self.train_dataset, self.validate_dataset = self.load_dataset()

    def create_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler(filename=self.args.log_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)

        return logger

    def load_dataset(self):
        self.logger.info("loading training dataset and validating dataset...")
        with open(self.args.train_path, "rb") as f:
            input_list = pickle.load(f)

        input_list_train = input_list[self.args.val_num:]
        input_list_val = input_list[:self.args.val_num]

        train_dataset = MyDataset(input_list_train, self.args.max_len)
        val_dataset = MyDataset(input_list_val, self.args.max_len)

        return train_dataset, val_dataset

    def train(self, model: torch.nn.Module):
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True,
                                      num_workers=self.args.num_workers, collate_fn=collate_fn, drop_last=True)
        validate_dataloader = DataLoader(self.validate_dataset, batch_size=self.args.batch_size, shuffle=True,
                                         num_workers=self.args.num_workers, collate_fn=collate_fn, drop_last=True)
        early_stopping = EarlyStopping(self.args.patience, verbose=True, save_path=self.args.save_model_path)
        t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.epochs
        optimizer = transformers.AdamW(model.parameters(), lr=self.args.lr, eps=self.args.eps)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        self.logger.info('starting training...')
        train_losses, validate_losses = [], []
        best_val_loss = 10000

        for epoch in range(self.args.epochs):
            train_loss = self._train_epoch(
                model=model, train_dataloader=train_dataloader, optimizer=optimizer, scheduler=scheduler, epoch=epoch)
            train_losses.append(train_loss)

            validate_loss = self._validate_epoch(model=model, validate_dataloader=validate_dataloader, epoch=epoch)
            validate_losses.append(validate_loss)

            if validate_loss < best_val_loss:
                best_val_loss = validate_loss
                self.logger.info('saving current best model for epoch {}...'.format(epoch + 1))
                model_path = os.path.join(self.args.save_model_path, 'min_ppl_model'.format(epoch + 1))
                if not os.path.exists(model_path):
                    os.mkdir(model_path)
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(model_path)

            if self.args.patience == 0:
                continue
            early_stopping(validate_loss, model)
            if early_stopping.early_stop:
                self.logger.info("early stopping...")
                break
        self.logger.info('training finished...')
        self.logger.info("train_losses:{}".format(train_losses))
        self.logger.info("validate_losses:{}".format(validate_losses))

    def _train_epoch(self, model: torch.nn.Module, train_dataloader: DataLoader, optimizer, scheduler, epoch):
        model.train()
        epoch_start_time = datetime.now()
        total_loss, epoch_correct_num, epoch_total_num = 0, 0, 0

        for batch_idx, (input_ids, labels) in enumerate(train_dataloader):
            try:
                input_ids = input_ids.to(self.args.device)
                labels = labels.to(self.args.device)
                outputs = model.forward(input_ids, labels=labels)
                logits = outputs.logits
                loss = outputs.loss
                loss = loss.mean()

                batch_correct_num, batch_total_num = calculate_acc(logits, labels, ignore_index=self.args.ignore_index)
                epoch_correct_num += batch_correct_num
                epoch_total_num += batch_total_num
                batch_acc = batch_correct_num / batch_total_num

                total_loss += loss.item()
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if (batch_idx + 1) % self.args.log_step == 0:
                    self.logger.info("batch {} of epoch {}, loss {}, batch_acc {}, lr {}".format(
                            batch_idx + 1, epoch + 1, loss.item() * self.args.gradient_accumulation_steps, batch_acc,
                            scheduler.get_lr()
                        )
                    )

                del input_ids, outputs

            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    self.logger.info("WARNING: ran out of memory...")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    self.logger.info(str(exception))
                    raise exception

        epoch_mean_loss = total_loss / len(train_dataloader)
        epoch_mean_acc = epoch_correct_num / epoch_total_num
        self.logger.info("epoch {}: loss {}, predict_acc {}".format(epoch + 1, epoch_mean_loss, epoch_mean_acc))

        self.logger.info('saving model for epoch {}'.format(epoch + 1))
        model_path = os.path.join(self.args.save_model_path, 'epoch{}'.format(epoch + 1))
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(model_path)
        self.logger.info('epoch {} finished...'.format(epoch + 1))
        epoch_finish_time = datetime.now()
        self.logger.info('time for one epoch: {}'.format(epoch_finish_time - epoch_start_time))

        return epoch_mean_loss

    def _validate_epoch(self, model, validate_dataloader, epoch):
        self.logger.info("start validating...")
        model.eval()
        device = self.args.device
        epoch_start_time = datetime.now()
        total_loss = 0
        try:
            with torch.no_grad():
                for batch_idx, (input_ids, labels) in enumerate(validate_dataloader):
                    input_ids = input_ids.to(device)
                    labels = labels.to(device)
                    outputs = model.forward(input_ids, labels=labels)
                    loss = outputs.loss
                    loss = loss.mean()

                    total_loss += loss.item()
                    del input_ids, outputs

                epoch_mean_loss = total_loss / len(validate_dataloader)
                self.logger.info("validate epoch {}: loss {}".format(epoch + 1, epoch_mean_loss))
                epoch_finish_time = datetime.now()
                self.logger.info('time for validating one epoch: {}'.format(epoch_finish_time - epoch_start_time))
                return epoch_mean_loss
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                self.logger.info("WARNING: run out of memory...")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                self.logger.info(str(exception))
                raise exception

    def predict_one(self):
        pass

    def predict_all(self):
        pass
