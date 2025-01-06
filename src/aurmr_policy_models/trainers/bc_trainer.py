import os
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from aurmr_policy_models.trainers.base_trainer import BaseTrainer

class BCTrainer(BaseTrainer):
    """
    Implementation of a Behavioral Cloning (BC) trainer for continuous prediction tasks.
    """
    def __init__(self, batch_size, epochs, learning_rate, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    def train(self):
        self.model.cuda()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            start_time = time.time()
            progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch}")

            for batch_idx, (conditions, actions, _, _, _) in progress_bar:
                optimizer.zero_grad()
                conditions['state'] = conditions['state'].cuda()
                actions = actions.cuda()

                # outputs = model(conditions)
                loss = self.model.loss(actions, conditions)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            epoch_duration = time.time() - start_time
            average_loss = total_loss / len(train_dataloader)

            self.writer.add_scalar("Loss/Train", average_loss, epoch)
            self.writer.add_scalar("Time/Epoch", epoch_duration, epoch)

            print(f"Epoch {epoch}: Average Loss = {average_loss}, Duration = {epoch_duration:.2f} seconds")

            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(self.model, epoch)

        self.save_final_model(self.model)
        self.writer.close()
