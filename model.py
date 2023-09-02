from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoModel, AdamW, get_cosine_schedule_with_warmup
import torch.nn as nn
import math
from torchmetrics.functional.classification import auroc
import torch.nn.functional as F


class ToxicCommentsDataset(Dataset):

    def __init__(
                self,
                config,
                data: pd.DataFrame,
                tokenizer: BertTokenizer,
                max_token_len: int = 128,
              ):
        self.config = config
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len

  # def _prepare_data(self):
  #   data = pd.read_csv(self.data_path)
  #   data['unhealthy'] = np.where(data['healthy'] == 1, 0, 1)
  #   if self.sample is not None:
  #     unhealthy = data.loc[data[attributes].sum(axis=1) > 0]
  #     clean = data.loc[data[attributes].sum(axis=1) == 0]
  #     self.data = pd.concat([unhealthy, clean.sample(self.sample, random_state=7)])
  #   else:
  #     self.data = data

    def __len__(self):
      return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]

        comment_text = data_row.comment_text
        labels = data_row[self.config["data_labels"]]

        encoding = self.tokenizer.encode_plus(
                                              comment_text,
                                              add_special_tokens=True,
                                              max_length=self.max_token_len,
                                              return_token_type_ids=False,
                                              padding="max_length",
                                              truncation=True,
                                              return_attention_mask=True,
                                              return_tensors='pt',
                                            )
        return dict(
                    # comment_text=comment_text,
                    input_ids=encoding["input_ids"].flatten(),
                    attention_mask=encoding["attention_mask"].flatten(),
                    labels=torch.FloatTensor(labels)
                  )


class ToxicCommentDataModule(pl.LightningDataModule):

  def __init__(self, config, train_df, val_df, batch_size: int = 16, max_token_length: int = 128,  model_name='roberta-base'):
    super().__init__()
    self.config = config
    self.train_df = train_df
    self.val_df = val_df
    self.batch_size = batch_size
    self.max_token_length = max_token_length
    self.model_name = model_name
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)

  def setup(self, stage = None):
    if stage in (None, "fit"):
      self.train_dataset = ToxicCommentsDatasetNew(self.config, self.train_df, tokenizer=self.tokenizer)
      self.val_dataset = ToxicCommentsDatasetNew(self.config, self.val_df, tokenizer=self.tokenizer)
    if stage == 'predict':
      self.val_dataset = ToxicCommentsDatasetNew(self.val_path, attributes=self.attributes, tokenizer=self.tokenizer, sample=None)

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers=4, shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=4, shuffle=False)

  def predict_dataloader(self):
    return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=4, shuffle=False)
    
    
class ToxicCommentClassifier(pl.LightningModule):

  def __init__(self, config: dict):
    super().__init__()
    self.config = config
    self.pretrained_model = AutoModel.from_pretrained(config['model_name'], return_dict = True)
    self.hidden = torch.nn.Linear(self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size)
    self.classifier = torch.nn.Linear(self.pretrained_model.config.hidden_size, self.config['n_labels'])
    torch.nn.init.xavier_uniform_(self.classifier.weight)
    self.loss_func = nn.BCEWithLogitsLoss(reduction='mean')
    self.dropout = nn.Dropout()

  def forward(self, input_ids, attention_mask, labels=None):
    # roberta layer
    output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
    pooled_output = torch.mean(output.last_hidden_state, 1)
    # final logits
    pooled_output = self.dropout(pooled_output)
    pooled_output = self.hidden(pooled_output)
    pooled_output = F.relu(pooled_output)
    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)
    # calculate loss
    loss = 0
    if labels is not None:
      loss = self.loss_func(logits.view(-1, self.config['n_labels']), labels.view(-1, self.config['n_labels']))
    return loss, logits

  def training_step(self, batch, batch_index):
    loss, outputs = self(**batch)
    self.log("train loss", loss, prog_bar = True, logger=True)
    return {"loss":loss, "predictions":outputs, "labels": batch["labels"]}

  def validation_step(self, batch, batch_index):
    loss, outputs = self(**batch)
    self.log("validation loss", loss, prog_bar = True, logger=True)
    return {"val_loss": loss, "predictions":outputs, "labels": batch["labels"]}

  def predict_step(self, batch, batch_index):
    loss, outputs = self(**batch)
    return outputs

  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
    total_steps = self.config['train_size']/self.config['batch_size']
    warmup_steps = math.floor(total_steps * self.config['warmup'])
    warmup_steps = math.floor(total_steps * self.config['warmup'])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    return [optimizer],[scheduler]

  # def validation_epoch_end(self, outputs):
  #   losses = []
  #   for output in outputs:
  #     loss = output['val_loss'].detach().cpu()
  #     losses.append(loss)
  #   avg_loss = torch.mean(torch.stack(losses))
  #   self.log("avg_val_loss", avg_loss)


