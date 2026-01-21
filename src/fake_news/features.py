import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel

def roberta_vectorize(text_series, device, tokenizer, model, batch_size=16, max_length=128):

  embeddings = []

  for i in range(0, len(text_series), batch_size):
    encoded = tokenizer(
        text_series[i:i+batch_size].to_list(),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    encoded = {key: val.to(device) for key, val in encoded.items()}

    with torch.no_grad():
      output = model(**encoded)

     # output.last_hidden_state shape:
     # (batch_size, sequence_length, hidden_size)
     # hidden_size = 768
    cls_embedding = output.last_hidden_state[:,0,:] # <s> token
    embeddings.append(cls_embedding.cpu().numpy())

  return np.vstack(embeddings)
