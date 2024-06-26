from dataclasses import dataclass

from transformers.file_utils import ModelOutput
from typing import Optional, Tuple
import torch


@dataclass
class MySequenceClassifierOutput(ModelOutput):
    """
    Customized class for outputs of sentence classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        treatment_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `treatment_labels` is provided):
            Classification loss for treatment assignment.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        all_potential_outcome_logits (Dict[int, torch.FloatTensor], optional):
            Dictionary of logits for counterfactual treatments, keyed by treatment ID.
        treatment_logits (`torch.FloatTensor` of shape `(batch_size, config.num_treatments)`):
            Classification scores for treatment assignment (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    treatment_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    outcome_attentions: Optional[Tuple[torch.FloatTensor]] = None
