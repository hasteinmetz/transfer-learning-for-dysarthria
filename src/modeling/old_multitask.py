#!/bin/env python

'''
NOTE: This file is used to rerun previously trained models and avoid configuration issues
File that contains data structures and model architectures to run a training or
evaluation sequence
'''

from torch import nn
import numpy as np
import torch
import os
from typing import *
from .base import BaseWav2Vec2
from .multitask import MultitaskWav2Vec2Config
from transformers.modeling_outputs import CausalLMOutput

''' ############ Logging ############ '''
import logging
logger = logging.getLogger("wav2vec2_logger")
''' ################################# '''


''' ######################## Mulitask model ######################## '''


class OldMTWav2Vec2(BaseWav2Vec2):
    '''Multitask learning model based on Wav2Vec2
        It consists of two branching linear layers after 
        passing inputs through the base model
    '''

    _keys_to_ignore_on_load_missing = []

    def __init__(self, config: MultitaskWav2Vec2Config):
        super().__init__(config) # initialize pretrained model
        
        self.config = config
        num_tasks = config.num_tasks
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and \
                config.add_adapter else config.hidden_size
        )

        self.heads = nn.ModuleList([
            nn.Linear(output_hidden_size, config.vocab_size) for _ in range(num_tasks)
        ])
        assert len(self.heads) == num_tasks
        
        super().post_init()


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        if os.path.isdir(pretrained_model_name_or_path):
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        else:
            cls._keys_to_ignore_on_load_missing.append(r'heads')
            model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
            for head_i in range(len(model.heads)):
                model.heads[head_i].load_state_dict(model.lm_head.state_dict())
            assert [head_i.weight == model.lm_head.weight for head_i in model.heads]
            delattr(model, 'lm_head')
            assert [head_i.weight == model.heads[0].weight for head_i in model.heads]
            return model
        

    def forward(
            self,
            input_values: torch.Tensor,
            task: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            labels: Optional[torch.Tensor] = None
        ) -> Union[Tuple, CausalLMOutput]:
        """Code adapted from Huggingface's Transformers package. CITATION:
        Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, 
	        R., Funtowicz, M., and Brew, J. (2020-10) Transformers: State-of-the-Art Natural Language 
            Processing (https://github.com/huggingface/transformers) arXiv preprint arXiv:1910.03771.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if torch.any(torch.isnan(input_values)):
            print("NAN in INPUT!!!")
        # logger.warn(f"inputs: {input_values.size()}")
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        # here, seq_length means the number of windows generated via wav2vec2 conv layers
        batch_size, seq_length, _ = hidden_states.size()

        # prepare logits tensor
        fp_type = torch.half if self.config.fp_precision == 'half' else torch.float
        logits = torch.empty(
            (batch_size, seq_length, self.config.vocab_size),
            device=self.device,
            dtype=fp_type
        )

        # get the hidden states corresponding to each task
        # and apply to the linear layer
        for t_i in range(self.config.num_tasks):
            task_i = self.config.task_dict[t_i]
            hidden_task_i = hidden_states[task == task_i]
            logits_i = self.heads[t_i](hidden_task_i)
            logits[task == task_i] = logits_i

        loss = None

        if labels is not None:

            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # get the length of logits for each row
            input_lengths, target_lengths, flattened_targets = self.get_output_lengths(input_values, labels)
            # input_lengths roughly equivalent to seq_length when no attention

            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                _losses = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    zero_infinity=self.config.ctc_zero_infinity,
                    reduction = 'none'
                )

            _losses = torch.div(_losses, target_lengths.to(_losses.device))
            _losses = self.compute_loss(_losses, task)
            loss = torch.mean(_losses)
                    
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )