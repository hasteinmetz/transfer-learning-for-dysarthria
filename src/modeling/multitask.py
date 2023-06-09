#!/bin/env python

'''
File that contains data structures and model architectures to run a training or
evaluation sequence
'''

from torch import nn
import numpy as np
import torch
import os
from typing import *
from transformers import Wav2Vec2Config
from .base import BaseWav2Vec2, MultitaskWav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Encoder
from transformers.modeling_outputs import CausalLMOutput, BaseModelOutput

''' ############ Logging ############ '''
import logging
logger = logging.getLogger("wav2vec2_logger")
''' ################################# '''


''' ######################## Multitask model ######################## '''

class SmallEncoder(Wav2Vec2Encoder):
    def __init__(self, config: Wav2Vec2Config, num_layers: int):
        '''Make Wav2Vec2Encoder with fewer layers'''
        original_config_layers = config.num_hidden_layers
        config.num_hidden_layers = num_layers
        super().__init__(config)
        config.num_hidden_layers = original_config_layers

    def forward(self, hidden_states: torch.tensor, attention_mask: Optional[torch.Tensor] = None, 
                output_attentions: bool = None, output_hidden_states: bool = None, 
                return_dict: bool = None):
        '''Adapted from: https://github.com/huggingface/transformers/blob/410b61ad7e8f69113a86d0003190e3c392c7c39a/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L663'''
        all_hidden_states = () if output_attentions else None
        all_self_attentions = () if output_attentions else None
        
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        for layer in self.layers:
            all_hidden_states = all_hidden_states + (hidden_states,) if output_hidden_states else None

            dropout_probability = np.random.uniform(0, 1)

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            if not skip_the_layer:
                if self.gradient_checkpointing and self.training:
                    # create gradient checkpointing function
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        hidden_states,
                        attention_mask,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class MultitaskWav2Vec2(BaseWav2Vec2):
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

        if not hasattr(config, 'reinitialize_last_n_layers'):
            raise ValueError("reinitialize_last_n_layers currently required for multitask learning")

        # for debugging: self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('facebook/wav2vec2-base-960h')

        self.branches = nn.ModuleList([
            SmallEncoder(config, config.reinitialize_last_n_layers) for _ in range(num_tasks)
        ])
        
        self.heads = nn.ModuleList([
            nn.Linear(output_hidden_size, config.vocab_size) for _ in range(num_tasks)])

        assert len(self.branches) == num_tasks
                
        super().post_init()


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        if os.path.isdir(pretrained_model_name_or_path):
            model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        else:
            cls._keys_to_ignore_on_load_missing.append(r'(heads|branches)')
            model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
            for head_i in range(len(model.heads)):
                model.heads[head_i].load_state_dict(model.lm_head.state_dict())
            assert [head_i.weight == model.lm_head.weight for head_i in model.heads]
            delattr(model, 'lm_head')
            assert [head_i.weight == model.heads[0].weight for head_i in model.heads]
        remove_layers = model.config.reinitialize_last_n_layers
        model.wav2vec2.encoder.layers = model.wav2vec2.encoder.layers[:-remove_layers]
        return model
 

    def forward(
            self,
            input_values: torch.Tensor,
            task: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
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
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_state = outputs.last_hidden_state

        # get the hidden states corresponding to each task
        # and apply to the linear layer
        logit_tensors, indexes = [], []
        all_hidden_states = outputs.hidden_states if output_hidden_states else None
        all_attentions = outputs.attentions if output_attentions else None
        for t_i in range(self.config.num_tasks):
            task_i = self.config.task_dict[t_i]
            # store original indices of the order
            indexes.extend(list(torch.argwhere(task == task_i).detach().cpu().flatten().numpy()))
            hidden_state_i = hidden_state[task == task_i]
            attention_mask_i = attention_mask[task == task_i] if output_attentions else None
            if hidden_state_i.size()[0] > 0:
                output_i = self.branches[t_i](hidden_state_i, attention_mask_i, output_attentions, 
                                              output_hidden_states, return_dict)
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (output_i.hidden_states,)
                if output_attentions:
                    all_attentions = all_attentions + (output_i.all_attentions,)
                last_hidden_state_i = self.dropout(output_i.last_hidden_state)
                logits_i = self.heads[t_i](last_hidden_state_i)
                logit_tensors.extend(torch.unbind(logits_i))

        logits_ordered = [logit_tensors[indexes.index(idx)] for idx in range(len(indexes))]
        logits = torch.stack(logits_ordered, dim=0)

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
            #print(_losses)
            # _labels = labels
            # _labels[_labels == -100] = self.tokenizer.pad_token_id
            # print("labels", self.tokenizer.batch_decode(labels, group_tokens=False))
            # preds = torch.argmax(logits, dim=-1)
            # print("predictions", self.tokenizer.batch_decode(preds))
            loss = torch.mean(_losses)
                    
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, 
            hidden_states=all_hidden_states, attentions=all_attentions
        )