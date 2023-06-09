'''
File that contains data structures and model architectures to train a multitask model with 
an auxiliary task involving classifying speech as dysarthric, non-dysarthric, or L2 speech
'''

from torch import nn
import torch
from typing import *
from transformers import Wav2Vec2ForCTC, Wav2Vec2Config
from transformers.modeling_outputs import CausalLMOutput

''' ############ Logging ############ '''
import logging
logger = logging.getLogger("wav2vec2_logger")
''' ################################# '''


class MultitaskWav2Vec2Config(Wav2Vec2Config):
    '''Subclass to pass additional parameters specific to MultitaskWav2Vec2'''
    def __init__(self, *args, **kwargs):
        self.loss_weight = kwargs.pop('dataset_loss_weighting', None)
        self.task_values = kwargs.pop('task_values', [0, 1])
        self.num_tasks = len(self.task_values)
        self.task_dict = {i: v for i, v in enumerate(self.task_values)}
        self.fp_precision = kwargs.pop('fp16', 'full')
        self.no_control_layer = kwargs.pop('no_control_layer', False)
        super().__init__(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        '''Load config from json file but adjust the task_dict for correct typing'''
        config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        config.task_dict = {int(k): int(v) for k, v in config.task_dict.items()}
        return config
    

class BaseWav2Vec2(Wav2Vec2ForCTC):
    '''
    Finetuning wav2vec2. Largely the same model with some small additions to loss calculations
    '''

    def __init__(self, config: Wav2Vec2Config):
        config.ctc_loss_reduction = 'none'
        super().__init__(config) # initialize pretrained model
        self.config = config
        
        # weights for each task
        self.loss_weight = config.loss_weight
        if self.loss_weight is not None:
            self.compute_loss = self._compute_weighted_loss
            logger.message(
                "Computing weighted loss with weights: " + 
                ", ".join([f"task{i}: {self.loss_weight[i]}" for i in range(len(self.loss_weight))])
            )
        else:
            self.compute_loss = self._compute_unweighted_loss
        
        super().post_init()
    

    def get_output_lengths(
            self,
            input_values: torch.Tensor, 
            labels: torch.Tensor,
            attention: Optional[torch.Tensor] = None
        ):
        """Obtain the output lengths from encoder. Code adapted from Huggingface's Transformers package. 
        CITATION:
        Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, 
	        R., Funtowicz, M., and Brew, J. (2020-10) Transformers: State-of-the-Art Natural Language 
            Processing (https://github.com/huggingface/transformers) arXiv preprint arXiv:1910.03771.
        """
        attention_mask = (
            attention if attention is not None else torch.ones_like(input_values, dtype=torch.long)
        )
        input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
        # assuming that padded tokens are filled with -100
        # when not being attended to
        labels_mask = labels >= 0
        target_lengths = labels_mask.sum(-1)
        flattened_targets = torch.masked_select(labels, labels_mask)
        return input_lengths, target_lengths, flattened_targets


    def _compute_unweighted_loss(self, _losses: torch.Tensor, *args):
        if torch.any(_losses > 500):
            logger.watch_out(f"Encountered really high CTC loss. " +
            "If you continue to see this warning frequently, or losses are above 1000 " +
            "check your configuration or code for bugs.\n" +
            f"losses: {_losses}\ntasks: {args[0] if len(args) > 0 else None}")
        return _losses


    def _compute_weighted_loss(self, _losses: torch.Tensor, task: torch.Tensor):
        task_weights = [self.loss_weight[i] for i in task]
        weights = torch.tensor(task_weights, requires_grad=True, device=_losses.device)
        # _losses.register_hook(lambda grad: print(grad))
        _loss = _losses * weights
        # print(_loss, weights, task)
        if torch.any(_loss > 500):
            logger.watch_out(f"Encountered really high CTC loss {_loss}. " +
            "If you continue to see this warning frequently, or losses are above 1000 " +
            "check your configuration or code for bugs.")
        
        return _loss
    

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
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        final_state = self.dropout(outputs.last_hidden_state)

        logits = self.lm_head(final_state)

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
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, 
            hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )