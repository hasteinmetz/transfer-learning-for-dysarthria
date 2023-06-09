'''
File that contains data structures and model architectures to train a multitask model with 
an auxiliary task involving classifying speech as dysarthric, non-dysarthric, or L2 speech
'''

from torch import nn
import torch
from typing import *
from .base import BaseWav2Vec2
from transformers import Wav2Vec2Config
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from .dann import GRL
from args import Arguments
from datasets import DatasetDict
from collections import namedtuple

''' ############ Logging ############ '''
import logging
logger = logging.getLogger("wav2vec2_logger")
''' ################################# '''


'''Utility functions'''

TaskValues = namedtuple('TaskValues', ['tasks', 'mapping'])

def get_dann_task_values(args: Arguments, task_values: List[int], dataset: DatasetDict
                         ) -> Tuple[DatasetDict, TaskValues]:
    '''Function used in train.py to generate task_values and mappings relevant for each type of 
       DANN classifier (task, speaker, intelligibility)
       Arguments:
        - args: Set of arguments generated in args.py for training
        - task_values: A current list of task values (dys, ctl, l2) that the classifier uses to
                       reduce domain differences
    '''
    if args.model_config['classes'] == 'speakers' or args.model_config['classes'] == 'intelligibility':
        from re import match
        # set variables here and perform same mapping function
        if args.model_config['classes'] == 'speakers':
            logger.message("Using speakers as auxiliary classifier classes...")
            dys_regex, ctl_regex = r'(T[MF]|U[FM])\d', r'(T[MF]C|UC[FM])\d'
            feature = 'speaker'
        else:
            logger.message("Using speaker intelligibility/L1 as classes...")
            dataset = dataset.add_intelligibility_scores('src/eval/intelligibility_data.json')
            dys_regex, ctl_regex = r'(low|mid|high|very low)', r'EN'
            feature = 'l1'
        
        ctl = 0 if 'no_control_layer' in args.model_config and args.model_config['no_control_layer'] else 2
        # different train and all features so that DANN can generalize to new settings
        train_features = sorted(dataset['train'].unique(feature)) if 'train' in dataset else None
        all_features = dataset.get_unique(feature)
        s2t = lambda x: ctl if match(dys_regex, x) else 0 if match(ctl_regex, x) else 1
        # generate mappings for whole dataset and just for training
        def get_mapping(feat):
            '''Helper function to generate tasks based on features defined above'''
            return {t: s2t(dataset.decode_class_label(feature, t)) for t in feat} if feat else None
        mapping = {'train': get_mapping(train_features), 'all': get_mapping(all_features)}
        # switch task value to the other values in dataset (mapping later disambiguates this)
        def make_auxiliary_vals(x):
            x['task'] = x[feature]
            return x
        dataset = dataset.map(make_auxiliary_vals)
    else:
        logger.message("Using task as classes...")
        # regular task mapping
        mapping = {'train': {t: t for t in task_values}, 'all': {t: t for t in task_values}}

    return dataset, TaskValues(task_values, mapping)


'''Model class definitions'''

class AuxWav2Vec2Config(Wav2Vec2Config):
    def __init__(self, *args, **kwargs):
        self._loss_weight = kwargs.pop('dataset_loss_weighting', None)
        self.task_values = kwargs.pop('task_values', [0, 1])
        self._num_tasks = len(self.task_values)
        self._task_dict = {i: v for i, v in enumerate(self.task_values)}
        self.fp_precision = kwargs.pop('fp16', 'full')
        self.no_control_layer = kwargs.pop('no_control_layer', False)
        self.training_type = kwargs.pop('training_type', 'supervised')

        self._task_mapping = kwargs.pop('task_mapping', None)
        self.domaincls_loc = kwargs.pop('domaincls_loc', "ctc_layer")

        if self._loss_weight is None:
            self._loss_weight = [1.] * self._num_tasks
        else:
            assert len(self._loss_weight) == self._num_tasks
        
        if self._task_mapping:
            self.task_mapping_all = self._task_mapping['all']
            self.task_mapping = self._task_mapping['train']
            self.loss_weight = [self._loss_weight[v] for v in self.task_mapping_all.values()]
            self.task_dict = self.task_mapping_all
            self.num_tasks = len(self.task_mapping)

        super().__init__(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        '''Load config from json file but adjust the task_dict for correct typing'''
        config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        config._task_dict = {int(k): int(v) for k, v in config._task_dict.items()}
        config.task_mapping = {int(k): int(v) for k, v in config.task_mapping.items()}
        config.task_mapping_all = {int(k): int(v) for k, v in config.task_mapping_all.items()}
        config.loss_weight = [config._loss_weight[v] for v in config.task_mapping_all.values()]
        config.task_dict = config.task_mapping
        config.num_tasks = len(config.task_dict)
        return config


@dataclass
class MultitaskOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    logits_classifier: torch.FloatTensor = None


class LinearClassifier(nn.Module):
    def __init__(self, in_features: int, out_features: int, loss_weight: List[float]) -> None:
        super(LinearClassifier, self).__init__()
        self.projector = nn.Linear(in_features, in_features)
        self.classifier = nn.Linear(in_features, out_features)
        self.loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(loss_weight))

    def forward(self, hidden_state_i: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        projected_hidden_state = self.projector(hidden_state_i)
        final_states = torch.mean(projected_hidden_state, dim=1)
        logits = self.classifier(final_states)
        loss = self.loss_fct(logits, task)
        # if self.training:
        loss = self.loss_fct(logits, task)
        return loss, logits
        # else:
        #     return None, logits

    

class LinearClassifierGRL(nn.Module):
    def __init__(self, in_features: int, out_features: int, 
                 loss_weight: List[float], lmbda: float = 1.) -> None:
        super(LinearClassifierGRL, self).__init__()
        self.projector = nn.Linear(in_features, in_features)
        self.classifier = nn.Linear(in_features, out_features)
        self.loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(loss_weight[0:out_features]))
        self.grl = GRL(lmbda)

    def forward(self, hidden_state: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        reversed_hidden_state = self.grl(hidden_state)
        projected_hidden_state = self.projector(reversed_hidden_state)
        final_states = torch.mean(projected_hidden_state, dim=1)
        logits = self.classifier(final_states)
        # if self.training:
        loss = self.loss_fct(logits, task)
        return loss, logits
        # else:
        #     return None, logits


class AuxtaskWav2Vec2(BaseWav2Vec2):
    '''
    Multitask model that trains a Wav2Vec2 model on the CTC objective and a classification
    task with the hope of learning representations that way
    '''

    def __init__(self, config: AuxWav2Vec2Config):
        super().__init__(config) # initialize pretrained model
        
        self.config = config

        task_weight = config.task_weight if hasattr(config, 'task_weight') else 0.5
        if isinstance(task_weight, list) and len(task_weight) == 2:
            self.task_weights = {'ctc': task_weight[0], 'classifier': task_weight[1]}
        else:
            self.task_weights = {'ctc': task_weight, 'classifier': 1 - task_weight}

        self.unsupervised = True if config.training_type == 'unsupervised' else False

        self.classifier = LinearClassifier(config.hidden_size, config.num_tasks, config.loss_weight)

        super().post_init()

    def forward(
            self, 
            input_values: torch.Tensor, 
            task: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None, 
            output_attentions: Optional[bool] = None, 
            output_hidden_states: Optional[bool] = None, 
            return_dict: Optional[bool] = None, 
            labels: Optional[torch.Tensor] = None
        ) -> Union[Tuple, MultitaskOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output_hidden_states = True if self.config.domaincls_loc == 'extractor' else output_hidden_states

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict
        )

        final_state = self.dropout(outputs.last_hidden_state)
        
        loss_classifier, logits_classifier = None, None

        if self.config.domaincls_loc == 'extractor':
            loss_classifier, logits_classifier = self.classifier(outputs.hidden_states[-1], task)
        elif self.config.domaincls_loc == 'encoder':
            loss_classifier, logits_classifier = self.classifier(outputs.hidden_states[2], task)
        else:
            loss_classifier, logits_classifier = self.classifier(final_state, task)

        logits = self.lm_head(final_state)

        loss_ctc = 0.0

        if self.unsupervised and self.training:
            mask = task != self.config.task_dict[0]
            loss_labels = labels[mask] if labels is not None else None
            loss_logits, loss_tasks = logits[mask], task[mask]
            input_values = input_values[mask]
        else:
            loss_labels, loss_logits = labels, logits
            loss_tasks = task

        if labels is not None and loss_labels.size()[0] != 0:

            if loss_labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # get the length of logits for each row
            input_lengths, target_lengths, flattened_targets = self.get_output_lengths(input_values, loss_labels)
            # input_lengths roughly equivalent to seq_length when no attention

            log_probs = nn.functional.log_softmax(loss_logits, dim=-1, 
                                                  dtype=torch.float32).transpose(0, 1)

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
            _losses = self.compute_loss(_losses, loss_tasks)
            loss_ctc = torch.mean(_losses)

        loss = (self.task_weights['ctc'] * loss_ctc) 

        # if self.training:
        loss += (self.task_weights['classifier'] * loss_classifier)
        # else:
        #     probs = nn.functional.softmax(logits_classifier, dim=-1)
        #     logger.info(f"(classifier) classes: {task}\npredictions: {torch.argmax(probs, -1)}")
                    
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MultitaskOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, 
            attentions=outputs.attentions, logits_classifier=logits_classifier,
        )