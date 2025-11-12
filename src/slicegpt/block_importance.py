# Code borrowed from : https://github.com/sramshetty/ShortGPT
# Minor tweaks to work with model_adapter and utils


import logging

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import config
from .model_adapter import LayerAdapter, ModelAdapter
from .model_utils import get_layer0_inputs, get_signals
from .slicing_scheduler import ConfigSlicingScheduler, ConstSlicingScheduler, SlicingScheduler
from .utils import cleanup_memory, map_tensors

@torch.no_grad()
def collect_block_importances(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    angular: bool = False
) -> list[float]:
    """
    Compute Block Importance scores for model layers
    """
    model_adapter.model.eval()
    layers_num = len(model_adapter.get_layers())
    importances = [0 for _ in range(layers_num)] # +1 for embedding layer 
    n = 1
    num_batches = 0 

    # TODO: Implement using model_adapter methods
    for idx, batch in enumerate(tqdm(dataloader)):
        batch = map_tensors(batch, config.device)
        outputs = model_adapter.model(**batch, output_hidden_states=True)
        hiddens = outputs.hidden_states
        num_batches += 1

        assert len(hiddens) ==  layers_num + 1

        for i in range(len(hiddens) - n ):
                in_hidden = hiddens[i]
                out_hidden = hiddens[i+n]
                if angular:
                    # use only last token for angular distance as described in section 3.2
                    # https://arxiv.org/pdf/2403.17887.pdf
                    in_hidden = in_hidden[:,-1:]
                    out_hidden = out_hidden[:,-1:]
                
                importances[i] += block_influence(
                    in_hidden,
                    out_hidden,
                    angular=angular
                ).mean().cpu().item()

    importances = [imp / num_batches for imp in importances]

    return importances

def block_influence(
    input_hidden_state: torch.Tensor,
    output_hidden_state: torch.Tensor,
    angular: bool = False
) -> torch.Tensor:
    """
    input_hidden_state: B, T, C
    output_hidden_state: B, T, C
    """
    _, _, d = input_hidden_state.shape
    input_hidden_state = input_hidden_state.reshape(-1, d)
    output_hidden_state = output_hidden_state.reshape(-1, d)

    norm_input = input_hidden_state.norm(dim=-1, keepdim=True)
    norm_output = output_hidden_state.norm(dim=-1, keepdim=True)

    sim = (input_hidden_state @ output_hidden_state.T) / (norm_input * norm_output)
    sim = sim.diagonal().nan_to_num(nan=0.5)

    if angular:
        return (torch.arccos(sim) / torch.pi)

    return 1 - sim