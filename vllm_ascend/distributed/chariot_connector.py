import os
import weakref
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Callable

import torch
from dllm.cpp_ext.kvc import KvcStore
from dllm.kvc import TorchAdaptor

from dllm.cpp_ext.kvc import KvcFuture

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole
)
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import MLACommonMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.distributed.parallel_state import (get_world_group, get_tp_group)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.request import Request

logger = init_logger(f"vllm.{__name__}")


@dataclass
class ReqMeta:
    request_id: str
    token_ids: torch.Tensor
    slot_mapping: torch.Tensor

    @staticmethod
    def make_meta(request_id: str, token_ids: list[int], block_ids: list[int], block_size: int) -> "ReqMeta":
        valid_num_tokens = align_to_block_size(len(token_ids), block_size)
        block_ids_tensor = torch.tensor(block_ids)
        num_blocks = block_ids_tensor.shape[0]
        block_offsets = torch.arange(0, block_size)
        slot_mapping = block_offsets.reshape(1, block_size) + block_ids_tensor.reshape(num_blocks, 1) * block_size
        token_ids_tensor = torch.tensor(token_ids)

        token_ids_tensor = torch.tensor(token_ids, dtype=torch.int32)
        slot_mapping = slot_mapping.flatten()[:len(token_ids)]
        return ReqMeta(request_id, token_ids_tensor, slot_mapping)
    

@dataclass
class ChariotConnectorMetadata(KVConnectorMetadata):
    request_metas = list[ReqMeta]


class ChariotFuture:
    def __init__(self):
        pass

    def wait(self, timeout):
        pass

    def __del__(self):
        pass


class ChariotConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        super().__init__(vllm_config, role)
        pass

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        pass
        def inject_kv_into_layer(dst_kv_layer, src_kv_layer, slot_mapping, request_id, layer_idx):
            pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: "AttentionMetadata", **kwargs) -> None:
        pass
    
        def extract_kv_from_layer(layer: torch.Tensor, slot_mapping: torch.Tensor) -> torch.Tensor:
            pass

    def wait_for_save(self) -> None:
        pass

    def get_num_new_matched_tokens(self, request, num_computed_tokens: int) -> int:
        pass

    def update_state_after_alloc(self, request, num_external_tokens: int):
        pass

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        pass
