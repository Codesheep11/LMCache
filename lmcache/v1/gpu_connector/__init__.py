# SPDX-License-Identifier: Apache-2.0
# Third Party
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.gpu_connector.mock_gpu_connector import MockGPUConnector
from lmcache.v1.gpu_connector.utils import LayoutHints, need_gpu_interm_buffer
from lmcache.v1.metadata import LMCacheMetadata


def CreateGPUConnector(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
    engine: EngineType,
    layout_hints: LayoutHints | None = None,
) -> GPUConnectorInterface:
    """
    Create a GPU Connector based on the configuration and metadata.

    Args:
        config: The LMCache engine configuration.
        metadata: The LMCache metadata.
        engine: The serving engine type (EngineType.VLLM, EngineType.SGLANG,
                or EngineType.MOCK).
        layout_hints: Optional hints from the serving engine about KV cache
            layout (e.g. ``{"kv_layout": "HND"}``).
    """
    if config.get_extra_config_value("xds_layerwise_readv", False) and (
        engine != EngineType.VLLM
        or not config.use_layerwise
        or not config.xds_backend
        or metadata.use_mla
        or config.enable_blending
        or not config.get_extra_config_value("xds_layerwise_block_blob", False)
    ):
        raise ValueError("xds_layerwise_readv requires VLLM xDS layerwise block blobs")
    use_gpu = need_gpu_interm_buffer(config)

    if engine == EngineType.SGLANG:
        # First Party
        from lmcache.v1.gpu_connector.gpu_connectors import (
            SGLangGPUConnector,
            SGLangLayerwiseGPUConnector,
        )

        num_layer, _, chunk_size, num_kv_head, head_dim = metadata.kv_shape
        hidden_dim_size = num_kv_head * head_dim
        local_worker_id = metadata.local_worker_id
        torch.cuda.device(local_worker_id)
        device = torch.device(f"cuda:{local_worker_id}")
        kv_dtype = metadata.kv_dtype
        if config.use_layerwise:
            return SGLangLayerwiseGPUConnector(
                hidden_dim_size,
                num_layer,
                use_gpu=use_gpu,
                chunk_size=chunk_size,
                dtype=kv_dtype,
                device=device,
                use_mla=metadata.use_mla,
            )
        else:
            return SGLangGPUConnector(
                hidden_dim_size,
                num_layer,
                use_gpu=use_gpu,
                chunk_size=chunk_size,
                dtype=kv_dtype,
                device=device,
                use_mla=metadata.use_mla,
            )
    elif engine == EngineType.VLLM:
        # First Party
        from lmcache.integration.vllm.utils import get_vllm_torch_dev
        from lmcache.v1.gpu_connector.gpu_connectors import (
            VLLMBufferLayerwiseGPUConnector,
            VLLMPagedMemGPUConnectorV2,
            VLLMPagedMemGPUConnectorV3,
            VLLMPagedMemLayerwiseGPUConnector,
        )

        local_worker_id = metadata.local_worker_id
        torch_dev, dev_name = get_vllm_torch_dev()
        if (
            config.get_extra_config_value("xds_layerwise_readv", False)
            and dev_name != "cuda"
        ):
            raise ValueError("xds_layerwise_readv requires CUDA")
        torch_dev.set_device(local_worker_id)
        device = torch.device(f"{dev_name}:{local_worker_id}")

        if dev_name == "xpu":
            # First Party
            from lmcache.v1.gpu_connector.xpu_connectors import (
                VLLMPagedMemLayerwiseXPUConnector,
                VLLMPagedMemXPUConnectorV2,
            )

            if config.use_layerwise:
                return VLLMPagedMemLayerwiseXPUConnector.from_metadata(
                    metadata, use_xpu=use_gpu, device=device
                )
            return VLLMPagedMemXPUConnectorV2.from_metadata(metadata, use_gpu, device)

        if config.use_layerwise:
            if config.get_extra_config_value("xds_layerwise_readv", False):
                from lmcache.v1.gpu_connector.xds_layerwise import (
                    XDSReadvLayerwiseGPUConnector,
                )

                if (
                    not config.xds_backend
                    or metadata.use_mla
                    or config.enable_blending
                    or not config.get_extra_config_value(
                        "xds_layerwise_block_blob", False
                    )
                ):
                    raise ValueError(
                        "xds_layerwise_readv requires xDS block blobs "
                        "without MLA/blending"
                    )
                return XDSReadvLayerwiseGPUConnector.from_metadata(
                    metadata, use_gpu, device, layout_hints=layout_hints
                )
            if config.enable_blending:
                return VLLMBufferLayerwiseGPUConnector.from_metadata(
                    metadata, use_gpu, device, layout_hints=layout_hints
                )
            else:
                return VLLMPagedMemLayerwiseGPUConnector.from_metadata(
                    metadata, use_gpu, device, layout_hints=layout_hints
                )

        elif dev_name == "cuda":
            if config.use_gpu_connector_v3:
                return VLLMPagedMemGPUConnectorV3.from_metadata(
                    metadata, use_gpu, device, layout_hints=layout_hints
                )
            else:
                return VLLMPagedMemGPUConnectorV2.from_metadata(
                    metadata, use_gpu, device, layout_hints=layout_hints
                )

        elif dev_name == "hpu":
            # First Party
            from lmcache.v1.gpu_connector.hpu_connector import (
                VLLMPagedMemHPUConnectorV2,
            )

            return VLLMPagedMemHPUConnectorV2.from_metadata(metadata, use_gpu, device)
        else:
            raise RuntimeError("No supported connector found for the current platform.")

    elif engine == EngineType.MOCK:
        kv_shape = metadata.kv_shape
        return MockGPUConnector(kv_shape=kv_shape)
    else:
        raise RuntimeError(f"Unsupported engine: {engine}")
