import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers.utils.import_utils import is_flash_attn_2_available

from ..utils import cleanup_torch_resources, setup_multiprocessing


class ColQwen2_5Loader:
    """
    Loader for the `ColQwen2.5` model and processor.
    """

    def __init__(self, model_name: str = "vidore/colqwen2.5-v0.2"):
        self.model_name = model_name
        self._device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self._dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        self._attn_implementation = (
            "flash_attention_2" if is_flash_attn_2_available() else None
        )
        self._model = None
        self._processor = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup."""
        self.cleanup()

    def cleanup(self):
        """Clean up resources to prevent memory leaks."""
        if self._model is not None:
            del self._model
            self._model = None

        if self._processor is not None:
            del self._processor
            self._processor = None

        # Use the utility function for cleanup
        cleanup_torch_resources()

    def load(self):
        """
        Load the ColQwen2.5 model and processor.
        """
        # Setup multiprocessing to prevent semaphore issues
        setup_multiprocessing()

        if is_flash_attn_2_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self._model = self.load_model()
        self._processor = self.load_processor()
        return self._model, self._processor

    def load_model(self) -> ColQwen2_5:
        model = ColQwen2_5.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            device_map=self._device,
            torch_dtype=self._dtype,
            attn_implementation=self._attn_implementation,
        ).eval()
        return model

    def load_processor(self) -> ColQwen2_5_Processor:
        processor = ColQwen2_5_Processor.from_pretrained(
            pretrained_model_name_or_path=self.model_name
        )
        assert isinstance(processor, ColQwen2_5_Processor)
        return processor
