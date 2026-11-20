import copy
import ctypes
import weakref
from enum import IntEnum
import numpy as np
from numpy.ctypeslib import ndpointer
from .loader import get_lib
from .tokenizer import TokenizerStruct

_SIZE_T = np.dtype(f"u{ctypes.sizeof(ctypes.c_size_t)}")
SIZE_MAX = (1 << (8 * ctypes.sizeof(ctypes.c_size_t))) - 1
engine_size_t = copy.copy(_SIZE_T)

class WorkspaceType(IntEnum):
    ENGINE_INFERENCE = 0
    ENGINE_TRAINING = 1

class ModelStruct(ctypes.Structure):
    pass

def _destroy_model(lib, ptr):
    """Freeing Model* via FreeModel(Model**). Not a method - without self."""
    ref = ctypes.pointer(ptr)   # Model* → Model**
    lib.FreeModel(ref)

class Model:
    def __init__(self):
        self.lib = get_lib()
        self._setup()
        self._ptr = None
        self._finalizer = None
        self._context_buf = None
        self._label_buf = None
        self.forward_type = None
        self._use_train = self._build_type()
        if self._use_train:
            self._train_setup()

    def _setup(self):
        lib = self.lib

        # Model* CreateModel(
        # 	const size_t vocab_size,
        # 	const size_t position_size,
        # 	const size_t d_model,
        # 	const size_t num_heads,
        # 	const size_t num_layers,
        # 	const size_t ffn_dim,
        # 	const float eps
        # );
        lib.CreateModel.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
        ]
        lib.CreateModel.restype = ctypes.POINTER(ModelStruct)

        # void InitModel(
        # 	Model* model,
        # 	const float stddev
        # );
        lib.InitModel.argtypes = [ctypes.POINTER(ModelStruct), ctypes.c_float]
        lib.InitModel.restype = None

        # int InitWorkspace(
        # 	Model* model,
        # 	WorkspaceType type,
        # 	size_t chunk_size
        # );
        lib.InitWorkspace.argtypes = [
            ctypes.POINTER(ModelStruct),
            ctypes.c_int,
            ctypes.c_size_t
        ]
        lib.InitWorkspace.restype = ctypes.c_int

        # size_t InferenceStep(
        # 	Model* model,
        # 	const size_t* inputs,
        # 	const size_t inputs_size,
        # 	const float temperature
        # );
        lib.InferenceStep.argtypes = [
            ctypes.POINTER(ModelStruct),
            ndpointer(dtype=_SIZE_T, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
            ctypes.c_float]
        lib.InferenceStep.restype = ctypes.c_size_t

        # void ShowDetails(Model* self);
        lib.ShowDetails.argtypes = [ctypes.POINTER(ModelStruct)]
        lib.ShowDetails.restype = None

        # int SaveModel(const Model* model, const int with_gradients, const char* filepath);
        lib.SaveModel.argtypes = [ctypes.POINTER(ModelStruct), ctypes.c_int, ctypes.c_char_p]
        lib.SaveModel.restype = ctypes.c_int

        # Model* LoadModel(const char* filepath, const int use_mmap);
        lib.LoadModel.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.LoadModel.restype = ctypes.POINTER(ModelStruct)

        # void FreeModel(Model** self);
        lib.FreeModel.argtypes = [ctypes.POINTER(ctypes.POINTER(ModelStruct))]
        lib.FreeModel.restype = None

        # int _build_type(void);
        lib._build_type.argtypes = []
        lib._build_type.restype = ctypes.c_int

        # void AttachTokenizer(
        # 	Model* model,
        # 	Tokenizer* tokenizer
        # );
        lib.AttachTokenizer.argtypes = [ctypes.POINTER(ModelStruct), ctypes.POINTER(TokenizerStruct)]
        lib.AttachTokenizer.restype = None

    def _train_setup(self):
        lib = self.lib

        # int InitTrain(
        # 	Model* model
        # );
        lib.InitTrain.argtypes = [ctypes.POINTER(ModelStruct)]
        lib.InitTrain.restype = ctypes.c_int

        # float TrainStep(
        # 	Model* model,
        # 	const size_t* inputs,
        # 	const size_t* target,
        # 	const uint8_t* labels,
        # 	const size_t inputs_size
        # );
        lib.TrainStep.argtypes = [
            ctypes.POINTER(ModelStruct),
            ndpointer(dtype=_SIZE_T, ndim=1, flags="C_CONTIGUOUS"),
            ndpointer(dtype=_SIZE_T, ndim=1, flags="C_CONTIGUOUS"),
            ndpointer(dtype=np.uint8, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_size_t
        ]
        lib.TrainStep.restype = ctypes.c_float

        # float OptimizerStep(
        # 	Model* self,
        # 	const float learning_rate,
        # 	const float beta1,
        # 	const float beta2,
        # 	const float eps,
        # 	const float clip_norm,
        # 	const float weight_decay
        # );
        lib.OptimizerStep.argtypes = [
            ctypes.POINTER(ModelStruct),
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float
        ]
        lib.OptimizerStep.restype = ctypes.c_float

        # void OptimizerReset(
        # 	Model* model
        # );
        lib.OptimizerReset.argtypes = [ctypes.POINTER(ModelStruct)]
        lib.OptimizerReset.restype = None

    def _build_type(self):
        res = self.lib._build_type()
        return bool(res)

    def pin_ptr(self, ptr) -> None:
        if not ptr:
            raise ValueError("Cannot pin NULL pointer")
        if self._ptr is not None:
            raise RuntimeError("You cannot rebind the Model!")

        self._ptr = ptr
        self._finalizer = weakref.finalize(
            self,
            _destroy_model,
            self.lib,
            ptr,
        )

    def create(self,
               vocab_size: int,
               position_size: int,
               d_model: int,
               num_heads: int,
               num_layers: int,
               ffn_dim: int,
               eps: float
               ) -> None:
        if self._ptr is not None:
            raise RuntimeError("Model already exists in this object")
        ptr = self.lib.CreateModel(
            vocab_size,
            position_size,
            d_model,
            num_heads,
            num_layers,
            ffn_dim,
            eps
        )
        self.pin_ptr(ptr)

    def load(self, filepath: str, use_mmap: bool) -> None:
        if self._ptr is not None:
            raise RuntimeError("Model already exists in this object")
        if not filepath:
            raise ValueError("filepath must be a non-empty string")
        ptr = self.lib.LoadModel(filepath.encode("utf-8"), int(use_mmap))
        self.pin_ptr(ptr)

    def close(self) -> None:
        self._finalizer()
        self._ptr = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        if self._ptr is None:
            if self._finalizer is None:
                state = "has not been created"
            else:
                state = "closed"
        else:
            state = f"ptr={self._ptr}"
        return f"<Model {state}>"

    def show_details(self) -> None:
        if self._ptr is None:
            raise RuntimeError("The model has not been created or has been cleared")
        self.lib.ShowDetails(self._ptr)

    def init_model(self, stddev: float) -> None:
        if self._ptr is None:
            raise RuntimeError("The model has not been created or has been cleared")
        self.lib.InitModel(self._ptr, stddev)

    def init_workspace(self, _type: WorkspaceType, chunk_size: int) -> None:
        if self._ptr is None:
            raise RuntimeError("The model has not been created or has been cleared")
        if not self._use_train and _type == WorkspaceType.ENGINE_TRAINING:
            raise ValueError(f"This build does not support: {WorkspaceType.ENGINE_TRAINING.name}")
        res = self.lib.InitWorkspace(self._ptr, int(_type), chunk_size)
        if res != 0:
            raise RuntimeError("Failed to prepare the workspace")
        self.forward_type = _type

    def inference_step(self, x: np.ndarray, temperature: float=1.0) -> int:
        if self._ptr is None:
            raise RuntimeError("The model has not been created or has been cleared")
        if self.forward_type != WorkspaceType.ENGINE_INFERENCE:
            raise ValueError(
                f"inference_step requires ENGINE_INFERENCE workspace, "
                f"got {self.forward_type.name}"
            )
        tok = self.lib.InferenceStep(self._ptr, x, x.size, temperature)
        if tok == SIZE_MAX:
            raise RuntimeError("Error during inference")
        return tok
    #

    def init_train(self) -> None:
        if self._ptr is None:
            raise RuntimeError("The model has not been created or has been cleared")
        if not self._use_train:
            raise RuntimeError("The library was assembled without a training mode")
        res = self.lib.InitTrain(self._ptr)
        if res != 0:
            raise RuntimeError("Failed to prepare the train")

    def train_step(self, x: np.ndarray, y: np.ndarray, labels: np.ndarray) -> float:
        """For x and y, use dtype=engine_size_t;
        otherwise, you will encounter a type incompatibility error."""
        if self._ptr is None:
            raise RuntimeError("The model has not been created or has been cleared")
        if not self._use_train:
            raise RuntimeError("This build does not support training mode.")
        if self.forward_type != WorkspaceType.ENGINE_TRAINING:
            raise ValueError(
                f"train_step requires ENGINE_TRAINING workspace, "
                f"got {self.forward_type.name}"
            )
        if not (x.size == y.size == labels.size):
            raise ValueError(
                f"size mismatch: x={x.size}, y={y.size}, labels={labels.size}"
            )
        if x.size == 0:
            raise ValueError("empty batch")
        loss = self.lib.TrainStep(self._ptr, x, y, labels, x.size)
        return loss


    def optimizer_step(self,
                       learning_rate: float,
                       beta1: float,
                       beta2: float,
                       eps: float,
                       clip_norm: float,
                       weight_decay: float
                       ) -> float:
        if self._ptr is None:
            raise RuntimeError("The model has not been created or has been cleared")
        if not self._use_train:
            raise RuntimeError("The library was assembled without a training mode")
        global_norm = self.lib.OptimizerStep(self._ptr, learning_rate, beta1, beta2, eps, clip_norm, weight_decay)
        return global_norm

    def optimizer_reset(self) -> None:
        if self._ptr is None:
            raise RuntimeError("The model has not been created or has been cleared")
        if not self._use_train:
            raise RuntimeError("The library was assembled without a training mode")
        self.lib.OptimizerReset(self._ptr)