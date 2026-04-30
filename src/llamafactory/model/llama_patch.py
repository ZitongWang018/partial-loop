from transformers.models.llama.modeling_llama import LlamaForCausalLM as OriginalLlamaForCausalLM
from .modeling.modeling_llama import LlamaForCausalLM as CustomLlamaForCausalLM
from .modeling.modeling_llama_new import LlamaForCausalLM as CustomLlamaForCausalLMOurs
from .modeling.modeling_llama_pause import LlamaForCausalLM as CustomLlamaForCausalLMPause
from .modeling.modeling_llama_loop import LlamaForCausalLM as CustomLlamaForCausalLMLoop
from .modeling.modeling_gpt_neox_addhidden import GPTNeoXForCausalLM as addhiddenCustomGPTNeoXForCausalLM
from .modeling.modeling_gpt_neox_addhidden_weightshare import GPTNeoXForCausalLM as weightshareCustomGPTNeoXForCausalLM
from .modeling.modeling_gpt_neox_addpausetoken import GPTNeoXForCausalLM as pausetokenCustomGPTNeoXForCausalLM
from .modeling.modeling_gpt_neox_orin_random import GPTNeoXForCausalLM as orinrandomCustomGPTNeoXForCausalLM
from .modeling.modeling_gpt_neox_orin import GPTNeoXForCausalLM as orinCustomGPTNeoXForCausalLM
from .modeling.modeling_gpt_neox_addhidden_mlp import GPTNeoXForCausalLM as mlpCustomGPTNeoXForCausalLM
from .modeling.modeling_gpt_neox_bptt import GPTNeoXForCausalLM as bpttCustomGPTNeoXForCausalLM
from .modeling.modeling_gpt_neox import GPTNeoXForCausalLM as CustomGPTNeoXForCausalLM
from .modeling.modeling_llama_orin import LlamaForCausalLM as CustomLlamaForCausalLMOrin
from .modeling.modeling_gpt2 import GPT2LMHeadModel as CustomGPT2ForCausalLM
from .modeling.modeling_mamba2 import Mamba2ForCausalLM as CustomMamba2ForCausalLM
from .modeling.modeling_qwen2 import Qwen2ForCausalLM as CustomQwen2ForCausalLM
from .modeling.modeling_mixtral import MixtralForCausalLM as CustomMixtralForCausalLM
from .modeling.modeling_gpt_neox_base import GPTNeoXForCausalLM as CustomGPTNeoXForCausalLMBase
from .modeling.modeling_gpt_neox_baseline import GPTNeoXForCausalLM as CustomGPTNeoXForCausalLMBaseline
def patch_llama():
    """
    替换transformers库中的LlamaForCausalLM为我们的自定义版本
    """
    import transformers.models.llama.modeling_llama as llama_module
    llama_module.LlamaForCausalLM = CustomLlamaForCausalLM
def patch_llama_ours():
    """
    替换transformers库中的LlamaForCausalLM为我们的自定义版本
    """
    import transformers.models.llama.modeling_llama as llama_module
    llama_module.LlamaForCausalLM = CustomLlamaForCausalLMOurs
def patch_llama_pause():
    """
    替换transformers库中的LlamaForCausalLM为我们的自定义版本
    """
    import transformers.models.llama.modeling_llama as llama_module
    llama_module.LlamaForCausalLM = CustomLlamaForCausalLMPause
def patch_llama_orin():
    """
    替换transformers库中的LlamaForCausalLM为我们的自定义版本
    """
    import transformers.models.llama.modeling_llama as llama_module
    llama_module.LlamaForCausalLM = CustomLlamaForCausalLMOrin
def patch_llama_loop():
    """
    替换transformers库中的LlamaForCausalLM为我们的自定义版本
    """
    import transformers.models.llama.modeling_llama as llama_module
    llama_module.LlamaForCausalLM = CustomLlamaForCausalLMLoop

def bptt_patch_gpt_neox():
    """
    替换transformers库中的GPTNeoXForCausalLM为我们的自定义版本
    """
    import transformers.models.gpt_neox.modeling_gpt_neox as gpt_neox_module
    gpt_neox_module.GPTNeoXForCausalLM = bpttCustomGPTNeoXForCausalLM

def addhidden_patch_gpt_neox():
    """
    替换transformers库中的GPTNeoXForCausalLM为我们的自定义版本
    """
    import transformers.models.gpt_neox.modeling_gpt_neox as gpt_neox_module
    gpt_neox_module.GPTNeoXForCausalLM = addhiddenCustomGPTNeoXForCausalLM
def weightshare_patch_gpt_neox():
    """
    替换transformers库中的GPTNeoXForCausalLM为我们的自定义版本
    """
    import transformers.models.gpt_neox.modeling_gpt_neox as gpt_neox_module
    gpt_neox_module.GPTNeoXForCausalLM = weightshareCustomGPTNeoXForCausalLM

def pausetoken_patch_gpt_neox():
    """
    替换transformers库中的GPTNeoXForCausalLM为我们的自定义版本
    """
    import transformers.models.gpt_neox.modeling_gpt_neox as gpt_neox_module
    gpt_neox_module.GPTNeoXForCausalLM = pausetokenCustomGPTNeoXForCausalLM

def orin_patch_gpt_neox():
    """
    替换transformers库中的GPTNeoXForCausalLM为我们的自定义版本
    """
    import transformers.models.gpt_neox.modeling_gpt_neox as gpt_neox_module
    gpt_neox_module.GPTNeoXForCausalLM = orinCustomGPTNeoXForCausalLM

def mlp_patch_gpt_neox():
    """
    替换transformers库中的GPTNeoXForCausalLM为我们的自定义版本
    """
    import transformers.models.gpt_neox.modeling_gpt_neox as gpt_neox_module
    gpt_neox_module.GPTNeoXForCausalLM = mlpCustomGPTNeoXForCausalLM

def orinrandom_patch_gpt_neox():
    """
    替换transformers库中的GPTNeoXForCausalLM为我们的自定义版本
    """
    import transformers.models.gpt_neox.modeling_gpt_neox as gpt_neox_module
    gpt_neox_module.GPTNeoXForCausalLM = orinrandomCustomGPTNeoXForCausalLM
def patch_gpt_neox():
    """
    替换transformers库中的GPTNeoXForCausalLM为我们的自定义版本
    """
    import transformers.models.gpt_neox.modeling_gpt_neox as gpt_neox_module
    gpt_neox_module.GPTNeoXForCausalLM = CustomGPTNeoXForCausalLM
def patch_gpt_neox_base():
    """
    替换transformers库中的GPTNeoXForCausalLM为我们的自定义版本
    """
    import transformers.models.gpt_neox.modeling_gpt_neox as gpt_neox_module
    gpt_neox_module.GPTNeoXForCausalLM = CustomGPTNeoXForCausalLMBase
def patch_gpt_neox_baseline():
    """
    替换transformers库中的GPTNeoXForCausalLM为baseline版本
    """
    import transformers.models.gpt_neox.modeling_gpt_neox as gpt_neox_module
    gpt_neox_module.GPTNeoXForCausalLM = CustomGPTNeoXForCausalLMBaseline
def patch_gpt2():
    """
    替换transformers库中的GPT2ForCausalLM为我们的自定义版本
    """
    import transformers.models.gpt2.modeling_gpt2 as gpt2_module
    gpt2_module.GPT2LMHeadModel = CustomGPT2ForCausalLM
def patch_mamba2():
    """
    替换transformers库中的Mamba2ForCausalLM为我们的自定义版本
    """
    import transformers.models.mamba2.modeling_mamba2 as mamba2_module
    mamba2_module.Mamba2ForCausalLM = CustomMamba2ForCausalLM
def patch_qwen2():
    """
    替换transformers库中的Qwen2ForCausalLM为我们的自定义版本
    """
    import transformers.models.qwen2.modeling_qwen2 as qwen2_module
    qwen2_module.Qwen2ForCausalLM = CustomQwen2ForCausalLM
def patch_mixtral():
    """
    替换transformers库中的MixtralForCausalLM为我们的自定义版本
    """
    import transformers.models.mixtral.modeling_mixtral as mixtral_module
    mixtral_module.MixtralForCausalLM = CustomMixtralForCausalLM

