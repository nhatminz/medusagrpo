"""Native Medusa/SDPA/cache compatibility probe."""
from contextlib import contextmanager,redirect_stdout
from io import StringIO
import os
from tempfile import TemporaryDirectory
import warnings


@contextmanager
def _probe_proposal_profiles():
    """Keep the tiny FP32 probe independent of production proposal profiles."""
    names=('OPD_PROPOSAL_PROFILE','OPD_PROPOSAL_PROFILE_DIR')
    previous={name:os.environ.get(name) for name in names}
    output=StringIO()
    with TemporaryDirectory(prefix='medusa-runtime-probe-') as directory:
        try:
            os.environ['OPD_PROPOSAL_PROFILE']=''
            os.environ['OPD_PROPOSAL_PROFILE_DIR']=directory
            # The synthetic V=32 FP32 workload intentionally has no calibration.
            # Keep auto-dispatch and actual fallback execution in this probe.
            with warnings.catch_warnings(),redirect_stdout(output):
                warnings.filterwarnings('ignore',category=UserWarning,
                    message=r'No exact compatible OPD proposal profile;',
                    module=r'helper\.opd_reflex')
                yield
        finally:
            for name,value in previous.items():
                if value is None:os.environ.pop(name,None)
                else:os.environ[name]=value
            for line in output.getvalue().splitlines():
                print('[runtime probe] '+line,flush=True)


def probe_training_runtime(device='cpu'):
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from medusa.model import MedusaModel
    from medusa.generate import speculative_generate
    from types import SimpleNamespace
    with _probe_proposal_profiles():
        print('Runtime compatibility probe: synthetic V=32 FP32 model; production proposal profiles are checked when the training model starts.',flush=True)
        config=Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1)
        target=Qwen2ForCausalLM(config).to(device).eval()
        model=MedusaModel(config,target).to(device)
        out=speculative_generate(model,torch.tensor([[1,2,3]],device=device),
            torch.ones(1,3,device=device,dtype=torch.long),SimpleNamespace(eos_token_id=31),
            max_length=5,do_sample=False,opd_backend='auto')
    return dict(architecture='medusa_parallel_3',target_forward_calls=out['target_forward_calls'])
