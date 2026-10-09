"""Initialization-only adapter audit. Never called from generation/training loops."""
import hashlib
import json
from pathlib import Path


TARGET_MODULES = {'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}


def validate_adapter_recipe(path,model_dir=None):
    path = Path(path)
    config = json.loads((path/'adapter_config.json').read_text())
    expected = {'r':64, 'lora_alpha':32, 'lora_dropout':0.0,
                'bias':'none', 'task_type':'CAUSAL_LM', 'peft_type':'LORA'}
    errors = {k:(config.get(k),v) for k,v in expected.items() if config.get(k) != v}
    if set(config.get('target_modules',())) != TARGET_MODULES:
        errors['target_modules'] = 'must match the shared target recipe'
    # Per-module overrides or saved extra trainable modules are not this recipe.
    for key in ('rank_pattern','alpha_pattern','modules_to_save','use_dora','use_rslora'):
        if config.get(key):
            errors[key] = 'not supported in the shared initialization recipe'
    if errors:
        raise ValueError('TARGET_ADAPTER does not match SpecNaacl target LoRA: '+json.dumps(errors))
    manifest = path/'shared_initialization.json'
    if model_dir is not None and manifest.is_file():
        recorded = json.loads(manifest.read_text()).get('model_dir')
        if recorded is not None and Path(recorded).resolve() != Path(model_dir).resolve():
            raise ValueError('Shared initialization was created for a different target model')
    return config


def adapter_artifact(path):
    validate_adapter_recipe(path)
    path = Path(path)
    candidates = [path/'adapter_model.safetensors',path/'adapter_model.bin']
    artifact = next((p for p in candidates if p.is_file()),None)
    if artifact is None:
        raise FileNotFoundError('Missing adapter weights in '+str(path))
    digest = hashlib.sha256()
    with artifact.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    manifest = path/'shared_initialization.json'
    if manifest.is_file():
        expected = json.loads(manifest.read_text())['adapter_weights_sha256']
        if digest.hexdigest() != expected:
            raise ValueError('Shared adapter weights changed after initialization')
    return artifact,digest.hexdigest()


def verify_loaded_adapter(target,path,model_dir=None):
    """Check every saved parameter; copies happen only once during initialization."""
    import torch
    from peft import get_peft_model_state_dict
    validate_adapter_recipe(path,model_dir)
    artifact,digest = adapter_artifact(path)
    if artifact.suffix == '.safetensors':
        from safetensors.torch import load_file
        saved = load_file(str(artifact),device='cpu')
    else:
        saved = torch.load(artifact,map_location='cpu',weights_only=True)
    actual = get_peft_model_state_dict(target,adapter_name='default')
    if target.active_adapter != 'default':
        raise ValueError('Shared target adapter must be active')
    if any(not p.requires_grad for name,p in target.named_parameters() if 'lora_' in name):
        raise ValueError('Shared target adapter must remain trainable')
    if saved.keys() != actual.keys():
        raise ValueError('Loaded target adapter keys do not match shared initialization')
    for key,value in actual.items():
        expected = saved[key].to(dtype=value.dtype)
        if not torch.equal(value.detach().cpu(),expected):
            raise ValueError('Loaded target adapter differs at '+key)
    result = {'status':'PASS', 'adapter_weights_sha256':digest,
              'loaded_tensor_sha256':state_dict_sha256(actual),
              'adapter_parameters_verified':len(saved), 'model_dir':str(Path(model_dir).resolve()) if model_dir else None}
    target._shared_initialization = result
    print('Shared target initialization verified: '+json.dumps(result),flush=True)
    return result


def state_dict_sha256(state):
    """Canonical hash of tensor names, shapes, dtypes and actual loaded bytes."""
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        raw=value.reshape(-1).view(torch.uint8).numpy()
        for offset in range(0,raw.size,1024*1024):
            digest.update(memoryview(raw[offset:offset+1024*1024]))
        del raw,value
    return digest.hexdigest()


def preflight_shared_adapter(path, model_dir=None):
    import os
    if not path:
        if os.environ.get('GRPO_BENCHMARK', '0') == '1':
            raise ValueError('Official benchmark requires TARGET_ADAPTER / --load_lora_path shared checkpoint')
        return
    validate_adapter_recipe(path, model_dir)
    adapter_artifact(path)


def initialization_report(target, args, *, draft=None, method=None):
    """Startup proof only; no calls in decoding, feedback or optimizer loops."""
    import importlib.metadata
    import os
    import torch
    official = os.environ.get('GRPO_BENCHMARK', '0') == '1'
    result = dict(target_lora=getattr(target, '_shared_initialization',
                     {'status':'NOT VERIFIED','reason':'No shared target checkpoint loaded'}),
                  method=method, alignment_version='response_rows_v2',
                  effective_args=vars(args).copy(),
                  environment={name: os.environ.get(name) for name in (
                      'CPEAK_NODES','MAX_TREE_NODES_PER_SEQ','FIXED_TREE_TOPK_BY_DEPTH',
                      'OPD_ENABLED','OPD_SELECTION','OPD_MAX_FRONTIER_PER_HEAD',
                      'OPD_FRONTIER_WEIGHT','OPD_FAST_LR','OPD_RANK','OPD_TOPK')},
                  packages={name:importlib.metadata.version(name) for name in ('torch','transformers','peft')},
                  cuda=torch.version.cuda, gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                  target_dtype=str(next(target.parameters()).dtype),
                  attention_implementation=getattr(target.config, '_attn_implementation', None))
    if official:
        result['target_backbone'] = dict(status='PASS',loaded_tensor_sha256=state_dict_sha256(
            {k:v for k,v in target.state_dict().items() if 'lora_' not in k}))
        files = {}
        for name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json','config.json'):
            file=Path(args.model_dir)/name
            if file.is_file():files[name]=hashlib.sha256(file.read_bytes()).hexdigest()
        result['tokenizer_files']={'status':'PASS' if 'tokenizer_config.json' in files else 'NOT VERIFIED','hashes':files}
        dataset=Path(getattr(args,'dataset_path',''))
        if dataset.is_file():
            digest=hashlib.sha256()
            with dataset.open('rb') as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
            result['dataset']={'status':'PASS','file_sha256':digest.hexdigest()}
        else:result['dataset']={'status':'NOT VERIFIED'}
    if official and result['target_lora']['status'] != 'PASS':
        raise ValueError('Official benchmark requires verified loaded shared target tensors')
    if draft is not None and official:
        if getattr(args,'draft_initialization_mode','pretrained')!='pretrained':
            raise ValueError('Official benchmark requires a pretrained draft checkpoint')
        # Medusa owns A in both methods. FastGRPO adds it only for Reflex.
        result['draft'] = dict(status='PASS',loaded_tensor_sha256=state_dict_sha256(
            {k:v for k,v in draft.state_dict().items()
             if method in ('medusa','medusa_reflex') or 'opd_projector' not in k}))
    elif draft is not None:
        result['draft'] = {'status':'NOT VERIFIED','reason':'Enable GRPO_BENCHMARK=1 for loaded draft hash'}
    destination=getattr(args, 'summary_file', '')
    if destination and destination != os.devnull:
        rank=int(os.environ.get('RANK','0'))
        path=Path(destination).parent / f'initialization.rank{rank}.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2,default=str)+'\n')
    target._initialization_report=result
    return result
