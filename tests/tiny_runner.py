"""Synthetic integration fixture with controlled rewards; no performance claims."""
import os
import sys
import json
import runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM,PreTrainedTokenizerFast
from tokenizers import Tokenizer,models,pre_tokenizers,decoders
from peft import get_peft_model,LoraConfig,TaskType
from medusa.model import MedusaModel

torch.set_num_threads(1)
root=Path(sys.argv.pop(1));root.mkdir(parents=True,exist_ok=True)
if not (root/'model/config.json').exists():
    torch.manual_seed(44)
    config=Qwen2Config(vocab_size=97,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=2048)
    target=Qwen2ForCausalLM(config);target.save_pretrained(root/'model')
    MedusaModel(config,target).save_model(root/'heads/draft.pth')
    lora=LoraConfig(task_type=TaskType.CAUSAL_LM,r=64,lora_alpha=32,lora_dropout=0.,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
    get_peft_model(target,lora).save_pretrained(root/'initial_lora')
    vocab={'t0':0,**{chr(i):i-32 for i in range(33,127)},'Ġ':95,'t96':96}
    tok=Tokenizer(models.BPE(vocab,merges=[],unk_token='t0'))
    tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False);tok.decoder=decoders.ByteLevel()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=tok,unk_token='t0',pad_token='t0',eos_token='t96')
    tokenizer.chat_template="{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
    tokenizer.save_pretrained(root/'model')
    (root/'train.json').write_text(json.dumps([{'question':'x','answer':'2'}]*4))
    (root/'sharegpt.json').write_text(json.dumps([{'conversations':[{'from':'human','value':'What?'},{'from':'gpt','value':'Let us reason carefully step by step and give the answer.'}]}]*5))
if len(sys.argv)==1:sys.exit(0)
method,name=sys.argv[1:3];extra=sys.argv[3:];out=root/name
for sub in ('logs','target','draft','statistics','resume'):(out/sub).mkdir(parents=True,exist_ok=True)
import helper.rewards as rewards
import itertools
counter=itertools.count()
rewards.accuracy_reward_func=lambda completions,solution,**kw:[float(next(counter)%2) for _ in completions]
if '--test_constant_rewards' in extra:
    extra.remove('--test_constant_rewards')
    rewards.accuracy_reward_func=lambda completions,solution,**kw:[0.]*len(completions)
rewards.format_reward_func=lambda completions,**kw:[0.]*len(completions)
os.environ['TQDM_DISABLE']='1'
sys.argv=['grpo_speculative.py','--method',method,'--model_dir',str(root/'model'),'--adapter_path',str(root/'heads'),
 '--load_lora_path',str(root/'initial_lora'),'--dataset_path',str(root/'train.json'),'--train_data_fraction','1',
 '--batch_size','2','--accumulation_steps','2','--repeated_generate_nums','2','--num_epochs','1',
 '--max_length','36','--max_prompt_length','24','--max_training_token','1024','--max_training_padding_gap','1024',
 '--num_workers','0','--persistent_workers','false','--dtype','fp32','--device','cpu',
 '--log_file',str(out/'logs/metrics.jsonl'),'--timing_file',str(out/'logs/timing.csv'),'--summary_file',str(out/'summary.json'),
 '--saved_model_dir',str(out/'target'),'--saved_draft_model_dir',str(out/'draft'),'--saved_statistics_dir',str(out/'statistics'),
 '--checkpoint_dir',str(out/'resume'),'--opd_backend','auto',*extra]
runpy.run_path(str(ROOT/'grpo_speculative.py'),run_name='__main__')
