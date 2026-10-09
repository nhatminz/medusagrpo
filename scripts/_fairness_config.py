"""Read-only extraction of actual CLI parser defaults and shell config layers."""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess


def effective_cli(path,flags,pure=False):
    """Execute parser declarations only, never import a trainer/model/runtime."""
    tree=ast.parse(path.read_text())
    if pure:
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parse_args')
        scope={'argparse':argparse,'__doc__':'Read-only configuration audit'}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
        tokens=[part for k,v in flags.items() if k!='nproc_per_node' for part in ('--'+k,v)]
        args=scope['parse_args'](tokens)
    else:
        parser=argparse.ArgumentParser()
        nodes=[n for n in tree.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
               and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name)
               and n.value.func.value.id=='parser' and n.value.func.attr=='add_argument']
        scope={'parser':parser}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
        options={option for action in parser._actions for option in action.option_strings}
        tokens=[part for k,v in flags.items() if '--'+k in options for part in ('--'+k,v)]
        args=parser.parse_args(tokens)
    values=vars(args)
    values['nproc_per_node']=int(flags.get('nproc_per_node','1'))
    return values


def model_config(root,key):
    """Inspect model-config defaults in an isolated shell, before wrapper overrides."""
    path=root/'configs'/key/'b200.env'
    if not path.exists():return None
    fields=('BATCH_SIZE','ACCUMULATION_STEPS','DRAFT_ACCUMULATION_STEPS',
            'MAX_TRAINING_TOKEN','MAX_TRAINING_PADDING_GAP','MODEL','MODEL_TYPE')
    command='source "$1"\n'+"printf '%s\\n' "+' '.join('"${'+field+':-}"' for field in fields)
    env={k:v for k,v in os.environ.items() if k in ('PATH','HOME','LANG')}
    result=subprocess.run(['bash','-c',command,'config-audit',str(path)],env=env,
                          text=True,capture_output=True,check=True)
    return dict(zip(fields,result.stdout.splitlines()))
