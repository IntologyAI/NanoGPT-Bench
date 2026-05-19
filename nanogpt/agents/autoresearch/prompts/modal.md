# Modal Execution

All benchmark runs in this environment must go through `/workspace/modal_run.py`.

Run each experiment from `/workspace` like this:

```bash
python "modal_run.py" --code-dir "." > run.log 2>&1
```

`modal_run.py` uploads the current workspace root, mounts the benchmark image context, and executes `/workspace/run.sh` remotely. Do not call `torchrun` directly in this mode.
