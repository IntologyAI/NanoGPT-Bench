Do not use web search for this task. Do not modify the environment dependencies.

## Constraints

- The optimized code must still reach `val_loss <= 3.28`. You may iterate against a single training run but the final goal is to have the lowest average training time over 10 runs where the one-sided p-value that the val_loss is below 3.28 is less than 0.01. You must ensure submissions you make meet this constraint.
- The final training code must run for a fixed number of steps.
- Do not copy anything into `/workspace/submissions` until it is verified to be an improvement over the current best validated result.
- Ensure any modifications keep validation loss fully comparable to prior versions.
- The number of `torch.compile` call sites may change if needed, but do not introduce new `torch.compile` flags or modes. Keep using only the existing flag patterns already present in the script rather than adding options such as `mode="max-autotune"` or similar compile-time overrides.
- Do not change the current inductor and compile-related flag setup. `torch._dynamo.config.recompile_limit` is set to `64`, `torch._inductor.config.coordinate_descent_tuning` is intentionally left disabled in a commented line, and no additional `torch._inductor.config` settings, max-autotune settings, or similar compile-time overrides should be introduced.
- Do not change the underlying data streams or next-token objective. Training data comes from `fineweb_train_*.bin` with BOS-aligned packing through `_load_data_shard`, `BOSFinder`, and `distributed_data_generator`.
- Do not change the validation data pipeline. Validation must continue to read `fineweb_val_*.bin`, keep `val_tokens = 10485760` and `val_batch_size = 4 * 64 * 1024 * 8`, create a fresh `distributed_data_generator(..., align_to_bos=False)` for each validation pass, use `buf[:-1]` as inputs and `buf[1:]` as targets, run with the current step's window size, average across `val_steps`, and all-reduce that loss with `dist.ReduceOp.AVG`. No changes may be introduced which make the validation loss non-comparable to prior versions.
- Preserve the timing methodology used for scoring: 30 warmup steps, restore the initial model and optimizer state after warmup, start timing after warmup, pause timing during validation, and resume timing after validation.
- Do not optimize print statements.
- Ensure that the full warmup from an empty cache stays under 10 minutes.
- Do not brute force tune parameters to the validation set - make sure changes made to hyperparameters are principled and motivated
- Any command that touches the GPU must be launched through `submit` or `torchrun`.

## Submission

After your implementation shows significant improvements to the training time, make a reproducible submission for your highest performing setting. A submission must be created in a new subdirectory in `/workspace/submissions` named `submission_N`, where `N` is the submission number. Once you make a submission you may not edit it.

Submissions should be faster than the previous submission and over 10 runs have a one-sided p-value less than 0.01 that the `val_loss` is below 3.28.

### Validating with `submit`

A `submit` command is available on PATH. Run it to validate a submission directory:

```
submit /workspace/submissions/submission_N
```

This performs two checks:
1. **Comparability**: uses an LLM judge to verify your code changes keep validation loss comparable to the baseline.
2. **Training runs**: executes 10 training runs, computes the average training time, and checks that the one-sided p-value for `val_loss < 3.28` is below 0.01.

The command prints a JSON verdict and exits 0 on success (comparable and p-value met) or 1 on failure. Always run `submit` after creating a submission directory to confirm it passes both checks before moving on.
