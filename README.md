# record_reg_vals_thread

RRVT is a [NVBit](https://github.com/NVlabs/NVBit) tool for capturing a trace of all register values for specified block and warp.

It was forked from the `record_reg_vals` tool included in the NVBit
documentation.

Changes to functionality include:

  - Recording predicate values
  - More correct handling of widths and implicit registers
  - Option to choose whether to instrument before or after an instruction
  - The `process_rrv.py` tool in the [gpu-code-analyzer](https://github.com/pyxis-roc/gpucode-analyzer repository (SASS2C branch) can convert traces to a SQLite database

## Installation

1. Unpack the NVBit 1.7.7 release.

2. Clone this directory into the `tools/` repository.

3. Run `make`

4. Optionally, symlink `run_rrvt.py` to a directory in the path.


## Running using `run_rrvt.py`

You can use the `run_rrvt.py` like so to run the program to be instrumented:

```
run_rrvt.py --trace /path/to/trace.txt -- /path/to/cudaprogram program-opts
```

This will output the trace to `/path/to/trace.txt`. Use `run_rrvt.py
--help` for other options to control the run.

## Running manually

The following environment variables are supported:

  1. `TRACE_FILE`: Path to the output trace file.
  2. `INSTR_IPOINT_PRE`: Set to 1 to instrument values before execution, default is after.
  3. `INSTR_WARP`: Identify a warp as `ctax,ctay,ctaz,warp_id`, default is `0,0,0,0`

If you do not want to post-process the trace using the
`gpu-code-analyzer` tools, then `INSTR_BEGIN` and `INSTR_END` are also
available.
