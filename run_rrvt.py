#!/usr/bin/env python3

import sys
import shlex
from pathlib import Path
import subprocess
import os

DEFAULT_WARP = "0,0,0,0"
SOFILE = "record-reg-vals-thread.so"

def get_sopath(sopath):
    if sopath is None:
        sopath = Path(__file__).parent

    if not str(sopath).endswith(".so"): # allows changing sofile name if needed
        sopath = sopath / SOFILE

    if not sopath.exists():
        print(f"ERROR: {sopath} does not exist. Use --sopath to specify path to {SOFILE}")
        sys.exit(1)

    return sopath

def parse_warp(warp_value):
    x = warp_value.split(",")

    if len(x) > 4:
        print(f"ERROR: Too many values in {warp_value}")
        sys.exit(1)

    out = []
    for v in x:
        try:
            out.append(int(v))
        except ValueError:
            print(f"ERROR: Can't parse {v} as integer.")
            sys.exit(1)

    out = out + [0]*(4 - len(out))
    return [str(s) for s in out]

def parse_range(rangespec, name):
    r = rangespec.split(",")

    if len(r) != 2:
        print("ERROR: {name}: range must be START,END")
        sys.exit(1)
    try:
        r = [int(x) for x in r]
        if r[0] > r[1]:
            print("ERROR: {name}: invalid range {r[0]}-{r[1]}")
            sys.exit(1)
        return map(str, r)
    except ValueError as e:
        print(e)
        sys.exit(1)

def main():
    import argparse
    p = argparse.ArgumentParser(description="Trace a CUDA program with register values")
    p.add_argument("--trace", help="Trace file")
    p.add_argument("--instr-before", help="Instrument before execution", action="store_true")
    p.add_argument("--warp", help="Warp to trace", default=DEFAULT_WARP)
    p.add_argument("--sopath", help="Path to record_reg_vals_thread.so", default=None, type=Path)
    p.add_argument("--use-ld-preload", help="Use LD_PRELOAD instead of CUDA_INJECTION64_PATH", action="store_true")
    p.add_argument("--no-compress", help="Do not compress trace file", action="store_true")
    p.add_argument("--kernel-range", help="Instrument a range of kernels", metavar="START,END")
    p.add_argument("--name", help="Instrument this kernel by mangled name")
    p.add_argument("--range", help="Instrument a range of instructions", metavar="START,END")
    p.add_argument("--verbose", help="Set TOOL_VERBOSE", action="store_true")
    p.add_argument("-n", dest="dry_run", help="Dry-run", action="store_true")

    args, cmdline = p.parse_known_args()

    if len(cmdline) == 0:
        print("ERROR: You must specify a command to execute.")
        sys.exit(1)

    env = {}
    if args.trace:
        env['TRACE_FILE'] = args.trace

    if args.instr_before:
        env['INSTR_IPOINT_PRE'] = "1"

    if args.warp != DEFAULT_WARP:
        env['INSTR_WARP'] = ",".join(parse_warp(args.warp))

    if cmdline[0] == "--":
        cmdline = cmdline[1:]

    sopath = get_sopath(args.sopath)
    if args.use_ld_preload:
        env['LD_PRELOAD'] = str(sopath)
    else:
        env['CUDA_INJECTION64_PATH'] = str(sopath)

    if args.verbose:
        env['TOOL_VERBOSE'] = '1'

    if args.range:
        st, end = parse_range(args.range, "--range")
        env['INSTR_BEGIN'] = st
        env['INSTR_END'] = end

    if args.kernel_range:
        st, end = parse_range(args.kernel_range, "--kernel-range")
        env['KERNEL_BEGIN'] = st
        env['KERNEL_END'] = end

    if args.name:
        env['KERNEL_NAME'] = args.name

    print(" ".join([f"%s=%s" % (k, v) for k, v in env.items()]), shlex.join(cmdline))

    if args.dry_run:
        sys.exit(0)

    try:
        runenv = dict(os.environ)
        runenv.update(env)

        subprocess.run(cmdline, env=runenv, check=True)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(e)
        sys.exit(1)

    if args.trace:
        tracefile = Path(args.trace)
        if not tracefile.exists():
            print(f"ERROR: {tracefile} not found. NVBit may not have run.")
            sys.exit(1)

        if not args.no_compress:
            subprocess.run(['bzip2', '-f', str(tracefile)])
            print(f"Compressed {tracefile} to {tracefile}.bz2")
        else:
            print(f"Wrote trace to {tracefile}")

if __name__ == '__main__':
    main()
