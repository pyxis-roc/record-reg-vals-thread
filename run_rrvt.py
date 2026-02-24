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

def main():
    import argparse
    p = argparse.ArgumentParser(description="Trace a CUDA program with register values")
    p.add_argument("--trace", help="Trace file")
    p.add_argument("--instr-before", help="Instrument before execution", action="store_true")
    p.add_argument("--warp", help="Warp to trace", default=DEFAULT_WARP)
    p.add_argument("--sopath", help="Path to record_reg_vals_thread.so", default=None, type=Path)
    p.add_argument("--use-ld-preload", help="Use LD_PRELOAD instead of CUDA_INJECTION64_PATH", action="store_true")

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

    print(" ".join([f"%s=%s" % (k, v) for k, v in env.items()]), shlex.join(cmdline))

    try:
        runenv = dict(os.environ)
        runenv.update(env)

        subprocess.run(cmdline, env=env, check=True)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(e)
        sys.exit(1)

if __name__ == '__main__':
    main()
