import argparse

from transport_bridge.study import fit, test

parser = argparse.ArgumentParser()
parser.add_argument("phase", choices=["fit", "test"])
parser.add_argument("--model", default="gpt2-medium", choices=["gpt2-medium", "gpt2-small"])
parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
args = parser.parse_args()
(fit if args.phase == "fit" else test)(args.model, args.device)
