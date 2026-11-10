"""
Run DP-ICL (PATE + Gaussian RNM) with a local Ollama model.

Usage:
    cd /home/claude/DP-ICL/Generate_seperate_reponses/mutltiple_choice
    python run_ollama.py prompting_configs/ollama-pate-sst2.yaml

The script is equivalent to src/prompting/main.py but with cleaner output
and no GPT-2/GPT-3 dependency.
"""

import sys
import os

# Add src to path so `import prompting` works
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from omegaconf import OmegaConf
from prompting import MODELS, PROCESSORS, STRATEGIES
from prompting.misc_utils import seed_everything, setup_logging, setup_output_dir

import logging
logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <config.yaml> [overrides...]")
        sys.exit(1)

    base_conf  = OmegaConf.load("prompting_configs/base.yaml")
    user_confs = [OmegaConf.load(f) for f in sys.argv[1:]]
    conf       = OmegaConf.merge(base_conf, *user_confs)

    setup_output_dir(conf)
    setup_logging(conf)
    seed_everything(conf.seed)

    logger.info(OmegaConf.to_yaml(conf))

    # Build processor, model, strategy
    processor  = PROCESSORS[conf.task](seed=conf.seed, **conf.processor_kwargs)
    model_cls  = MODELS[conf.model]
    model      = model_cls(conf.model_name, **processor.model_kwargs, **conf.get("model_kwargs", {}))
    strategy   = STRATEGIES[conf.strategy](conf)

    for shot in sorted(conf.shots):
        logger.info(f"Running with shot={shot}")
        results = strategy(processor, model, shot)
        logger.info(f"Results: {results}")


if __name__ == "__main__":
    main()