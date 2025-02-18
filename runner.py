from pathlib import Path

import comet_ml  # noqa
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from vital.runner import VitalRunner
from contour_uncertainty.utils.path import get_nonexistent_path


class Runner(VitalRunner):
    """Abstract runner that runs the main training/val loop, etc. using Lightning Trainer."""

    @classmethod
    def pre_run_routine(cls) -> None:
        """Sets-up the environment before running the training/testing."""
        super().pre_run_routine()
        OmegaConf.register_new_resolver(
            "labels",
            lambda x: '-' + '-'.join([n.lower() for n in x if n != 'bg']) if x is not None and len(x) != 4 else ''
        )
        OmegaConf.register_new_resolver(
            "frac", lambda x: int(x * 100)
        )

        OmegaConf.register_new_resolver(
            "if", lambda condition, flip, option1, option2: option1 if bool(condition) == bool(flip) else option2
        )

    @staticmethod
    @hydra.main(version_base=None, config_path="config", config_name="default.yaml")
    def run_system(cfg: DictConfig) -> None:
        """Handles the training and evaluation of a model.

        Redefined to add @hydra.main decorator with correct config_path and config_name
        """
        # print(OmegaConf.to_yaml(cfg, resolve=True))
        # # print("NAME", cfg.best_model_save_path)
        # # print("NAME", cfg.weights)
        # # print("NAME", cfg.name)
        # exit(0)
        import sys
        args = sys.argv[1:]
        result = ''
        for arg in args:
            result += arg + ' '
        with open_dict(cfg):
            cfg.command_str = result
        if cfg.get("best_model_save_path", None):
            best_model_path = Path(cfg.best_model_save_path)
            best_model_path.parent.mkdir(exist_ok=True)
            if cfg.train:
                cfg.best_model_save_path = get_nonexistent_path(cfg.best_model_save_path)
        return VitalRunner.run_system(cfg)


if __name__ == "__main__":
    Runner.main()
