import runpy
import sys


def _run_module(dotted: str) -> None:
    mod_path = dotted.replace(".", "/") + ".py"
    sys.argv = [mod_path] + sys.argv[1:]
    runpy.run_path(mod_path, run_name="__main__")
