"""P1(b): repeat the (a) depth and attention probes on each variant's best-LR run.

    uv run python -m experiments.a4_stability.p1b_measure

Best LR = lowest final validation loss among the five supplied rates (W&B sweep).
Same checkpoints (init, step 100, final) and held-out sequences as P1(a).
Writes results/a4_stability/p1b/measurements/<variant>.json.
"""
from pathlib import Path

OUTPUT = Path('results/a4_stability/p1b/measurements')
BEST_LR = {
    'd8': .003,
    'd8-no-residual': .00075,  # all five LRs end at 6.8746-6.8747: flat, the choice does not matter
    'd8-no-prenorm': .001,
    'd8-no-qknorm': .003,
    'd8-no-rope': .003,
    'd8-no-final-norm': .003,
    'd8-postnorm': .001,
    'd8-gated-residual': .003,
    'd8-gated-attention': .006,
    'd8-gated-norm': .00225,  # lowest supplied rate: at the edge of the range
}


def measure(variant):
    from . import p1a_attention, p1a_depth
    lr = BEST_LR[variant]
    return dict(variant=variant, lr=lr,
                depth=p1a_depth.measure(variant=variant, lr=lr),
                attention=p1a_attention.measure(variant=variant, lr=lr))


if __name__ == '__main__':
    import modal
    from modal_utils import app, build_image, VOLUME_MOUNTS, MODAL_ENVIRONMENT
    from .diagnostics import save
    probe = app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu='H100', timeout=1800)(measure)
    with modal.enable_output(), app.run(environment_name=MODAL_ENVIRONMENT):
        for result in probe.map(list(BEST_LR)):
            path = OUTPUT / f"{result['variant']}.json"
            save(path, result)
            print(path)
