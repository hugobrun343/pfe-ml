"""Step 7: Supercharge training (trainer args examples)."""

import lightning as L


def example_multi_gpu():
    trainer = L.Trainer(devices=4, accelerator="gpu")
    return trainer


def example_deepspeed():
    trainer = L.Trainer(devices=4, accelerator="gpu", strategy="deepspeed_stage_2", precision=16)
    return trainer


def example_flags():
    trainer = L.Trainer(max_epochs=10, min_epochs=5, overfit_batches=1)
    return trainer


if __name__ == "__main__":
    print("Examples created: multi_gpu, deepspeed, flags")
