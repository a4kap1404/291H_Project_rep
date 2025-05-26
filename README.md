current state as of 12:00am May 26th,

Description: This is a diffusion model implementaion heavily based on the paper "Chip Placement with Diffusion Models".

Currently the code implementation attempts to mirror the many aspects in the paper, but the code itself does not stem the paper we could not find any associated repository.

Note: "initial" below really means working, but not neccesarily robust in performance.

initial model, training, and sampling code is 95% finished, but having shape issues so cant currently run, abd thus cant current train.
- model code: model.py (has bugs)
- training code: train.py (has bugs)
- placement: place.py (very incomplete)

initial synthetic data generator appears to be complete, but effectiveness cannot be evaluted without model.
- see datagen.py
