# nested_torch — common tasks. Run `make install` first.
# PYTHONPATH=. lets these work even before an editable install.
PY = PYTHONPATH=. python3

.PHONY: install install-viz test demo rl figure clean

install:        ## editable install (+ torch, numpy)
	pip install -e .

install-viz:    ## editable install with plotting + test extras
	pip install -e ".[viz,dev]"

test:           ## run both test suites
	$(PY) tests_pytorch/test_nested_torch.py
	$(PY) tests_pytorch/test_rl_and_eeg.py

demo:           ## the two headline demos (update profile + forgetting)
	$(PY) examples_pytorch/demo_multifrequency.py
	$(PY) examples_pytorch/cl_forgetting_pretrained.py

rl:             ## train the RL plasticity controller
	$(PY) examples_pytorch/rl_controller.py

figure:         ## regenerate results/nested_torch_poc.png (needs matplotlib)
	$(PY) examples_pytorch/make_figure.py

clean:          ## remove __pycache__
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
