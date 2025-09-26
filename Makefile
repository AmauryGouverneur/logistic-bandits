PY=python
PIP=pip
REQ=requirements.txt
SCRIPT=logistic_ts_persistent.py

.PHONY: deps run ci clean gpuinfo

deps:
	$(PY) -m pip install -U pip
	$(PIP) install -r $(REQ)

run:
	$(PY) $(SCRIPT)

ci:
	$(PY) - <<'PY'
from logistic_ts_persistent import run_logistic_bandits_TS_exp, mean_and_ci
avg, runs = run_logistic_bandits_TS_exp(d=10, beta=2.0, T=200, num_exp=50, progress=True)
mean, lo, hi = mean_and_ci(runs)
print('Mean[0:5]=', mean[:5])
print('CI half-width at t=199:', float((hi[-1]-lo[-1])/2))
PY

clean:
	rm -rf __pycache__ *.png results_experiments

gpuinfo:
	nvidia-smi || true
	$(PY) - <<'PY'
import torch
print('Torch:', torch.__version__)
print('CUDA available?', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
PY