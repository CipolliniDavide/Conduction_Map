# Conduction Map
This repository contains files to reproduce Fig 5 and S5 in Rieck, J. L., Cipollini, D., Salverda, M., Quinteros, C. P., Schomaker, L. R. B., & Noheda, B. (2022). Ferroelastic Domain Walls in BiFeO3 as Memristive Networks. In Advanced Intelligent Systems (Vol. 5, Issue 1). Wiley. https://doi.org/10.1002/aisy.202200292

Configure `Python 3.11.0rc1` or `python3.9` on Linux with pip env. Then install `requirements.txt`.

Run `run_counter_example_ibw.sh` with `do_fit=0` to reproduce the figures. 
Use `do_fit=1` to re-fit, overwrite [counterEx_ibw_.npy](Output%2FcounterEx_ibw_.npy), and reproduce the figures.

Raw data JR38_CoCr_6_70000.ibw included in this repository are the product of the experimental work of Jan Rieck and Prof. dr Beatriz Noheda.
