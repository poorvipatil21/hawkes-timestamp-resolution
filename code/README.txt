Reproduce (about 10 minutes on 2 cores):
  pip install numpy scipy numba pandas matplotlib
  python3 run_exp.py      # main grid  -> ../results/synthetic.json
  python3 run_exp2.py     # fine grid  -> ../results/synthetic_fine.json
  python3 make_figs.py    # figures + tables
Real data (not run by the draft's author): python3 real_data_check.py trades.csv time ms
