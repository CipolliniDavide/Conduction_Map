
# Set do_fit to 0 if only want to run plots
# Set do_fit to 1 if running fitting procedure is desired

do_fit=0

python ./counter_example_ibw.py --do_fit $do_fit --save_path ./Output/
