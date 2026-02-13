for N in alignment_solving alignment remapping streaming_histogram streaming tuning_plot_medium tuning_plot_turbo; do
    inkscape $N.svg --export-filename=pdf_plots/$N.pdf
done