# Tuning
##############

# medium
# bash /userspace/soa/slurm.sh whisper_tune \
python pub/whisper_tune.sh --model openai/whisper-medium --output-dir tmp/tuning/whisper-medium-rudevices

# large
# bash /userspace/soa/slurm.sh whisper_tune2 \
python pub/whisper_tune.sh --model openai/whisper-large-v3-turbo --output-dir tmp/tuning/whisper-turbo-rudevices

# Make predictions on test set
#######################################

# medium
# bash /userspace/soa/slurm.sh whisper_eval \
python pub/whisper_eval.py \
--base openai/whisper-medium \
--dataset-multi sova-rudevices-multivariant \
--dataset-single sova-rudevices-single-variant \
--tuned_pattern tmp/tuning/whisper-medium-rudevices/checkpoint-{step} \
--storage tmp/tuning_evals.db

# large
# bash /userspace/soa/slurm.sh whisper_eval2 \
python pub/whisper_eval.py \
--base openai/whisper-large-v3-turbo \
--dataset-multi sova-rudevices-multivariant \
--dataset-single sova-rudevices-single-variant \
--tuned_pattern tmp/tuning/whisper-turbo-rudevices/checkpoint-{step} \
--storage tmp/tuning_evals_2.db

# Evaluate predictions on test set
#######################################

# medium
python pub/whisper_plots.py \
--base openai/whisper-medium \
--dataset-multi sova-rudevices-multivariant \
--dataset-single sova-rudevices-single-variant \
--storage tmp/tuning_evals.db \
--cache tmp/tuning_evals_cache.db \
--output tmp/tuning_plot_medium.svg

# large
python pub/whisper_plots.py \
--base openai/whisper-large-v3-turbo \
--dataset-multi sova-rudevices-multivariant \
--dataset-single sova-rudevices-single-variant \
--storage tmp/tuning_evals_2.db \
--cache tmp/tuning_evals_cache_2.db \
--output tmp/tuning_plot_turbo.svg