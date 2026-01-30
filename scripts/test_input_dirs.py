import numbers
import pyquickbench

input_dir = "/mnt/e/test/pyquickbench_tests/comp_KV"
# output_dir = "/mnt/e/test/pyquickbench_benchs/bench"

pyquickbench.compare_app.CLI.create_dirwise_benchmark(input_dir, n_out = 4)
