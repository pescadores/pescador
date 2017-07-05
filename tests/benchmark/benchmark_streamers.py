"""benchmark_streamers is a test utility to generate benchmarks for
different streamer configurations. It compares a standard generator vs a plain
Streamer, and buffer_stream'd streamers of various buffer sizes.

This utility is primarily a regression test, with the additional functionality
of demonstrating potential weak points in the Pescador batching.
"""
import click
import colorama
import logging
# import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pescador
import tempfile
import time

logger = logging.getLogger("benchmark_streamers")


sample_shapes = [(1, 10), (1, 10, 79)]
n_samples = [100, 1000, 5000, 10000]
buffer_sizes = [8, 32, 128, 256]
# n_samples = [100, 1000, 5000, 10000, 20000, 25000, 50000]
# sample_shapes = [(1, 10), (1, 10, 79), (1, 3, 28, 28), (1, 6, 26, 26)]
# buffer_sizes = [8, 32, 128, 256]

n_to_test = 1000
regression_vector_path = os.path.join(os.path.dirname(__file__),
                                      "benchmark_vectors.json")


def statcode_colored(status_code, text=None):
    if status_code == 0:
        color = colorama.Fore.GREEN
        print_text = "PASS"
    else:
        color = colorama.Fore.RED
        print_text = "FAIL"

    if text:
        print_text = str(text) + ": " + print_text

    return "{}{}{}".format(
        color + colorama.Style.BRIGHT,
        print_text,
        colorama.Style.RESET_ALL)


def create_npy(n_samples, sample_shapes, filename):
    shape = list(sample_shapes)
    shape[0] = n_samples
    arrays_to_save = np.random.random(shape)

    np.save(filename, arrays_to_save)


def npy_gen(filename, mmap='c'):
    data_in = np.load(filename, mmap_mode=mmap)

    while True:
        i = np.random.randint(len(data_in))
        yield dict(X=np.array(data_in[i]))


def benchmark_one(n_samples, sample_shape):
    """Time sampling from a .npy with the passed configuration."""
    key_str = "{}_{}".format(n_samples,
                             "-".join([str(x) for x in sample_shape]))
    filename = "random_{}.npy".format(key_str)

    results = pd.Series(name=key_str)
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, filename)
        # Create the test file
        c0 = time.time()
        create_npy(n_samples, sample_shape, filepath)
        results['create_time'] = time.time() - c0
        logger.info("{} created in {}s".format(filename,
                                               results['create_time']))

        # Sample from it with no streamer
        sample_times = []
        s0 = time.time()
        for i, batch in enumerate(npy_gen(filepath)):
            sample_times.append(time.time() - s0)

            if i > n_to_test:
                break
            s0 = time.time()

        results['generator'] = np.mean(sample_times)
        logger.info("{} generator sampling average: {:.7f}s".format(
            filename, results['generator']))

        # Sample from it with a basic streamer
        streamer = pescador.Streamer(npy_gen(filepath))
        sample_times = []
        s0 = time.time()
        for sample in streamer.iterate(max_iter=n_to_test):
            sample_times.append(time.time() - s0)
            s0 = time.time()

        results['streamer'] = np.mean(sample_times)
        logger.info("{} streamer sampling average: {:.7f}s".format(
            filename, results['streamer']))

        # Run it with all the buffer sizes
        for buffer_size in buffer_sizes:
            # sample from it with a batch streamer
            buffered = pescador.buffer_stream(streamer, buffer_size)
            buffstream = pescador.Streamer(buffered)

            sample_times = []
            s0 = time.time()
            for batch in buffstream.iterate(max_iter=n_to_test):
                sample_times.append(time.time() - s0)
                s0 = time.time()

            results['buffer-{}'.format(buffer_size)] = np.mean(sample_times)
            results['buffer-{} (per sample)'.format(buffer_size)] = results[
                'buffer-{}'.format(buffer_size)] / buffer_size
            logger.info("{} buffered sampling average: {:.7f}s".format(
                filename, results['buffer-{}'.format(buffer_size)]))
            logger.info("{} buffered sampling per sample average: {:.7f}s"
                        .format(filename,
                                results['buffer-{} (per sample)'.format(
                                    buffer_size)]))
    return results


@click.command()
@click.option('--dump_plots', is_flag=True)
@click.option('-v', '--verbose', count=True)
def benchmark(dump_plots, verbose):
    global logger
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Run the analysis
    results_ls = []
    for n_samps in n_samples:
        for sample_shape in sample_shapes:
            results = benchmark_one(n_samps, sample_shape)
            results_ls.append(results)
    results_df = pd.concat(results_ls, axis=1)
    # Print the results
    print("* * * * Shape Dimensionality * * * *")
    for s in sample_shapes:
        print(s, ":", np.prod(s))
    print("* * * * Computed Results * * * *")
    print(results_df)

    # Check the result against a previous run
    success = True
    if os.path.exists(regression_vector_path):
        old_vectors = pd.read_json(regression_vector_path)

        # We don't need to compare create_time and generator because they
        #  don't indicate anything about the Pescador library itself.
        time_difference = (results_df - old_vectors).drop([
            "create_time", "generator"])

        time_diff_absolute = time_difference.loc[
            time_difference.index.map(lambda x: 'per sample' not in x)]
        time_diff_per_sample = time_difference.loc[
            time_difference.index.map(lambda x: 'per sample' in x)]

        # Check that the absolute times haven't increased by more than 10%
        #  TODO: 10%? what's a good number here.
        time_diff_errors = (time_diff_absolute / old_vectors.loc[
            time_diff_absolute.index]) > .1
        if np.any(time_diff_errors):
            success = False
            logger.error("Time diff fails; change > 10%:")
            logger.error(time_diff_absolute[time_diff_errors])

        # Check the per-sample comparison
        per_sample_errors = (time_diff_per_sample / old_vectors.loc[
            time_diff_per_sample.index] > .05)
        if np.any(per_sample_errors):
            success = False
            logger.error("Per-sample errors: change > 5%:")
            logger.error(time_diff_per_sample[per_sample_errors])

    # Save the new results if it's good
    if success:
        results_df.to_json(regression_vector_path)

    statcode = success is not True
    statcode_colored(statcode)
    return statcode


if __name__ == "__main__":
    benchmark()
