#!/usr/bin/env python3
import os
import sys

import click
import numpy as np
from mpi4py import MPI
from neuroh5.io import read_population_ranges

from miv_simulator.env import Env
from miv_simulator.input_features import (
    ConstantModality,
    FeatureSpace,
    generate_input_features,
)
from miv_simulator.utils import list_find


@click.command()
@click.option("--config", required=True, type=str)
@click.option(
    "--config-prefix",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="config",
)
@click.option(
    "--coords-path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--distances-namespace", "-n", type=str, default="Arc Distances")
@click.option(
    "--output-path",
    type=click.Path(file_okay=True, dir_okay=False),
    default=None,
)
@click.option("--arena-id", type=str, default="A")
@click.option("--populations", "-p", type=str, multiple=True, required=True)
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--write-size", type=int, default=10000)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--debug", is_flag=True)
@click.option("--debug-count", type=int, default=10)
@click.option("--dry-run", is_flag=True)
def main(
    config,
    config_prefix,
    coords_path,
    distances_namespace,
    output_path,
    arena_id,
    populations,
    io_size,
    chunk_size,
    value_chunk_size,
    cache_size,
    write_size,
    verbose,
    debug,
    debug_count,
    dry_run,
):
    env = Env(config=config, config_prefix=config_prefix)
    comm = MPI.COMM_WORLD

    population_ranges = read_population_ranges(coords_path, comm)[0]

    selectivity_type = env.selectivity_types.get("constant")
    if selectivity_type is None:
        raise RuntimeError(
            "generate-input-features: 'constant' selectivity type is not defined "
            "in the 'Input Selectivity Types' section of the model definitions"
        )

    feature_space = FeatureSpace(name="Input Features")
    modality = ConstantModality()
    feature_space.register_modality(modality)

    local_random = np.random.RandomState()

    for population_name in populations:
        if population_name not in population_ranges:
            raise RuntimeError(
                f"generate-input-features: population {population_name} not found "
                f"in coords_path {coords_path}"
            )
        pop_start, pop_size = population_ranges[population_name]

        try:
            peak_rate = float(
                env.stimulus_config["Peak Rate"][population_name][selectivity_type]
            )
        except KeyError as e:
            raise RuntimeError(
                "generate-input-features: no 'Peak Rate' configured for population "
                f"'{population_name}', selectivity type 'constant', in the Stimulus "
                "configuration"
            ) from e

        population = feature_space.create_population(
            name=population_name,
            modality_name=modality.name,
            n_features=pop_size,
            encoding_distribution={
                "feature_type": "linear_rate",
                "peak_rate": peak_rate,
            },
            start_gid=pop_start,
            local_random=local_random,
            rank=comm.rank,
            size=comm.size,
        )

        generate_input_features(
            env,
            population,
            coords_path,
            distances_namespace,
            output_path,
            f"Constant Selectivity {arena_id}",
            io_size,
            chunk_size,
            value_chunk_size,
            cache_size,
            write_size,
            dry_run,
            verbose,
            debug,
            debug_count,
        )


if __name__ == "__main__":
    main(
        args=sys.argv[
            (
                list_find(
                    lambda x: os.path.basename(x) == os.path.basename(__file__),
                    sys.argv,
                )
                + 1
            ) :
        ],
        standalone_mode=False,
    )
