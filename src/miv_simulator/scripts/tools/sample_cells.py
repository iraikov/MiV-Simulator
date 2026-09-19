"""
Samples cells from a text file of gids, and writes a neuroh5 selection
file with the data needed to instantiate the selected cells.
"""

import os
import sys

import click
from miv_simulator import utils
from miv_simulator.env import Env
from miv_simulator.utils import io as io_utils
from mpi4py import MPI
from neuroh5.io import (
    append_cell_attributes,
    read_cell_attribute_selection,
    read_population_ranges,
)


def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    sys.stdout.flush()
    sys.stderr.flush()
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)


sys_excepthook = sys.excepthook
sys.excepthook = mpi_excepthook


@click.command()
@click.option("--config", "-c", required=True, type=str)
@click.option(
    "--config-prefix",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="config",
    help="path to directory containing network config files",
)
@click.option(
    "--dataset-prefix",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="path to directory containing required neuroh5 data files",
)
@click.option("--distances-namespace", "-n", type=str, default="Arc Distances")
@click.option(
    "--spike-input-path",
    required=False,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="path to file for input spikes",
)
@click.option(
    "--spike-input-namespace",
    required=False,
    multiple=True,
    type=str,
    help="namespace for input spikes",
)
@click.option(
    "--spike-input-attr",
    required=False,
    type=str,
    help="attribute name for input spikes",
)
@click.option(
    "--input-features-path",
    required=False,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="path to file containing input features",
)
@click.option(
    "--input-features-namespaces",
    type=str,
    required=False,
    multiple=True,
    default=[],
    help="namespaces containing input features",
)
@click.option(
    "--selection-path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="text file containing the gids of the cells to sample",
)
@click.option(
    "--output-path",
    "-o",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option("--io-size", type=int, default=-1)
@click.option("--verbose", "-v", is_flag=True)
def main(
    config,
    config_prefix,
    dataset_prefix,
    distances_namespace,
    spike_input_path,
    spike_input_namespace,
    spike_input_attr,
    input_features_namespaces,
    input_features_path,
    selection_path,
    output_path,
    io_size,
    verbose,
):
    """
    Instantiates a subset of the network consisting of the cells whose gids
    are listed in the given selection file, and writes a neuroh5 selection
    file containing the coordinates, connections, spike trains, and input
    features of the selected cells.
    """

    utils.config_logging(verbose)
    logger = utils.get_script_logger(os.path.basename(__file__))

    comm = MPI.COMM_WORLD
    rank = comm.rank
    if io_size == -1:
        io_size = comm.size

    env = Env(
        comm=comm,
        config=config,
        dataset_prefix=dataset_prefix,
        results_path=output_path,
        spike_input_path=spike_input_path,
        spike_input_namespaces=spike_input_namespace,
        spike_input_attr=spike_input_attr,
        io_size=io_size,
        config_prefix=config_prefix,
    )

    selection = []
    f = open(selection_path)
    for line in f.readlines():
        selection.append(int(line))
    f.close()
    selection = set(selection)

    pop_ranges, pop_size = read_population_ranges(env.connectivity_file_path, comm=comm)

    distance_U_dict, distance_V_dict, range_U_dict, range_V_dict = (
        io_utils.read_soma_distances(
            env,
            comm,
            distances_namespace,
            gid_filter=lambda gid: gid in selection,
            populations=pop_ranges,
        )
    )

    selection_dict = {}
    if rank == 0:
        for population in pop_ranges:
            if population not in distance_U_dict:
                continue
            distance_U = distance_U_dict[population]
            selection_dict[population] = {gid for gid in distance_U if gid in selection}

    env.comm.barrier()

    write_selection_file_path = f"{env.results_path}/{env.modelName}_selection.h5"

    if rank == 0:
        io_utils.mkout(env, write_selection_file_path)
    env.comm.barrier()
    selection_dict = env.comm.bcast(selection_dict, root=0)
    env.cell_selection = selection_dict
    io_utils.write_cell_selection(env, write_selection_file_path)
    input_selection = io_utils.write_connection_selection(
        env, write_selection_file_path
    )

    has_spike_input = (spike_input_path is not None) or (len(spike_input_namespace) > 0)
    if has_spike_input:
        io_utils.write_input_cell_selection(
            env, input_selection, write_selection_file_path
        )

    if input_features_path:
        for this_input_features_namespace in sorted(input_features_namespaces):
            for population in sorted(input_selection):
                logger.info(
                    f"Extracting input features {this_input_features_namespace} for population {population}..."
                )
                it = read_cell_attribute_selection(
                    input_features_path,
                    population,
                    namespace=this_input_features_namespace,
                    selection=input_selection[population],
                    comm=env.comm,
                )
                output_features_dict = {
                    cell_gid: cell_features_dict for cell_gid, cell_features_dict in it
                }
                append_cell_attributes(
                    write_selection_file_path,
                    population,
                    output_features_dict,
                    namespace=this_input_features_namespace,
                    io_size=io_size,
                    comm=env.comm,
                )
    env.comm.barrier()
