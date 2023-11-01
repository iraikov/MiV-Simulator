from machinable import Component
from pydantic import BaseModel, Field, ConfigDict
from miv_simulator import config
from mpi4py import MPI
from miv_simulator.utils import io as io_utils, from_yaml
from typing import Dict


class H5Types(Component):
    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")

        cell_distributions: config.CellDistributions = Field("???")
        synapses: config.Synapses = Field("???")

    def config_from_file(self, filename: str) -> Dict:
        return from_yaml(filename)

    @property
    def output_filepath(self) -> str:
        return self.local_directory("neuro.h5")

    def __call__(self) -> None:
        if MPI.COMM_WORLD.rank == 0:
            io_utils.create_neural_h5(
                self.output_filepath,
                self.config.cell_distributions,
                self.config.synapses,
            )
        MPI.COMM_WORLD.barrier()
