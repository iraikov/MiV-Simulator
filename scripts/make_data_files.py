import os, sys
import h5py, pathlib
from dentate.utils import viewitems

def h5_copy_dataset(f_src, f_dst, dset_path):
    print(f"Copying {dset_path} from {f_src} to {f_dst} ...")
    target_path = str(pathlib.Path(dset_path).parent)
    f_src.copy(f_src[dset_path], f_dst[target_path])

h5types_file = 'MiV_h5types.h5'

MiV_populations = ["PYR", "OLM", "PVBC"]
MiV_IN_populations = ["OLM", "PVBC"]
#MiV_EXT_populations = ["EXT"]

MiV_cells_file = "MiV_Cells_Microcircuit_20220404.h5"
MiV_connections_file = "MiV_Connections_Microcircuit_20220404.h5"

MiV_coordinate_file  = "Microcircuit_coords.h5"

MiV_PYR_forest_file = "PYR_forest.h5"
MiV_PVBC_forest_file = "PVBC_forest.h5"
MiV_OLM_forest_file = "OLM_forest.h5"

MiV_connectivity_file = "Microcircuit_connections.h5"

connectivity_files = {
    'PYR': MiV_connectivity_file,
    'PVBC': MiV_connectivity_file,
    'OLM': MiV_connectivity_file,
}


coordinate_files = {
     'PYR':   MiV_coordinate_file,
     'PVBC':  MiV_coordinate_file,
     'OLM':   MiV_coordinate_file,
}

distances_ns = 'Arc Distances'
coordinate_ns = 'Generated Coordinates'
coordinate_namespaces = {
     'PYR': coordinate_ns,
     'OLM':  coordinate_ns,
     'PVBC':  coordinate_ns,
}
    


forest_files = {
     'PYR': MiV_PYR_forest_file,
     'PVBC': MiV_PVBC_forest_file,
     'OLM': MiV_OLM_forest_file,
}

forest_syns_files = {
     'PYR': MiV_PYR_forest_file,
     'PVBC': MiV_PVBC_forest_file,
     'OLM': MiV_OLM_forest_file,
}


## Creates H5Types entries
with h5py.File(MiV_cells_file, 'w') as f:
    input_file  = h5py.File(h5types_file,'r')
    h5_copy_dataset(input_file, f, '/H5Types')
    input_file.close()

## Creates coordinates entries
with h5py.File(MiV_cells_file, 'a') as f_dst:

    grp = f_dst.create_group("Populations")
                
    for p in MiV_populations:
        grp.create_group(p)

    for p in MiV_populations:
        coords_file = coordinate_files[p]
        coords_ns   = coordinate_namespaces[p]
        coords_dset_path = f"/Populations/{p}/{coords_ns}"
        distances_dset_path = f"/Populations/{p}/Arc Distances"
        with h5py.File(coords_file, 'r') as f_src:
            h5_copy_dataset(f_src, f_dst, coords_dset_path)
            h5_copy_dataset(f_src, f_dst, distances_dset_path)


## Creates forest entries and synapse attributes
for p in MiV_populations:
    if p in forest_files:
        forest_file = forest_files[p]
        forest_syns_file = forest_syns_files[p]
        forest_dset_path = f"/Populations/{p}/Trees"
        forest_syns_dset_path = f"/Populations/{p}/Synapse Attributes"
        cmd = f"h5copy -p -s '{forest_dset_path}' -d '{forest_dset_path}' " \
              f"-i {forest_file} -o {MiV_cells_file}"
        print(cmd)
        os.system(cmd)
        cmd = f"h5copy -p -s '{forest_syns_dset_path}' -d '{forest_syns_dset_path}' " \
              f"-i {forest_syns_file} -o {MiV_cells_file}"
        print(cmd)
        os.system(cmd)

                

## Creates vector stimulus entries
#for (vecstim_ns, vecstim_file) in viewitems(vecstim_dict):
#    for p in MiV_EXT_populations+['PYR']:
#        vecstim_dset_path = f"/Populations/{p}/{vecstim_ns}"
#        cmd = f"h5copy -p -s '{vecstim_dset_path}' -d '{vecstim_dset_path}' " \
#              f"-i {vecstim_file} -o {MiV_cells_file}"
#        print(cmd)
#        os.system(cmd)


with h5py.File(MiV_connections_file, 'w') as f:
    input_file  = h5py.File(h5types_file,'r')
    h5_copy_dataset(input_file, f, '/H5Types')
    input_file.close()

## Creates connectivity entries
for p in MiV_populations:
     if p in connectivity_files:
         connectivity_file = connectivity_files[p]
         projection_dset_path = f"/Projections/{p}"
         cmd = f'h5copy -p -s {projection_dset_path} -d {projection_dset_path} ' \
               f'-i {connectivity_file} -o {MiV_connections_file}'
         print(cmd)
         os.system(cmd)

