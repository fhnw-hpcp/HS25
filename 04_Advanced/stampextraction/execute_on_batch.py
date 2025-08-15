from collections import defaultdict
from pathlib import Path
import json
import time
from astropy.table import Table
import logging
import sys
import os

from stampextraction.vis_exposures import VisExposureFitsIO, VisExposureHDF5
from stampextraction.stamps import extract_exposure_stamp
from stampextraction.profiling import PROFILING_QUEUE as profiling_queue

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


datafiles = {}
datafiles["DET"] = "EUC_SHE_VIS-DET-1-1_0_20250701T085217.429173Z_09.10.fits"
datafiles["BKG"] = "EUC_SHE_VIS-BKG-1-1_0_20250701T085217.429307Z_09.10.fits"
datafiles["WGT"] = "EUC_SHE_VIS-WGT-1-1_0_20250701T085217.429294Z_09.10.fits"
datafiles["SEG"] = "EUC_SHE_EXP-RPJ-SEG_00_20250701T085301.676748Z_09.10.fits"
datafiles["MER"] = "EUC_SHE_MER-CAT_00_20250701T085255.628039Z_09.10.fits"
datafiles["HDF5"] = "image_data.hdf5"


def process_profiling(comm, file_type, sorting_type, tick, size):
    prof = defaultdict(float)
    # Process local profiling data
    while not profiling_queue.empty():
        tmp = profiling_queue.get_nowait()
        for key in tmp:
            prof[key] += tmp[key]
        prof["count"] += 1
    
    if comm is None:
        prof = [prof]
    else:
        prof = comm.gather(prof, root=0)

    if prof is None:
        return  # Rank > 0
    
    #process gathered profiling data
    all_prof = defaultdict(float)
    for tmp in prof:
        for key in tmp:
            all_prof[key] += tmp[key]
    
    for key, val in all_prof.items():
        if key != "count":
            all_prof[key] = val / all_prof["count"]
    all_prof["tick"] = tick
    all_prof["file_type"] = file_type
    all_prof["sorting_type"] = sorting_type

    with open(f"profiling/profiling_{file_type}_{sorting_type}_{size}.json", "a") as f:
        json.dump(all_prof, f)
        f.write("\n")


def extract_stamps(workdir, sorting_type, batch_number, file_type, comm=None, size=1):
    workdir = Path(workdir)

    with open(f"profiling/{sorting_type}_batches.json") as f:
        batches = json.load(f)

    # assume batch number starts at 0, and goes between 0 and nbatches-1
    if batch_number > len(batches)-1:
        raise ValueError(f"Batch number is out of range: {batch_number}")

    t = Table.read(workdir / datafiles["MER"], memmap=False)

    inds = batches[batch_number]

    batch_t = t[inds]

    # initialise exposure object, which is the IO-method agnostic class for accessing image data
    if file_type == "hdf5":
        exposure = VisExposureHDF5(workdir / datafiles["HDF5"])
    else:
        exposure = VisExposureFitsIO(
            workdir/datafiles["DET"], workdir/datafiles["BKG"], workdir/datafiles["WGT"], workdir/datafiles["SEG"]
        )

    # loop over objects in batch, extract stamps
    tick = 0
    for i, row in enumerate(batch_t):
        stamp = extract_exposure_stamp(exposure, row["RIGHT_ASCENSION"], row["DECLINATION"], size=400)
        # pretend we do something with the exposure stamp (e.g. this mimics compute)
        time.sleep(0.5)
        
        if i % 10 == 0 and i > 0:
            process_profiling(comm, file_type, sorting_type, tick, size)
            tick += 1


if __name__ == "__main__":

    # Set default values
    rank = 0
    comm = None
    file_type = "fits"
    sorting_type = "shuffled"
    size=1
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    except ImportError:
        print("Fallback to rank 0 and size 1 as mpi4py not available.")

    # Read from command line if provided
    if len(sys.argv) > 1:
        sorting_type = sys.argv[1]
    if len(sys.argv) > 2:
        file_type = sys.argv[2].lower()
    file_type = "hdf5" if file_type == "hdf5" else "fits"
    sorting_type = "shuffled" if sorting_type == "shuffled" else "sorted"
    if rank == 0:
        if os.path.exists(f"profiling/profiling_{file_type}_{sorting_type}_{size}.json"):
            os.remove(f"profiling/profiling_{file_type}_{sorting_type}_{size}.json")

    extract_stamps(
        "/shared-scratch/hpcp/data",
        sorting_type,
        batch_number=rank,
        file_type=file_type,
        comm=comm,
        size=size
    )
