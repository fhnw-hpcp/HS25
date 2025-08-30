#define _GNU_SOURCE  
#include <stdio.h>
#include <omp.h>
#include <mpi.h>
#include <sched.h>
//#include "getcpuid.h"

int main(int argc, char* argv[]) {
  int numprocs, rank, namelen;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int iam = 0, np = 1;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Get_processor_name(processor_name, &namelen);

  #pragma omp parallel default(shared) private(iam, np)
  {
    // np = omp_get_num_threads();
    // iam = omp_get_thread_num();
    // int cpu_id = sched_getcpu();
    // printf("Hello from %s: thread_id(%d) #threads(%d) rank_id(%d) #ranks(%d) cpu_id(%d)\n",
    //        processor_name, iam, np, rank, numprocs, cpu_id);
    printf("Hello from %s: thread_id(%d) #threads(%d) rank_id(%d) #ranks(%d) cpu_id(%d)\n",
           processor_name, omp_get_thread_num()+1, omp_get_num_threads(), rank, numprocs, sched_getcpu());
  }

  MPI_Finalize();
}
