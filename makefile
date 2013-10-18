FLAGS	         =
FFLAGS	         =
CPPFLAGS         =
FPPFLAGS         =
MANSEC           = KSP
CLEANFILES       = rhs.vtk solution.vtk
NP               = 1

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_KSP_LIB}
	${RM} ex1.o

gmres_runner: gmres_runner.o  chkopts
	-${CLINKER} -o gmres_runner gmres_runner.o  ${PETSC_KSP_LIB}
	${RM} gmres_runner.o
