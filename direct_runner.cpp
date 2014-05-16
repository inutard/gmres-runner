static char help[] = "Solves a given linear system with GMRES and the ildl preconditioner.\n\n";

#include "matrix-factor-master/source/solver.h"

#include <iostream>
#include <cassert>
#include <cstring>
#include <petscksp.h>

#undef __FUNCT__
#define __FUNCT__ "main"

bool CreateMatrix(Mat& M, PetscInt n) {
	PetscErrorCode ierr;
	// initialize PETSC matrix, and check for errors at every step.
	ierr = MatCreate(PETSC_COMM_WORLD,&M); CHKERRQ(ierr);
	ierr = MatSetSizes(M,PETSC_DECIDE,PETSC_DECIDE,n,n); CHKERRQ(ierr);
	ierr = MatSetFromOptions(M); CHKERRQ(ierr);
	ierr = MatSetUp(M); CHKERRQ(ierr);

	return true;
}

// converts sym-ildl matrix A into PETSC matrix M
bool ConvertMatrix(const lilc_matrix<double>& A, Mat& M) {
	PetscInt n = A.n_rows();
	PetscErrorCode ierr;
	CreateMatrix(M, n);
	for (int i = 0; i < n; i++) {
		if (A.m_idx[i][0] != i) {
			ierr = MatSetValue(M,i,i,0,INSERT_VALUES); CHKERRQ(ierr);
		}

		for (int j = 0; j < (int) A.m_x[i].size(); j++) {
			PetscInt col = i;
			PetscInt row = A.m_idx[col][j];
			PetscScalar value = A.m_x[col][j];
			ierr = MatSetValue(M,row,col,value,INSERT_VALUES); CHKERRQ(ierr);
			// this option will not be needed after we add support for symmetric matrix
			// conversion
			// future-TODO: add symmetric matrix conversion.
			ierr = MatSetValue(M,col,row,value,INSERT_VALUES); CHKERRQ(ierr);
		}
	}

	ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	return true;
}

int main(int argc, char* args[]) {
	if (argc < 2) {
		std::cout << "Filename needed." << std::endl;
		std::cout << "Program usage: ./minres_runner [in.mtx]" << std::endl;
		std::cout << "Sample usage: ./minres_runner test_matrices/testmat1.mtx" << std::endl;
		return 0;
	}

	solver<double> solv;
	solv.load(args[1]);
	
	/*
	   Begin filling PETSC matrices with our preconditioner.
	*/
	
	// Define needed variables
	Vec            x, b, u;      /* approx solution, RHS, exact solution */
	Mat            A;            /* linear system matrix */
	KSP            ksp;         /* linear solver context */
	PetscReal      norm,tolerance=1.e-7;  /* norm of solution error */
	PetscErrorCode ierr;
	PetscInt       n,its;
	PetscMPIInt    size;
	PC             pc;                      /* PC context */
	
	// Initialize PETSC options
	PetscInitialize(&argc,&args,(char*)0,help);
	ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
	if (size != 1) SETERRQ(PETSC_COMM_WORLD,1,"Uniprocessor mode on.");
	
	/*
	   Create matrix.  When using MatCreate(), the matrix format can
	   be specified at runtime.

	   note: preallocate memory later for performance.
	 */

	// Convert our sym-ildl matrices to PETSC matrices.
	// future-TODO: make convert matrix a function of the sym-ildl matrices instead,
	// so that we may make our package more accessible to researchers. also make
	// them automatically choose the right matrix types.
	
	ConvertMatrix(solv.A, A);
	n = solv.A.n_rows();	
	/*
	   Applying MINRES to the converted matrices.
	*/

	// first initialize the solution vectors and RHS
	// form one vector from scrate and duplicate the rest.
	ierr = VecCreate(PETSC_COMM_WORLD,&x); CHKERRQ(ierr);
	ierr = PetscObjectSetName((PetscObject) x, "Solution"); CHKERRQ(ierr);
	ierr = VecSetSizes(x,PETSC_DECIDE,n); CHKERRQ(ierr);
	ierr = VecSetFromOptions(x); CHKERRQ(ierr);
	ierr = VecDuplicate(x,&b); CHKERRQ(ierr);
	ierr = VecDuplicate(x,&u); CHKERRQ(ierr);
	
	// for now, we'll just test by setting a vector of all 1's on the LHS
	ierr = VecSet(u, PetscScalar(1.0) ); CHKERRQ(ierr);
	// also compute the RHS.
	ierr = MatMult(A,u,b); CHKERRQ(ierr);

	// initialize Kylov Subspace solver context
	ierr = KSPCreate(PETSC_COMM_WORLD,&ksp); CHKERRQ(ierr);

	// Give the solver its matrix as well as some preconditioning information
	ierr = KSPSetOperators(ksp,A,A,SAME_PRECONDITIONER); CHKERRQ(ierr);
	
	// for now, we dont use a preconditioner, since it takes a little work to
	// implement.
	// uncomment to use GMRES
	//ierr = KSPSetType(ksp, KSPMINRES); CHKERRQ(ierr);
	KSPGetPC(ksp,&pc);
	PCSetType(pc,PCLU);

	// finalize solver options
	ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
	// solve!
	ierr = KSPSolve(ksp,b,x); CHKERRQ(ierr);

	// view basic information about the solver
	PetscBool flg;
	ierr = PetscOptionsGetBool(NULL,"-nokspview",&flg,NULL); CHKERRQ(ierr);
	if (!flg) {
		ierr = KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
	}

	/*
	   Compute the error after convergence
	*/

	ierr = VecAXPY(x,PetscScalar(-1.0),u); CHKERRQ(ierr);
	ierr = VecNorm(x,NORM_2,&norm); CHKERRQ(ierr);
	ierr = KSPGetIterationNumber(ksp,&its); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"Done with error %G " 
			   "in %D iterations.\n",norm/sqrt(n),its); CHKERRQ(ierr);
	
	/*
	   Free work space.
	 */
	ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
	ierr = VecDestroy(&u); CHKERRQ(ierr);  ierr = VecDestroy(&x); CHKERRQ(ierr);
	ierr = VecDestroy(&b); CHKERRQ(ierr);
	ierr = MatDestroy(&A); CHKERRQ(ierr);

	ierr = PetscFinalize();
	return 0;
}

