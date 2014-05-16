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

/* Define context for user-provided preconditioner */
typedef struct {
	Mat M;
} LdlPC;

/* Declare routines for user-provided preconditioner */
extern PetscErrorCode LdlPCCreate(LdlPC**);
extern PetscErrorCode LdlPCSetUp(PC,Mat,Vec);
extern PetscErrorCode LdlPCApply(PC,Vec x,Vec y);
extern PetscErrorCode LdlPCDestroy(PC);

int main(int argc, char* args[]) {
	if (argc < 2) {
		std::cout << "Filename needed." << std::endl;
		std::cout << "Program usage: ./gmres_runner [in.mtx]" << std::endl;
		std::cout << "Sample usage: ./gmres_runner test_matrices/testmat1.mtx" << std::endl;
		return 0;
	}

	solver<double> solv;
	solv.load(args[1]);
	
	// compute preconditioners with sym-ildl, using fill_factor, and tolerance as the parameters
	// and AMD reordering by default.
	double fill_factor = 1.0, tol = 0.001;
	solv.solve(fill_factor, tol);
	
	/*
	   Begin filling PETSC matrices with our preconditioner.
	*/
	
	// Define needed variables
	Vec            x, b, u;      /* approx solution, RHS, exact solution */
	Mat            A, L, D;            /* linear system matrix */
	KSP            ksp;         /* linear solver context */
	PC             left_pc, right_pc;           /* preconditioner context */
	LdlPC  *shell;    /* user-defined preconditioner context */
	PetscReal      norm,tolerance=1.e-14;  /* norm of solution error */
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
	
	//cout << "creating matrix A... " << endl;
	ConvertMatrix(solv.A, A);
	MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE);
	//cout << "creating matrix L... " << endl;
	ConvertMatrix(solv.L, L);
	
	//cout << "creating matrix D... " << endl;
	//fill the diagonal matrix separately, since it uses a different storage scheme
	n = solv.D.n_rows();
	CreateMatrix(D, n);
	for (int i = 0; i < n;) {
		bool has_offdiag = (solv.D.block_size(i) == 2);
		PetscInt row = i, col = i;
		PetscScalar value;
		int next_i = i+1;
		if (has_offdiag) {
			PetscScalar value = solv.D.off_diagonal(i);
			ierr = MatSetValue(D,row+1,col,value,INSERT_VALUES); CHKERRQ(ierr);
			ierr = MatSetValue(D,row,col+1,value,INSERT_VALUES); CHKERRQ(ierr);
			next_i++;
		}
		value = solv.D[i];
		ierr = MatSetValue(D,row,col,value,INSERT_VALUES); CHKERRQ(ierr);
		
		i = next_i;
	}
	// finish assembling the matrix.
	ierr = MatAssemblyBegin(D,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(D,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	
	//ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	//ierr = MatView(L,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	//ierr = MatView(D,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	
	/*
	   Applying GMRES to the converted matrices.
	*/

	// first initialize the solution vectors and RHS
	// form one vector from scrate and duplicate the rest.
	ierr = VecCreate(PETSC_COMM_WORLD,&x); CHKERRQ(ierr);
	ierr = PetscObjectSetName((PetscObject) x, "Solution"); CHKERRQ(ierr);
	ierr = VecSetSizes(x,PETSC_DECIDE,n); CHKERRQ(ierr);
	ierr = VecSetFromOptions(x); CHKERRQ(ierr);
	ierr = VecDuplicate(x,&b); CHKERRQ(ierr);
	ierr = VecDuplicate(x,&u); CHKERRQ(ierr);
	
	// for now, we'll just test by setting a vector of all 1's on the RHS
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
	//ierr = KSPSetType(&ksp, KSPGMRES); CHKERRQ(ierr);
	KSPGetPC(ksp,&pc);
	//PCSetType(pc,PCILU);
	//ierr = KSPSetTolerances(ksp,tolerance,PETSC_DEFAULT,
	//		        PETSC_DEFAULT,PETSC_DEFAULT); CHKERRQ(ierr);
	
	
	// set preconditoners
	ierr = KSPGetPC(ksp, &left_pc); CHKERRQ(ierr);

	// indicate to PETSc that we're using a "shell" preconditioner
	ierr = PCSetType(left_pc,PCSHELL);CHKERRQ(ierr);

	// create a context for the user-defined preconditioner; this
	// context can be used to contain any application-specific data
	ierr = LdlPCCreate(&shell); CHKERRQ(ierr);

	// set the user-defined routine for applying the preconditioner
	ierr = PCShellSetApply(left_pc, LdlPCApply);CHKERRQ(ierr);
	ierr = PCShellSetContext(left_pc,shell); CHKERRQ(ierr);

	// set user-defined function to free objects used by custom preconditioner
	ierr = PCShellSetDestroy(left_pc, LdlPCDestroy); CHKERRQ(ierr);

	// set a name for the preconditioner, used for PCView()
	ierr = PCShellSetName(left_pc,"LdlPreconditioner"); CHKERRQ(ierr);

	// do any setup required for the preconditioner
	ierr = LdlPCSetUp(pc,A,x); CHKERRQ(ierr);

	// set preconditioner side
	ierr = KSPSetPCSide(ksp, PC_RIGHT);
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
	ierr = PetscPrintf(PETSC_COMM_WORLD,"Done with relative residual %G " 
			   "in %D iterations.\n",norm,its); CHKERRQ(ierr);
	
	/*
	   Free work space.
	 */
	ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
	ierr = VecDestroy(&u); CHKERRQ(ierr);  ierr = VecDestroy(&x); CHKERRQ(ierr);
	ierr = VecDestroy(&b); CHKERRQ(ierr);
	ierr = MatDestroy(&A); CHKERRQ(ierr); ierr = MatDestroy(&L); CHKERRQ(ierr);
	ierr = MatDestroy(&D); CHKERRQ(ierr);

	ierr = PetscFinalize();
	return 0;
}

// creates a user-defined ldl preconditioner
PetscErrorCode LdlPCCreate(LdlPC **shell) {
	LdlPC *newctx;
	PetscErrorCode ierr;

	//allocates memory for preconditioner.
	ierr = PetscNew(LdlPC, &newctx); CHKERRQ(ierr);
	*shell = newctx;
	return 0;
}

// setup the precoditioners and allocate memory for preconditioning matrix
PetscErrorCode LdlPCSetUp(PC pc, Mat pmat, Vec x) {
	LdlPC *shell;
	Mat M;
	PetscErrorCode ierr;

	// copy over preconditioning matrix, which was assumed to be precalculated.
	ierr = PCShellGetContext(pc,(void**)&shell); CHKERRQ(ierr);
	ierr = MatDuplicate(pmat, MAT_COPY_VALUES, &M);

	shell->M = M;
	return 0;
}

// the crux of the preconditioner. apply it to a given vector x.
PetscErrorCode LdlPCApply(PC pc, Vec x, Vec y) {
	LdlPC *shell;
	PetscErrorCode ierr;

	// TODO: change this to actually applying the preconditioner.
	// i.e. solving M^(-1) * x = y
	ierr = PCShellGetContext(pc, (void**)&shell); CHKERRQ(ierr);
	ierr = MatMult(shell->M, x, y); CHKERRQ(ierr);
	//ierr = MatSolve(shell->M, x, y);
	return 0;
}

// free up memory of preconditioner
PetscErrorCode LdlPCDestroy(PC pc) {
	LdlPC  *shell;
	PetscErrorCode ierr;

	ierr = PCShellGetContext(pc,(void**)&shell); CHKERRQ(ierr);
	ierr = MatDestroy(&(shell->M)); CHKERRQ(ierr);
	ierr = PetscFree(shell); CHKERRQ(ierr);

	return 0;
}
