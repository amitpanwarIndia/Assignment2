% 100x100 matrix A
n = 100;
A = diag(ones(1, n)) + diag(2 * ones(1, n-1), 1);

orig_eigenvalues = eig(A);

% perturbing A(100, 1) by 10^-10
A(100, 1) = A(100, 1) + 1e-10;

per_eigenvalues = eig(A);

cond_2_A = norm(A, 2) * norm(inv(A), 2);

disp('eig of matrix A:');
disp(orig_eigenvalues);
disp('eig of perturbed matrix:');
disp(per_eigenvalues);
disp('Cond2(A):');
disp(cond_2_A);
