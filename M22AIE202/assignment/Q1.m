A = [1 2; 3 4];
b = [3; 7];
b_prime = [3.0001; 7.0001];

% exact solution x (Ax = b)
x = A \ b;

x_prime = A \ b_prime;

delta_x = x_prime - x;
delta_b = b_prime - b;

relative_error = norm(delta_x, 2) / norm(x, 2);

cond_2_A = norm(A, 2) * norm(inv(A), 2);

disp('Exact solution x:');
disp(x);
disp('with b:');
disp(x_prime);
disp('Relative error ');
disp(relative_error);
disp('Cond2(A):');
disp(cond_2_A);
