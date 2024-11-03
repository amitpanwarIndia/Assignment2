function [L, U] = lu_decomposition(A)
    [n, m] = size(A);
    if n ~= m
        error('not square');
    end
    
    L = eye(n);
    U = zeros(n);
    
    for j = 1:n
        for i = 1:j
            U(i, j) = A(i, j);
            for k = 1:i-1
                U(i, j) = U(i, j) - L(i, k) * U(k, j);
            end
        end
      
        for i = j+1:n
            L(i, j) = A(i, j);
            for k = 1:j-1
                L(i, j) = L(i, j) - L(i, k) * U(k, j);
            end
            L(i, j) = L(i, j) / U(j, j);
        end
    end
end

% example:
A = [2, 1, 1; 4, -6, 0; -2, 7, 2];
[L, U] = lu_decomposition(A);

disp('L = ');
disp(L);
disp('U = ');
disp(U);


function x = solve_lu(A, b)
    [L, U] = lu_decomposition(A);
    
    n = length(b);
    y = zeros(n, 1);
    for i = 1:n
        y(i) = b(i);
        for j = 1:i-1
            y(i) = y(i) - L(i, j) * y(j);
        end
    end
    
    x = zeros(n, 1);
    for i = n:-1:1
        x(i) = y(i);
        for j = i+1:n
            x(i) = x(i) - U(i, j) * x(j);
        end
        x(i) = x(i) / U(i, i);
    end
end

% example
A = [2, 1, 1; 4, -6, 0; -2, 7, 2];
b = [5; -2; 9];
x = solve_lu(A, b);

disp('soln x = ');
disp(x);
