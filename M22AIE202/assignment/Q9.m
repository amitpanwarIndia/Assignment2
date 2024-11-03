function x = gauss_elimination(A, b)
    % Gauss Elimination to solve the system Ax = b
    [n, m] = size(A);
    
    if n ~= m
        error('Matrix A must be square');
    end
    
    % Augment the matrix A with the vector b
    AugmentedMatrix = [A b];
    
    % Forward elimination
    for k = 1:n-1
        % Partial pivoting (optional but recommended for numerical stability)
        [~, maxIndex] = max(abs(AugmentedMatrix(k:n, k)));
        maxIndex = maxIndex + k - 1;
        if maxIndex ~= k
            AugmentedMatrix([k, maxIndex], :) = AugmentedMatrix([maxIndex, k], :);
        end
        
        for i = k+1:n
            factor = AugmentedMatrix(i, k) / AugmentedMatrix(k, k);
            AugmentedMatrix(i, k:end) = AugmentedMatrix(i, k:end) - factor * AugmentedMatrix(k, k:end);
        end
    end
    
    % Back substitution
    x = zeros(n, 1);
    for i = n:-1:1
        x(i) = (AugmentedMatrix(i, end) - AugmentedMatrix(i, i+1:n) * x(i+1:n)) / AugmentedMatrix(i, i);
    end
end

% Example usage:
A = [6 2 2; 2 2/3 1/3; 1 2 -1];
b = [-2; 1; 0];
x = gauss_elimination(A, b);

disp('Solution x = ');
disp(x);
