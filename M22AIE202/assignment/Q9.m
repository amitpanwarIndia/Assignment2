function x = gauss_elimination(A, b)
    [n, m] = size(A);
    
    if n ~= m
        error('not a square');
    end
    
    AugmentedMatrix = [A b];
    
    for k = 1:n-1
        % pivoting
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
    
    x = zeros(n, 1);
    for i = n:-1:1
        x(i) = (AugmentedMatrix(i, end) - AugmentedMatrix(i, i+1:n) * x(i+1:n)) / AugmentedMatrix(i, i);
    end
end

% Example
A = [6 2 2; 2 2/3 1/3; 1 2 -1];
b = [-2; 1; 0];
x = gauss_elimination(A, b);

disp('x');
disp(x);
