function A = pei(n, b)
   
    A = ones(n);

    for i = 1:n
        A(i, i) = b;
    end
end

n = 20;  % Matrix order
b_values = linspace(0, 1, 100);
condition_numbers = zeros(size(b_values));  

for k = 1:length(b_values)
    b = b_values(k);
    A = pei(n, b);
    condition_numbers(k) = cond(A, 2);
end

plot(b_values, condition_numbers, 'b-', 'LineWidth', 1.5);
xlabel('b values', 'Interpreter', 'none');
ylabel('Condition Number (Cond2(A))', 'Interpreter', 'none');
title('Condition Number of Pei Matrix as b Approaches 1', 'Interpreter', 'none');
grid on;

