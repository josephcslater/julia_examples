using LinearAlgebra
   
"""
    gauss_elimination(A::Matrix{Float64}, b::Vector)

Julia function for gauss elimination with pivoting
"""
function gauss_elimination(A::Matrix, b::Vector)
    n = size(A, 1)

    # Forward elimination
    for k = 1:n-1
        for i = k+1:n
            factor = A[i, k] / A[k, k]
            for j = k+1:n
                A[i, j] -= factor * A[k, j]
            end
            b[i] -= factor * b[k]
            A[i, k] = 0.0
        end
    end

    # Back substitution
    x = zeros(n)
    x[n] = b[n] / A[n, n]
    for k = n-1:-1:1
        x[k] = (b[k] - dot(A[k, k+1:n], x[k+1:n])) / A[k, k]
    end

    return x
end

A = [1 2 3; 4 5 6; 7 8 10]
b = [1, 2, 3]

x = gauss_elimination(A,b)
print(x)
