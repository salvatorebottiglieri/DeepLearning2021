function Y = softmax(X)
    Y = exp(X)./sum(exp(X), 2);
end

