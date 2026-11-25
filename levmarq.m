function [x,resnorm,r] = levmarq(func,x0, options)
    % The idea of L-M is to update the parameters x0 to better fit the
    % data. This is done by simplifying the second gradient of func with
    % respect to x by replacing the sum with a simple scaled identity
    % matrix.
    
    if nargin < 3
        options = struct(); 
    end
    if ischar(func) || isstring(func)
        func = str2func(func); % convert text to function
    end
    if isfield(options, 'Jacobian')
        useGrad = strcmpi(options.Jacobian,'on'); % if we should estimate the jacobian or not
    else
        useGrad = nargout(func) >= 2; % assume second value from func is the gradient
    end
    if isfield(options, 'tol')
        tol = options.tol;
    else
        tol = 1e-10; % default tolerance
    end

    if isfield(options, 'maxIter')
        maxIter = options.maxIter;
    else
        maxIter = 200; % default max number of iterations
    end

    % func = F(x) = (f1(x), f2(x),..., fm(x))^T, m = # samples

    % First we need the gradient of F, we use second-order cental finite difference
    % df/dxi = (f(x+h e_i) - f(x-h e_i))/2h
    
    % Func is a column vector of f1,... fm   

    x = x0;
    r = res(x); 
    m = length(r);
    n = length(x0);

    prev_resnorm = Inf;
    resnorm = r.' * r;
    nrIterations = 0;

    lambda = 1; % damping factor
    while abs(resnorm-prev_resnorm) > tol  && nrIterations < maxIter
        jacobian = grad(x);
        assert(isequal(size(jacobian), [n,m]), 'Supplied gradient must be of size %dx%d',n,m); % we require correct size

        JTJ = jacobian * jacobian.';
        %norm_jtj = norm(JTJ);

        LH = JTJ + lambda*eye(n); % times p
        RH = -jacobian * r;
        p = LH \ RH;
        % we trial an update and see if it's better
        x_trial = x+p;
        r_trial = res(x_trial);
        R_trial = r_trial.'*r_trial;
        if R_trial < resnorm
            %accept
            prev_resnorm = resnorm;
            x = x_trial;
            r = r_trial;
            resnorm = R_trial;
            lambda = lambda/10;
        else
            % rejected, increase damping
            lambda = lambda*10;
        end
        nrIterations = nrIterations+ 1;
    end

    function r = res(x)
         r = func(x);
         r = r(:);
    end
    function G = grad(x)
        if useGrad
            [~,G] = func(x);
        else
            % central difference approximation of the gradient of x
            for i = n:-1:1 % reverse loop over each parameter of x
                h = eps^(1/3)*max(1,abs(x(i))); % eps(1/3) = smallest possible distance from 1/3
                e = zeros(n,1);
                e(i) = h;
                d = func(x+e) - func(x-e); 
                G(i,:) = d(:).' /(2*h); 
            end
        end
    end
end