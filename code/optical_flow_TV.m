function [u, v, I1, I2] = optical_flow_TV(img, frameNum, lambda, num_iters)

dx = 1;
dy = 1;
CFL_number = 1;
dt = round(min(dx, dy)/CFL_number);

I1 = img(:,:,frameNum);
I2 = img(:,:,frameNum+dt);

    function out = xGradient(I, dx)
        Iplus = I(:,1:end-2*dx);
        Iminus = I(:,2*dx+1:end);
        out = I;
        out(:,dx+1:end-dx) = (Iplus-Iminus)./(2*dx);
    end

    function out = yGradient(I, dy)
        Iplus = I(1:end-2*dy,:);
        Iminus = I(2*dy+1:end,:);
        out = I;
        out(dy+1:end-dy,:) = (Iplus-Iminus)./(2*dy);
    end

    function out = xxGradient(I, dx)
        Iplus = I(:,1:end-2*dx);
        Iminus = I(:,2*dx+1:end);
        out = I;
        out(:,dx+1:end-dx) = (Iplus-2*I(:,dx+1:end-dx)+Iminus)./(dx^2);
    end

    function out = yyGradient(I,dy)
        Iplus = I(1:end-2*dy,:);
        Iminus = I(2*dy+1:end,:);
        out = I;
        out(dy+1:end-dy,:) = (Iplus-2*I(dy+1:end-dy,:)+Iminus)./(dy^2);
    end


% Compute the image gradients
Ix = xGradient(I1, dx);
Iy = yGradient(I1, dy);
It = (I2 - I1)/dt;

% Initialize the motion field to zero
u = zeros(size(Ix));
v = zeros(size(I1));

% Perform gradient descent to minimize the energy functional
for i = 1:num_iters

    uxx = xxGradient(u, dx);
    uyy = yyGradient(u, dy);
    vxx = xxGradient(v, dx);
    vyy = yyGradient(v, dy);

    % Update the motion field using gradient descent
    u = u + (((1-lambda) * (Ix.*u + Iy.*v + It).*Ix) + (lambda * (uxx + uyy)));
    v = v + (((1-lambda) * (Ix.*u + Iy.*v + It).*Iy) + (lambda * (vxx + vyy)));

end

end