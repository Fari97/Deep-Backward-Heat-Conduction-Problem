% % % %
clc; clear all; close all;

figure; subplot(1,2,1)
scatter (x_star, y_star, [], u_star);
title('Exact u(x,y 0) when T = 1'); xlabel('x'); ylabel('y')
axis tight;
subplot(1,2,2)
scatter (x_star, y_star, [], u_pred);
title('Predicted u(x,y,0) when T = 1'); xlabel('x'); ylabel('y')
axis tight;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure; subplot(1,2,1)
scatter3 (x_star, y_star, u_star, [], u_star);
title('Exact u(x,y 0) when T = 1'); xlabel('x'); ylabel('y'); zlabel('u(x,y,0)')
axis tight;
subplot(1,2,2)
scatter3 (x_star, y_star, u_pred, [], u_pred);
title('Predicted u(x,y 0) when T = 1'); xlabel('x'); ylabel('y'); zlabel('u(x,y,0)')
axis tight;