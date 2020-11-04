function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
 a1 = [ones(m,1) ,X] % size 5000 into 401
 z2 = a1*Theta1'
 a2 = sigmoid(z2)
 a2 = [ones(size(a2,1),1), a2];  % 5000 x 26
 z3 = a2*Theta2'
 a3 = sigmoid(z3);   % Final op 5000 10 ; each value in each row is prob of each label max of them is the result
 h_x = a3     % Final op 5000 10
 y_result_vector = zeros(m,num_labels)  % 5000 10   K = 10  
  %Now we are calucalting vector form of output in form of 1 or 0 in K lables for each example so that we can calculate cost for each example for rach output label%
  for i = 1:m
    y_result_vector(i,y(i)) = 1
  endfor
  sumoverKLabels = sum((-y_result_vector.*log(h_x))-((1-y_result_vector).*log(1-h_x))) % 5000 1 
  sumoverAllExamples  = (1/m) * sum(sumoverKLabels) % scalor value, this is J ie cost
  J = sumoverAllExamples
  % Theta 1 25 401
 
 % Adding regularization term to cost
 theta1 = Theta1(:,2:size(Theta1,2));%25*400
theta2 = Theta2(:,2:size(Theta2,2)); %10*25
J = J + lambda / 2 / m * (sum(sum(theta1.^2))+ sum(sum(theta2.^2)));
  
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
##    for i = 1:m
##      % For Layer 1 computations
##      a1 = X(i,:); % 1 401 , input layer of example by example
##      % For layer 2 computations
##      z2 = a1*Theta1' % 1 401 and 401 25  gives 1 25
##      a2 = [1; sigmoid(z2)]; % 1 (hidden_layer_size+1)  == 1x 26
##      % for layer-3
##       z3 =   a2*Theta2'; % 1 num_labels  == 1    10 
##       a3 = sigmoid(z3); % 1 num_labels  == 1    10 
##      yVector = (1:num_labels)==y(t); % actual result vector of 1 10
##      % Calculating delta values per layer
##      delta3 = a3 - yVector;          % 1 10
##      delta2 = (Theta2' * delta3') .* [1; sigmoidGradient(z2)]' ; % (hidden_layer_size+1) x 1 == 26 x 1
##      delta2 = delta2(2:end); % hidden_layer_size x 1 == 25 x 1 %Removing delta2 for bias node      
##      % delta_1 is not calculated because we do not associate error with the input  
##       Theta1_grad = Theta1_grad + (delta2 * a1); % 25 x 401
##       Theta2_grad = Theta2_grad + (delta3 * a2); % 10 x 26
##
##   endfor
## Theta1_grad = (1/m) * Theta1_grad; % 25 x 401
##  Theta2_grad = (1/m) * Theta2_grad; % 10 x 26
 % Vectorised Implentation
 A1 = a1  %5000 401   [ones(m,1) X] 
 Z2 = A1 * Theta1' % 5000 25
 A2 = sigmoid(Z2)
 A2 = [ones(size(A2,1),1),A2]  % add bias unit column 5000 26
 Z3 = A2 * Theta2'  % 5000 10 
 A3 = sigmoid(Z3) % 5000 10
 yVector = (1:num_labels)==y % m x num_labels == 5000 x 10
 Delta3 = A3 - yVector  % 5000 x 10
 Delta2 = (Delta3*Theta2).* [ones(size(Z2,1),1) sigmoidGradient(Z2)] % 5000 26
 Delta2 = Delta2(:,2:end) % removing bias unit  % 5000 25
 Theta1_grad = (1/m) * (Delta2' * A1) % 25 x 401
 Theta2_grad = (1/m) * (Delta3' * A2)  % 10 x 26 
 
  
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Adding regularization term to cost
Theta1_grad = Theta1_grad + (lambda/m)* [zeros(size(Theta1,1),1) Theta1(:,2:end)]
Theta2_grad = Theta2_grad + (lambda/m)* [zeros(size(Theta2,1),1) Theta2(:,2:end)]
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
