function d = EuclideanDistance(a,b)
% DISTANCE - computes Euclidean distance matrix
%
% E = EuclideanDistance(A,B)
%
%    A - (MxD) matrix 
%    B - (NxD) matrix
%
% Returns:
%    E - (MxN) Euclidean distances between vectors in A and B
%
%
% Description : 
%    This fully vectorized (VERY FAST!) m-file computes the 
%    Euclidean distance between two vectors by:
%
%                 ||A-B|| = sqrt ( ||A||^2 + ||B||^2 - 2*A.B )
%
% Example : 
%    A = rand(100,400); B = rand(200,400);
%    d = EuclideanDistance(A,B);

% Author   : Roland Bunschoten
%            University of Amsterdam
%            Intelligent Autonomous Systems (IAS) group
%            Kruislaan 403  1098 SJ Amsterdam
%            tel.(+31)20-5257524
%            bunschot@wins.uva.nl
% Last Rev : Oct 29 16:35:48 MET DST 1999
% Tested   : PC Matlab v5.2 and Solaris Matlab v5.3
% Thanx    : Nikos Vlassis

% Copyright notice: You are free to modify, extend and distribute 
%    this code granted that the author of the original code is 
%    mentioned as the original author of the code.

if (nargin ~= 2)
    b=a;
end

if (size(a,2) ~= size(b,2))
   error('A and B should be of same dimensionality');
end

aa=sum(a.*a,2); bb=sum(b.*b,2); ab=a*b'; 
d = sqrt(abs(repmat(aa,[1 size(bb,1)]) + repmat(bb',[size(aa,1) 1]) - 2*ab));