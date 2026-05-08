function [w,it,err,time_elapsed]=PowerMethod(H,alpha,tol,itmax)
% Metodo delle potenze ottimizzato per il calcolo del PageRank  
% della matrice di Google G
%
% G = [(alpha)*D^-1*(H + dang*e') + (1-alpha)*(e*v')]'
%
% INPUT:
% H      : matrice di adiacenza (sparsa) di un grafo diretto semplice
% alpha  : fattore di damping (0 < alpha < 1)
% tol    : tolleranza sul criterio di arresto (in norma 1)
% itmax  : massimo numero di iterazioni ammesse
%
% OUTPUT:
% w            : vettore PageRank
% it           : numero iterazioni eseguite 
% err          : vettore degli errori (in norma 1)
% time_elapsed : tempo di esecuzione (misurato con cputime)

n = size(H,1); 
H_t = H';        % trasposta
e = ones(n,1);
frz = 1/n;

% Outdegree e dangling nodes
outdegree = H*e;
dang = outdegree==0;

% D = diag(d) con d =(H + dang*e')*e = H*e + dang*n
d = outdegree + dang*n; 
weight = 1./d;  % D^-1 = diag(weight)   

v = e * frz;    % vettore di personalizzazione
u = v;          % vettore iniziale 
err = zeros(itmax,1);
t_0 = cputime; % tempo CPU iniziale

% POWER METHOD 
for it=1:itmax
    % w = G*u= ... = [(alpha)*(H' + e*dang')*D^-1 + (1-alpha)*(v*e')]*u =
    % = (alpha)*[(H'*D^-1*u) + e*(dang'*D^-1)*u] + (1-alpha)*[v*(e'*u)]
    % = (alpha)*[(H'*D^-1*u) + e*1/n*(dang'*u)] + (1-alpha)*v 
    w = weight.*u;
    w = H_t*w + frz*sum(u.*dang);
    w = alpha*w + (1-alpha)*frz;
    w = w/sum(w); % normalizziamo, per evitare problemi di arrotondamento
    err(it) = norm(w-u,1);
    if err(it) < tol
 	   time_elapsed = cputime - t_0; % tempo CPU di esecuzione
       return
    end
    u = w;
end

time_elapsed = cputime - t_0;
end