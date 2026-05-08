function [w,step,err,time,r]=PowerMethod_extrapolatiONtrace(H,alpha,tol,itmax,m)
% Metodo delle potenze con restarting, ottimizzato per il calcolo del 
% PageRank della matrice di Google G
%
% G = [(alpha)*D^-1*(H + dang*e') + (1-alpha)*(e*v')]'
% 
% INPUT
% H      : matrice di adiacenza (sparsa) di un grafo diretto semplice
% alpha  : fattore di damping (0 < alpha < 1)
% tol    : tolleranza sul criterio di arresto (in norma 1)
% itmax  : numero massimo di iterazioni ammesse
% m      : numero di iterazioni del Power Method prima di ogni restart   
%
% OUTPUT:
% w            : vettore PageRank
% step         : numero totale iterazioni eseguite
% r            : indice dei restart del PM 
% err          : vettore degli errori (in norma 1)
% time         : tempo di esecuzione (misurato con cputime)

% Questa variante definisce il nuovo vettore di restart mediante un
% metodo di estrapolazione basato sulla traccia di G:
%   u(0) vettore iniziale
%   u(r) = w(r-1,m) - traccia(G)*w(r-1,m-1)    per r>0
% dove w(r,it) e' il vettore ottenuto dopo it iterazioni del PM su G
% applicato a partire dal vettore di restart u(r-1)
%
% La procedura esegue max m iterazioni del power method (PM) su G 
% dopo ogni restart
% NOTA: Assicurarsi che la matrice H abbia elementi diagonali tutti nulli

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
w = v;          % vettore iniziale u = w

leafs = sum(dang);
traccia = (1-alpha)+alpha*leafs/n; % traccia(G) 

err = zeros(itmax,1);
r = 0; 
step = 0;  
u_new = zeros(n,1);

t_0 = cputime; % tempo CPU iniziale

while step<itmax

    %% POWER METHOD  
    for it=1:m
    % w = G*u= ... = [(alpha)*(H' + e*dang')*D^-1 + (1-alpha)*(v*e')]*u =
    % = (alpha)*[(H'*D^-1*u) + e*(dang'*D^-1)*u] + (1-alpha)*[v*(e'*u)]
    % = (alpha)*[(H'*D^-1*u) + e*1/n*(dang'*u)] + (1-alpha)*v
        u = w;   
        w = weight.*u;
        w = H_t*w + frz*sum(u.*dang);
        w = alpha*w + (1-alpha)*frz;
        w = w/sum(w); % normalizziamo, per evitare problemi di arrotondamento
                      % w e' non negativo  
        step = step+1;
        err(step) = norm(w-u,1); 
        if err(step) < tol
           time = cputime - t_0; % tempo CPU di esecuzione
           return
        end     
    end
  
    r = r+1;
    %% RESTARTING
    % Prima di lanciare nuovamente il PM aggiorno il vettore iniziale
    % u_new = u(r), osservando che w = w(r-1,m)  e  u = w(r-1,m-1)
    
    u_new = w + (1-traccia)*u;
    unorm = 2-traccia; % norm(u_new,1) = norm(w,1) + (1-traccia)*norm(u,1)
                       % con w , u non negativi 
              
    u_new = u_new/unorm;         % normalizzo u(r)
   
    % tau = sum(abs(u_new - w)) = ...
    % = (1-traccia)/(2-traccia)*sum(abs(u-w))
    tau = (1-traccia)/(2-traccia)*err(step);
    
    w = u_new;
    if tau < tol
           time = cputime - t_0; % tempo CPU di esecuzione
           return
    end       
    
end

time = cputime - t_0; % tempo CPU di esecuzione
end
