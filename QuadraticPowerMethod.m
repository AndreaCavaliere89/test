function [w,step,err,time,r]=QuadraticPowerMethod(H,alpha,tol,itmax,m)
% Metodo delle potenze (PM) con restarting, ottimizzato per il calcolo del 
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
% metodo di ESTRAPOLAZIONE QUADRATICA:
%
% u_(0) vettore iniziale
% u_(r) = b_0 * w_(r-1,m-2) + b_1 * w_(r-1,m-1) + b_2 * w_(r-1,m)
%        dove w_(r,it) e' il vettore ottenuto dopo it iterazioni del PM su G
%        applicato a partire dal vettore di restart u_(r-1)
%
% La procedura esegue max m iterazioni del PM su G dopo ogni restart

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

w_prec = zeros(n,4);
Y = zeros(n,3); y = zeros(n,1); g = zeros(3,1); 

err = zeros(itmax,1);
r = 0; 
step = 0;  

t_0 = cputime; % tempo CPU iniziale

while step<itmax

    % POWER METHOD  
    for it=1:m
    % w = G*u= ... = [(alpha)*(H' + e*dang')*D^-1 + (1-alpha)*(v*e')]*u =
    % = (alpha)*[(H'*D^-1*u) + e*(dang'*D^-1)*u] + (1-alpha)*[v*(e'*u)]
    % = (alpha)*[(H'*D^-1*u) + e*1/n*(dang'*u)] + (1-alpha)*v
        u = w;
        w = weight.*u;
        w = H_t*w + frz*sum(u.*dang);
        w = alpha*w + (1-alpha)*frz;
        w = w/norm(w,1); % normalizziamo 
        step = step+1;
        err(step) = norm(w-u,1); 
        if err(step) < tol
           time = cputime - t_0; % tempo CPU di esecuzione
           return
        end
        
        % memorizzo i vettori w calcolati alle iterazione m-3,m-2,m-1,m
        if it>=m-3
           w_prec(:,it-m+4) = w;
        end   
    end
  
    r = r+1;
    % RESTARTING
    % Prima di lanciare nuovamente il PM 
    % aggiorno il vettore iniziale u_new = u(r) 
    u_new = QuadraticExtrapolation(w_prec);           
    u_new = u_new/norm(u_new,1);         % normalizzo u(r)
    
    w = u_new;  
end

time = cputime - t_0; % tempo CPU di esecuzione

% ESTRAPOLAZIONE QUADRATICA
function x=QuadraticExtrapolation(X)
    
    % X = [ w_(m-3) | w_(m-2) | w_(m-1) | w_(m) ]
    % Y = [ w_(m-2) - w_(m-3)  |  w_(m-1) - w_(m-3) ]
    Y = X(:,2:3) - X(:,1); 
    % y = w_(m) - w_(m-3)
    y = X(:,4) - X(:,1); 

    y_temp = zeros(n,1);  

    % risoluzione del problema ai minimi quadrati con thin QR decomposition
    % || Y * [g1|g2] = -y ||
    [Q,R]=qr(Y,"econ"); % Q matrice nx2 , R matrice 2x2  
    y_temp = - Q'*y;
    
    % R*[g1|g2] = - Q'*y con R triangolare superiore
    % si risolve per sostituzione all'indietro
    % oppure con g(1:2) = R \ y_temp;
    g(2) = y_temp(2)/R(2,2);
    g(1) = (y_temp(1) - R(1,2)*g(2))/R(1,1);
    g(3) = 1;
    
    for k=3:-1:1
        beta(k) = sum(g(k:3)); 
    end
   
    x=zeros(n,1);
    for k=1:3
        x = x + beta(k)*X(:,k+1);
    end    
    % oppure con x = sum(beta.*X(:,2:end),2);
   
end

end
