function [w,step,err,time,step1,r]=Arnoldi_type_extrapolatiONtrace(H,alpha,tol1,tol,itmax,i_PM,m)
% Algoritmo ibrido, ottimizzato per il calcolo del PageRank della matrice
% di Google, che combina due metodi iterativi:
% 1. metodo delle potenze con restarting (Xueyuan Tan) [ALG1]
% 2. metodo Arnoldi-type (Golub - Greif) [AtM]
%
%% Matrice di Google
% G = [(alpha)*D^-1*(H + dang*e') + (1-alpha)*(e*v')]'
% 
% INPUT:
% H      : matrice di adiacenza (sparsa) di un grafo diretto semplice
% alpha  : fattore di damping (0 < alpha < 1)
% tol1   : tolleranza sul criterio di arresto per ALG1
% tol    : tolleranza sul criterio di arresto per AtM
%          tol < tol1 
% itmax  : numero massimo di iterazioni ammesse
% i_PM   : numero di iterazioni del Power Method prima di ogni restart in ALG1
% m      : dimensione del sottospazio di Krylov 
%          K_m(G,u) = span{ u , A*u , ... , A^(m-1)*u }   
%
% OUTPUT:
% w            : vettore PageRank
% step         : numero totale iterazioni eseguite
% step1        : numero totale iterazioni ALG1 
% r            : indice dei restart del PM in ALG1 
% err          : vettore degli errori 
% time         : tempo di esecuzione (misurato con cputime), di cui
%                time(1) relativo ad ALG1
%                time(2) relativo ad AtM  
%
% NOTA 1: Assicurarsi che la matrice H abbia elementi diagonali tutti
% nulli.
% NOTA 2: si e' scelto di optare per il residuo in norma 1, piuttosto che 
% in norma 2, per ragioni di confrontabilità con altri metodi per i quali
% e' naturalmente conveniente la scelta di tale norma, come il Power M.

n = size(H,1);
H_t = H';        % trasposta
e = ones(n,1);
frz = 1/n;       % v = e*frz vettore di personalizzazione 

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
step = 0; step1 = 0; 
u_new = zeros(n,1);

%% >>> PRIMA PARTE - ALG1
% Scelto come vettore iniziale u(0) = v, la procedura esegue ad ogni passo r
% max m iterazioni del metodo delle potenze applicato a G a partire da un
% opportuno vettore di "restart" u(r), cosi' definito:
%   u(r) = w(r-1,m) - traccia(G)*w(r-1,m-1)    per r>0
% dove w(r,it) e' il vettore di PageRank ottenuto al passo r dopo 
% it iterazioni del PM su G
%
% La procedura si arresta se il residuo (in norma 1) tra i vettori ottenuti 
% al termine di due iterate consecutive è minore di tol1.
% Il vettore w ottenuto viene scelto come vettore iniziale della procedura
% esplicitata nella seconda parte dell'algoritmo

r = 0; 
stop1 = 0; % variabile di arresto 
t_0 = cputime; % tempo CPU iniziale

while step<itmax && ~stop1
    %% POWER METHOD  
    for it=1:i_PM
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
        if err(step) < tol1 
           t_1 = cputime - t_0; % tempo CPU di esecuzione ALG1
           stop1 = 1;
           break
        end     
    end
    
    %% RESTARTING
     % Prima di lanciare nuovamente il PM aggiorno il vettore iniziale
     % u_new = u(r), osservando che w = w(r-1,m)  e  u = w(r-1,m-1)
    if ~stop1       
        r = r+1;
        u_new = w + (1-traccia)*u;
        unorm = 2-traccia; % norm(u_new,1) = norm(w,1) + (1-traccia)*norm(u,1)
                           % con w , u non negativi 
              
        u_new = u_new/unorm;         % normalizzo u(r)
        w = u_new;

        % tau = sum(abs(u_new - w)) = ...
        % = (1-traccia)/(2-traccia)*sum(abs(u-w))
        tau = (1-traccia)/(2-traccia)*err(step);   
        if tau < tol1
           t_1 = cputime - t_0; % tempo CPU di esecuzione ALG1
           stop1 = 1;
        end       
    end
end
step1 = step;
time(1) = t_1;

%% >>> SECONDA PARTE - Metodo ARNOLDI-TYPE

stop2 = 0; % condizione di arresto in caso di breakdown di Arnoldi M.
sigma_min = tol;
t_2 = cputime;

while step<itmax && err(step)>=tol
%% ARNOLDI ITERATION MGS (+ reorth.)
% Questa procedura, adattata alla struttura di G, fornisce una base 
% ortonormale colspan{V_(m)} per il sottospazio di Krylov K_m(G,w)
% In forma compatta 
%                   G * V_(m) = V_(m+1) * Hess_(m) 
% con matrici
% V_(k)      : di dim. n x k per k in {1,...,m,m+1}, con colonne ortonormali
% Hess_(m)   : di dim. (m+1) x m in forma di Hessenberg superiore
%
% NOTA: usiamo GRAM-SCHMIDT MODIFICATO (riortogonalizzazione opzionale) per
% evitare fenomeni di cancellazione

Hess = zeros(m+1,m);    % Hessenberg m+1 x m
V(:,1) = w/norm(w,2);   % vettore iniziale normalizzato

    for j=1:m   % j colonna
        q = V(:,j);
        %% calcolo z = G * v_(j) 
        % z = G*q = [(alpha)*(H_t + e*dang')*D^-1 + (1-alpha)*(v*e')]*q =
        %   = (alpha)*[(H_t*D^-1*q) + e*1/n*(dang'*q)] + (1-alpha)*v*sum(q) 
        z = weight.*q;
        z = H_t*z + frz*sum(q.*dang);
        z = alpha*z;
        z = z + (1-alpha)*frz*sum(q);
      
        %% Primo passaggio MGS
         % v1_(j+1) = G * v_(j)  -  V(:,1:j) * Hess_1(1:j,1)
        for i=1:j % i riga
            Hess(i,j) = V(:,i)' * z;
            z = z - V(:,i) * Hess(i,j);                                 
        end

        % %% Secondo passaggio - riortogonalizzazione completa (OPZIONALE)
        %  % v2_(j+1) = v1_(j+1) - V(:,1:j) * Hess_2(1:j,1)
        %  % = G * v_(j) - V(:,1:j) * ( Hess_2(1:j,1) + Hess_1(1:j,1) )
        % for i=1:j 
        %     delta = V(:,i)' * z;
        %     z = z - V(:,i) * delta;
        %     Hess(i,j) = Hess(i,j) + delta;
        % end
        
        % normalizzo v2_(j+1)
        Hess(j+1,j) = norm(z,2);
        if Hess(j+1,j) < eps    % BREAKDOWN: K_j(G,q) è G-invariante
           stop2 = 1;
           break
        end      
        V(:,j+1) = z / Hess(j+1,j);    
    end

    %% Calcolo la decomposizione SVD della matrice di Hessenberg
    %% HH = Hess_(j) - [eye(j);zeros(1,j)] = U * E * S^T
    HH = Hess(1:j+1,1:j) - [eye(j);zeros(1,j)];
    [U,E,S] = svd(HH);

    s = S(:,j); % vettore singolare dx associato al min. valore singolare di HH
    w = V(:,1:j) * s;  % vettore di Ritz raffinato 

    step = step+1;
    % Calcolo del residuo
    % res(step) = || G * w - w || = ... = sigma_min * || V_(m+1) * U(:,m) ||
    if stop2
        sigma_min = 0; % si e' avuto BREAKDOWN, w e' in K_j(G,q) ed è un
                       % autovettore dominante esatto!
        err(step) = sigma_min;
    else
        sigma_min = E(j,j); % minimo valore singolare di HH
        x = V(:,1:j+1) * U(:,j);
        err(step) = sigma_min * norm(x,1); % in norma 1 
    end
    w = w/norm(w);   % normalizzo
end

time(2) = cputime - t_2; % tempo CPU di esecuzione AtM
end
