function [w,it,res,time]=Arnoldi_typeMethod(H,alpha,tol,itmax,m)
%% Metodo Arnoldi-type (Golub - Greif, 2006)
% variante dell'algoritmo di Arnoldi raffinato con restart (Z. Jia, 1997),
% ottimizzato per il calcolo del PageRank della matrice di Google
%
% G = [(alpha)*D^-1*(H + dang*e') + (1-alpha)*(e*v')]'
% 
% INPUT:
% H      : matrice di adiacenza (sparsa) di un grafo diretto semplice
% alpha  : fattore di damping (0 < alpha < 1)
% tol    : tolleranza sul criterio di arresto 
% itmax  : numero massimo di iterazioni ammesse
% m      : dimensione del sottospazio di Krylov 
%          K_m(G,u) = span{ u , A*u , ... , A^(m-1)*u }   
%
% OUTPUT:
% w            : vettore PageRank
% it           : numero totale iterazioni eseguite
% res          : vettore dei residui || G * w - w || in norma 1
% time         : tempo di esecuzione (misurato con cputime)
%
% NOTA: si e' scelto di optare per il residuo in norma 1, piuttosto che 
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

w = e/sqrt(n); % vettore iniziale normalizzato 
stop = 0;      % condizione di arresto in caso di breakdown di Arnoldi M.
it = 0;        % indice iterazioni 
res = zeros(itmax,1);

t_0 = cputime; % tempo CPU iniziale

%% ARNOLDI ITERATION MGS + R
% Questa procedura, adattata alla struttura di G, fornisce una base 
% ortonormale colspan{V_(m)} per il sottospazio di Krylov K_m(G,w)
% In forma compatta 
%                   G * V_(m) = V_(m+1) * Hess_(m) 
% con matrici
% V_(k)      : di dim. n x k per k in {m,m+1}, con colonne ortonormali
% Hess_(m)   : di dim. (m+1) x m in forma di Hessenberg superiore
%
% NOTA: usiamo GRAM-SCHMIDT MODIFICATO (riortogonalizzazione opzionale) per
% evitare fenomeni di cancellazione
while it<itmax  
Hess = zeros(m+1,m);    % Hessenberg
V(:,1) = w;             % vettore iniziale già normalizzato

    for j=1:m   % j colonna
        q = V(:,j); 
        %% calcolo z = G * v_(j) 
        % z = G*q = [(alpha)*(H_t + e*dang')*D^-1 + (1-alpha)*(v*e')]*q =
        % = (alpha)*[(H_t*D^-1*q) + e*1/n*(dang'*q)] + (1-alpha)*v*sum(q) 
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
        %  % v2_(j+1) = v1_(j+1)  -  V(:,1:j) * Hess_2(1:j,1)
        %  % = G * v_(j)  -  V(:,1:j) *( Hess_2(1:j,1) + Hess_1(1:j,1) )
        % for i=1:j 
        %     delta = V(:,i)' * z;
        %     z = z - V(:,i) * delta;
        %     Hess(i,j) = Hess(i,j) + delta;
        % end

        % normalizzo v2_(j+1)
        Hess(j+1,j) = norm(z);
        if Hess(j+1,j) < eps    % BREAKDOWN: K_j(G,q) è G-invariante
           stop = 1;
           break
        end      
        V(:,j+1) = z / Hess(j+1,j);    
    end

    %% Calcolo la decomposizione SVD della matrice
    %% HH = Hess_(j) - [eye(j);zeros(1,j)] = U * E * S^T  
    HH = Hess - [eye(j);zeros(1,j)];
    [U,E,S] = svd(HH);

    s = S(:,j); % vettore singolare dx associato al min. valore singolare di HH
    w = V(:,1:j) * s;  % vettore approssimante  

    it = it+1;

    % Calcolo del residuo
    % res(it) = || G * w - w || = ... = sigma_min * || V_(m+1) * U(:,m) || 
    if stop
        sigma_min = 0; % si e' avuto BREAKDOWN, w e' in K_j(G,q) ed è un
                       % autovettore dominante esatto!
        res(it) = sigma_min;               
    else
        sigma_min = E(j,j); % minimo valore singolare di HH
        x = V(:,1:j+1) * U(:,j);
        res(it) = sigma_min * norm(x,1); % in norma 1 
    end

    w = w/norm(w);   % normalizzo
    % Condizione di arresto
    if res(it)<tol
        time = cputime - t_0; % tempo CPU di esecuzione
        return
    end
end

end

