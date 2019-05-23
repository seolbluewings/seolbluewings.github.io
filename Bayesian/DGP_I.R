rm(list=ls())
set.seed(2018311133)
library(mvtnorm)
n=30; G=50; R=50
gamma=c(-1.5,0.5,-1)
beta=c(0.5,0.5,0.5)
lambda=0.05
delta=0.3
sigma=matrix(c(1,0.5,0.5,1.25),ncol=2)
sig_alpha=0.5
mu=c(0,0)

z=matrix(0,ncol=n,nrow=G)
e=matrix(0,ncol=n,nrow=G)
C_g=matrix(0,ncol=n,nrow=n)
C_glist=list()

for(g in 1:G){
  mat=rmvnorm(n,mu,sigma)
  u1=runif(n,0,1); u2=runif(n,0,1)
  for(i in 1:n){
    for(j in 1:n){
      if(u1[i]>=0.7 && u2[j]>=0.7){
        C_g[i,j]=1
      } else if(u1[i]<=0.3 && u2[j]<=0.3){
        C_g[i,j]=1
      } else{
        C_g[i,j]=0
      }
      if(i==j){
        C_g[i,j]=0
      }
    }
    z[g,i]=mat[i,1]
    e[g,i]=mat[i,2]
  }
  C_glist[[g]]=C_g
}

p=matrix(0,ncol=n,nrow=n);pg=list()
w=matrix(0,ncol=n,nrow=n);Wg=list()

for(g in 1:G){
  for(i in 1:n){
    for(j in 1:n){
      psi=gamma[1]+gamma[2]*C_glist[[g]][i,j]-gamma[3]*abs(z[g,i]-z[g,j])
      p[i,j]=exp(psi)/(1+exp(psi))
      if(runif(1,0,1)<=p[i,j]){
        w[i,j]=1
      } else{
        w[i,j]=0
      } 
      if(i==j){
        w[i,j]=0
      }
    }
  }
  pg[[g]]=p
  Wg[[g]]=w
}

ag=matrix(0,ncol=n,nrow=G)
lg=matrix(0,ncol=n,nrow=G)
Xg=list();yg=list()

for(g in 1:G){
  x=rnorm(n,0,1)
  alpha=rnorm(1,0,sqrt(sig_alpha))
  lmg=rep(1,n)
  Xg[[g]]=x
  ag[g,]=alpha
  lg[g,]=lmg
  yg[[g]]=solve(diag(1,n)-lambda*Wg[[g]])%*%(lg[g,]*beta[1]+Xg[[g]]*beta[2]+Wg[[g]]%*%Xg[[g]]%*%beta[3]+lg[g,]*ag[g,]+e[g,])
}

### assign parameter in prior distribution ###
gamma0=rep(0,3)
beta0=rep(0,3)
Gamma0=5*diag(1,3)
Beta0=5*diag(1,3)
rho0=5; eta0=1;
sig0=c(0,0)

iter=300
Z_mat=matrix(0,ncol=n,nrow=G) #save for Z
gamma_mat=matrix(0,ncol=3,nrow=iter) #save for gamma
lambda_mat=matrix(0,ncol=1,nrow=iter) #save for lambda
beta_mat=matrix(0,ncol=3,nrow=iter) #save for beta
sige_mat=matrix(0,ncol=1,nrow=iter) #save for sigma_e^2
siga_mat=matrix(0,ncol=1,nrow=iter) #save for sigma_a^2
sigez_mat=matrix(0,ncol=1,nrow=iter) #save for sigma_ez^2
alpha_mat=matrix(0,nrow=iter,ncol=G)

mini=c();maxi=c();Tau_g=c()
for(g in 1:G){
  for(i in 1:n){
    mini[i]=sum(Wg[[g]][i,])
    maxi[i]=sum(Wg[[g]][,i])
    Tau_g=min(mini[which.max(mini)],maxi[which.max(maxi)])
    tau=Tau_g[which.max(Tau_g)]
  }
}

### starting value of draw ###
gamma_mat[1,]=c(-1,0.5,1);lambda_mat[1,]=0.04
sige_mat[1,]=1; sigez_mat[1,]=0.5; siga_mat[1,]=0.3
beta_mat[1,]=c(0.3,0.2,0.4)

### save 
beta_list=list()
gamma_list=list()
lambda_list=list()
sige_list=list()
siga_list=list()
sigez_list=list()

for(r in 1:R){
  st_time=Sys.time()
  cat(r,"repetition of MCMC start!","\n");cat("\n")
  for(t in 2:iter){
    start_time=Sys.time()
    ### propose lambda*
    accept=0
    while(accept==0){
      if(t<3){
        lambda_1=rnorm(1,lambda_mat[t-1,],0.1)
      } else{
        lambda_1=rnorm(1,lambda_mat[t-1,],cov(as.matrix(lambda_mat[1:t-1,]))*2.38^2)*0.95+
          rnorm(1,lambda_mat[t-1,],0.1)*0.05
      }
      lambda_1=as.numeric(lambda_1)
      if(lambda_1>-1/tau && lambda_1<1/tau){
        accept=1
      }
    }
    
    ### propose gamma*
    if(t<6){
      gamma_1=rmvnorm(1,gamma_mat[t-1,],diag(1,3)*0.1^2/3)
    } else{
      gamma_1=rmvnorm(1,gamma_mat[t-1,],cov(as.matrix(gamma_mat[1:t-1,])))*0.95+
        rmvnorm(1,gamma_mat[t-1,],diag(1,3)*0.1^2/3)*0.05
    }
    pp_l=1;pp_G=1
    
    ### M-H algorithm for sampling Z ###
    V=(sige_mat[t-1,]-sigez_mat[t-1,]^2)*diag(1,n)+siga_mat[t-1,]*rep(1,n)%*%t(rep(1,n))
    for(g in 1:G){
      zz_1=Z_mat[g,]
      for(v in 1:n){
        zz_1[v]=rnorm(1,0,1)+Z_mat[g,v]
        pp=1
        for(i in 1:n){
          if(i==v){
            for(j in 1:n){
              if(j==v){
                p_1=1
                p_2=1
              } else{
                psi_1=gamma_mat[t-1,1]+gamma_mat[t-1,2]*C_glist[[g]][i,j]-gamma_mat[t-1,3]*abs(zz_1[i]-zz_1[j])
                psi_2=gamma_mat[t-1,1]+gamma_mat[t-1,2]*C_glist[[g]][i,j]-gamma_mat[t-1,3]*abs(Z_mat[g,i]-Z_mat[g,j])
                p_1=exp(psi_1*Wg[[g]][i,j])/(1+exp(psi_1))
                p_2=exp(psi_2*Wg[[g]][i,j])/(1+exp(psi_2))
              }
              pp=pp*(p_1/p_2)
            }
          } else{
            psi_1=gamma_mat[t-1,1]+gamma_mat[t-1,2]*C_glist[[g]][i,v]-gamma_mat[t-1,3]*abs(zz_1[i]-zz_1[v])
            psi_2=gamma_mat[t-1,1]+gamma_mat[t-1,2]*C_glist[[g]][i,v]-gamma_mat[t-1,3]*abs(Z_mat[g,i]-Z_mat[g,v])
            p_1=exp(psi_1*Wg[[g]][i,v])/(1+exp(psi_1))
            p_2=exp(psi_2*Wg[[g]][i,v])/(1+exp(psi_2))
            pp=pp*(p_1/p_2)
          }
        }
        SS=diag(1,n)-lambda_mat[t-1,]*Wg[[g]]
        ep=SS%*%(yg[[g]])-(rep(1,n)*beta_mat[t-1,1]+Xg[[g]]*beta_mat[t-1,2]+Wg[[g]]%*%Xg[[g]]%*%beta_mat[t-1,3])#-rep(1,n)*alpha_mat[t-1,g]
        like_Y1=exp(-0.5*t(ep-sigez_mat[t-1,]*zz_1)%*%solve(V)%*%(ep-sigez_mat[t-1,]*zz_1))
        like_Y2=exp(-0.5*t(ep-sigez_mat[t-1,]*Z_mat[g,])%*%solve(V)%*%(ep-sigez_mat[t-1,]*Z_mat[g,]))
        pp=pp*(like_Y1/like_Y2)*(dmvnorm(zz_1[v])/dmvnorm(Z_mat[g,v]))
        pp=min(pp,1)
        if(runif(1,0,1)<=pp){
          Z_mat[g,v]=zz_1[v]
        } else{
          zz_1=Z_mat[g,]
        }
      }
      ### M-H algorithm for sampling gamma and lambda
      pp=1
      for(i in 1:n){
        for(j in 1:n){
          psi_1=gamma_1[1]+gamma_1[2]*C_glist[[g]][i,j]-gamma_1[3]*abs(Z_mat[g,i]-Z_mat[g,j])
          psi_2=gamma_mat[t-1,1]+gamma_mat[t-1,2]*C_glist[[g]][i,j]-gamma_mat[t-1,3]*abs(Z_mat[g,i]-Z_mat[g,j])
          p_1=exp(psi_1*Wg[[g]][i,j])/(1+exp(psi_1))
          p_2=exp(psi_2*Wg[[g]][i,j])/(1+exp(psi_2))
          if(i==j){
            p_1=1;p_2=1
          }
          pp=pp*(p_1/p_2)
        }
      }
      pp_G=pp_G*pp
      pp_G=pp_G*(dmvnorm(gamma_1,gamma0,Gamma0)/dmvnorm(gamma_mat[t-1,],gamma0,Gamma0))
      pp_G=min(pp_G,1)
      
      S_1=diag(1,n)-lambda_1*Wg[[g]]
      S_2=diag(1,n)-lambda_mat[t-1,]*Wg[[g]]
      ep_1=S_1%*%yg[[g]]-(rep(1,n)*beta_mat[t-1,1]+Xg[[g]]*beta_mat[t-1,2]+Wg[[g]]%*%Xg[[g]]%*%beta_mat[t-1,3])#-rep(1,n)*alpha_mat[t-1,g]
      ep_2=S_2%*%yg[[g]]-(rep(1,n)*beta_mat[t-1,1]+Xg[[g]]*beta_mat[t-1,2]+Wg[[g]]%*%Xg[[g]]%*%beta_mat[t-1,3])#-rep(1,n)*alpha_mat[t-1,g]
      like_1=det(S_1)*exp(-0.5*t(ep_1-sigez_mat[t-1,]*Z_mat[g,])%*%solve(V)%*%(ep_1-sigez_mat[t-1,]*Z_mat[g,]))
      like_2=det(S_2)*exp(-0.5*t(ep_2-sigez_mat[t-1,]*Z_mat[g,])%*%solve(V)%*%(ep_2-sigez_mat[t-1,]*Z_mat[g,]))
      pp_l=pp_l*(like_1/like_2)
    }
    pp_l=min(pp_l,1)
    if(runif(1,0,1)<=pp_G){
      gamma_mat[t,]=gamma_1
    } else{
      gamma_mat[t,]=gamma_mat[t-1,]
    }
    if(runif(1,0,1)<=pp_l){
      lambda_mat[t,]=lambda_1
    } else{
      lambda_mat[t,]=lambda_mat[t-1,]
    }
    
    ### sample of beta from posterior distribution
    XVX=matrix(0,ncol=length(beta0),nrow=length(beta0))
    XVY=matrix(0,nrow=length(beta0),ncol=1)
    Vg=list()
    for(g in 1:G){
      SS=diag(1,n)-lambda_mat[t,]*Wg[[g]]
      YY=SS%*%yg[[g]]-sigez_mat[t-1,]*Z_mat[g,]#-rep(1,n)*alpha_mat[t-1,g]
      XX=cbind(rep(1,n),Xg[[g]],Wg[[g]]%*%Xg[[g]])
      Vg[[g]]=(sige_mat[t-1,]-(sigez_mat[t-1,])^2)*diag(1,n)+siga_mat[t-1,]*rep(1,n)%*%t(rep(1,n))
      XVX=XVX+t(XX)%*%solve(Vg[[g]])%*%XX
      XVY=XVY+t(XX)%*%solve(Vg[[g]])%*%YY
    }
    B=solve(solve(Beta0)+XVX)
    nbeta=B%*%(solve(Beta0)%*%beta0+XVY)
    beta_mat[t,]=rmvnorm(1,nbeta,B)
    
    ### sampling of sigmae^2 & sigmaez^2 from posterior 
    accept=0
    while(accept==0){
      if(t<=4){
        sig=rmvnorm(1,c(sige_mat[t-1,],sigez_mat[t-1,]),diag(1,2)*0.1^2/2)
      } else{
        a=sige_mat[1:t-1,]
        b=sigez_mat[1:t-1,]
        cov_mat=cbind(a,b)
        sig=rmvnorm(1,c(sige_mat[t-1,],sigez_mat[t-1,]),
                    cov(cov_mat)*2.38^2/2)*0.95+
          rmvnorm(1,c(sige_mat[t-1,],sigez_mat[t-1,]),diag(1,2)*0.01/2)*0.05
      }
      sige_1=sig[1]
      sigez_1=sig[2]
      if(sigez_1^2<sige_1 && sigez_1>0){
        accept=1
      }
    }
    V1=(sige_1-sigez_1^2)*diag(1,n)+siga_mat[t-1,]*rep(1,n)%*%t(rep(1,n))
    V2=(sige_mat[t-1,]-sigez_mat[t-1,]^2)*diag(1,n)+siga_mat[t-1,]*rep(1,n)%*%t(rep(1,n))
    pp_sig=1
    for(g in 1:G){
      SS=diag(1,n)-lambda_mat[t,]*Wg[[g]]
      ep=SS%*%yg[[g]]-(rep(1,n)*beta_mat[t,1]+Xg[[g]]*beta_mat[t,2]+Wg[[g]]%*%Xg[[g]]%*%beta_mat[t,3])#-rep(1,n)*alpha_mat[t-1,g]
      like_1=det(V1)^(-0.5)*exp(-0.5*t(ep-sigez_1*Z_mat[g,])%*%solve(V1)%*%(ep-sigez_1*Z_mat[g,]))
      like_2=det(V2)^(-0.5)*exp(-0.5*t(ep-sigez_mat[t-1,]*Z_mat[g,])%*%solve(V2)%*%(ep-sigez_mat[t-1,]*Z_mat[g,]))
      pp_sig=pp_sig*(like_1/like_2)
    }
    pp_sig=pp_sig*(dmvnorm(sig,sig0,diag(1,2))/dmvnorm(c(sige_mat[t-1,],sigez_mat[t-1,]),sig0,diag(1,2)))
    pp_sig=min(pp_sig,1)
    if(runif(1,0,1)<=pp_sig){
      sige_mat[t,]=sige_1
      sigez_mat[t,]=sigez_1
    } else{
      sige_mat[t,]=sige_mat[t-1,]
      sigez_mat[t,]=sigez_mat[t-1,]
    }
    
    ### sampling of alpha_g from posterior distribution
    dd=(siga_mat[t-1,]^(-1)+(sige_mat[t,]-sigez_mat[t,]^2)^(-1)*t(rep(1,n))%*%rep(1,n))^(-1)
    for(g in 1:G){
      SS=diag(1,n)-lambda_mat[t,]*Wg[[g]]
      YY=SS%*%yg[[g]]-sigez_mat[t,]*Z_mat[g,]
      XX=rep(1,n)*beta_mat[t,1]+Xg[[g]]*beta_mat[t,2]+Wg[[g]]%*%Xg[[g]]%*%beta_mat[t,3]
      alpha_mat[t,g]=rnorm(1,(sige_mat[t,]-sigez_mat[t,]^2)^(-1)*dd*(rep(1,n)%*%(YY-XX)),sqrt(dd))
    }
    ### sampling of sigma_alpha from posterior distribution
    inv_alpha=(rho0+G)/2
    inv_beta=(eta0+sum((alpha_mat[t,])^2))/2
    siga_mat[t,]=1/rgamma(1,shape=inv_alpha,rate=inv_beta)
    end_time=Sys.time()
    ###print result
    cat("gamma =", gamma_mat[t,],"\n");cat("\n")
    cat("beta =", beta_mat[t,],"\n");cat("\n")
    cat("lambda =", lambda_mat[t,],"\n");cat("\n")
    cat("sigma_e =",sige_mat[t,],"\n");cat("\n")
    cat("sigma_ez =",sigez_mat[t,],"\n");cat("\n")
    cat("siga_alpha =",siga_mat[t,],"\n");cat("\n")
    cat("current iteration",t,"is end","\n");cat("\n")
    cat("At",r,"repetition",t,"iteration spend",end_time-start_time,"seconds","\n");cat("\n")
  }
  beta_list[[r]]=beta_mat[200:300,]
  gamma_list[[r]]=gamma_mat[200:300,]
  lambda_list[[r]]=lambda_mat[200:300,]
  sigez_list[[r]]=sigez_mat[200:300,]
  sige_list[[r]]=sige_mat[200:300,]
  siga_list[[r]]=siga_mat[200:300,]
  ed_time=Sys.time()
  cat(r,"repetition of MCMC end!","\n");cat("\n")
  cat("total",iter,"iterations spend",ed_time-st_time,"!","\n");cat("\n")
}


for(r in 1:R){
  beta1=beta_list[[r]][,1]
  beta2=beta_list[[r]][,2]
  beta3=beta_list[[r]][,3]
  lambda=lambda_list[[r]]
  gamma1=gamma_list[[r]][,1]
  gamma2=gamma_list[[r]][,2]
  gamma3=gamma_list[[r]][,3]
  sigmaa=siga_list[[r]]
  sigmaez=sigez_list[[r]]
  sigmae=sige_list[[r]]
}

mean(beta1);sd(beta1)
mean(beta2);sd(beta2)
mean(beta3);sd(beta3)
mean(gamma1);sd(gamma1)
mean(gamma2);sd(gamma2)
mean(gamma3);sd(gamma3)
mean(lambda);sd(lambda)
mean(sigmae);sd(sigmae)
mean(sigmaez);sd(sigez_mat)
mean(sigmaa);sd(sigmaa)