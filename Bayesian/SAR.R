rm(list=ls())
set.seed(2018311133)
library(mvtnorm)
n=30; G=50; R=50
gamma=c(-1.5,0.5,1)
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
      psi=gamma[1]+gamma[2]*C_glist[[g]][i,j]+gamma[3]*abs(z[g,i]-z[g,j])
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

beta0=rep(0,3)
Beta0=5*diag(1,3)
rho0=5; eta0=1;
sig0=c(1,0.5)

iter=5500
lambda_mat=matrix(0,ncol=1,nrow=iter) #save for lambda
beta_mat=matrix(0,ncol=3,nrow=iter) #save for beta
sige_mat=matrix(0,ncol=1,nrow=iter) #save for sigma_e^2
siga_mat=matrix(0,ncol=1,nrow=iter) #save for sigma_a^2
alpha_mat=matrix(0,nrow=iter,ncol=G) #save for alpha

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
lambda_mat[1,]=0.04
sige_mat[1,]=1; siga_mat[1,]=0.3
beta_mat[1,]=c(0.3,0.2,0.4)

### save 
beta1_list=list();beta2_list=list();beta3_list=list()
lambda_list=list();sige_list=list();siga_list=list()


st_time=Sys.time()
for(r in 1:R){
  cat(r,"repetition of MCMC start!","\n")
  for(t in 2:iter){
    start_time=Sys.time()
    ### propose lambda*
    #accept=0
    #while(accept==0){
      if(t<3){
        lambda_1=rnorm(1,lambda_mat[t-1,],0.1)
      } else{
        lambda_1=rnorm(1,lambda_mat[t-1,],cov(as.matrix(lambda_mat[1:t-1,]))*2.38^2)*0.95+
          rnorm(1,lambda_mat[t-1,],0.1)*0.05
      }
      #lambda_1=as.numeric(lambda_1)
      #if(lambda_1>-1/tau && lambda_1<1/tau){
        #accept=1
      #}
    #}
    pp_l=1
    V=(sige_mat[t-1,])*diag(1,n)+siga_mat[t-1,]*rep(1,n)%*%t(rep(1,n))
    for(g in 1:G){
      ### M-H algorithm for sampling lambda
      S_1=diag(1,n)-lambda_1*Wg[[g]]
      S_2=diag(1,n)-lambda_mat[t-1,]*Wg[[g]]
      ep_1=S_1%*%yg[[g]]-(rep(1,n)*beta_mat[t-1,1]+Xg[[g]]*beta_mat[t-1,2]+Wg[[g]]%*%Xg[[g]]%*%beta_mat[t-1,3])
      ep_2=S_2%*%yg[[g]]-(rep(1,n)*beta_mat[t-1,1]+Xg[[g]]*beta_mat[t-1,2]+Wg[[g]]%*%Xg[[g]]%*%beta_mat[t-1,3])
      like_1=det(S_1)*exp(-0.5*t(ep_1)%*%solve(V)%*%(ep_1))
      like_2=det(S_2)*exp(-0.5*t(ep_2)%*%solve(V)%*%(ep_2))
      pp_l=pp_l*(like_1/like_2)
    }
    pp_l=min(pp_l,1)
    
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
      YY=SS%*%yg[[g]]
      XX=cbind(rep(1,n),Xg[[g]],Wg[[g]]%*%Xg[[g]])
      Vg[[g]]=(sigu_mat[t-1,])*diag(1,n)+siga_mat[t-1,]*rep(1,n)%*%t(rep(1,n))
      XVX=XVX+t(XX)%*%solve(Vg[[g]])%*%XX
      XVY=XVY+t(XX)%*%solve(Vg[[g]])%*%YY
    }
    B=solve(solve(Beta0)+XVX)
    nbeta=B%*%(solve(Beta0)%*%beta0+XVY)
    beta_mat[t,]=rmvnorm(1,nbeta,B)
    
    ### sampling of sigmae^2 from posterior
    ep_v=rep(0,n*G)
    for(g in 1:G){
      SS=diag(1,n)-lambda_mat[t,]*Wg[[g]]
      ep=SS%*%yg[[g]]-(rep(1,n)*beta_mat[t,1]+Xg[[g]]*beta_mat[t,2]+Wg[[g]]%*%Xg[[g]]%*%beta_mat[t,3])-rep(1,n)*alpha_mat[t-1,g]
      ep_v[seq((g-1)*n+1,g*n,1)]=ep
    }
    rho1=(rho0+length(ep_v))
    eta1=(eta0+t(ep_v)%*%ep_v)
    sige_mat[t,]=1/rgamma(1,shape=eta1,rate=rho1)

    ### sampling of alpha_g from posterior distribution
    dd=(siga_mat[t-1,]^(-1)+(sige_mat[t,])^(-1)*t(rep(1,n))%*%rep(1,n))^(-1)
    for(g in 1:G){
      SS=diag(1,n)-lambda_mat[t,]*Wg[[g]]
      YY=SS%*%yg[[g]]
      XX=rep(1,n)*beta_mat[t,1]+Xg[[g]]*beta_mat[t,2]+Wg[[g]]%*%Xg[[g]]%*%beta_mat[t,3]
      alpha_mat[t,g]=rnorm(1,(sige_mat[t,])^(-1)*dd*(rep(1,n)%*%(YY-XX)),sqrt(dd))
    }
    ### sampling of sigma_alpha from posterior distribution
    inv_alpha=(rho0+G)/2
    inv_beta=(eta0+sum((alpha_mat[t,])^2))/2
    siga_mat[t,]=1/rgamma(1,shape=inv_alpha,rate=inv_beta)
    
    end_time=Sys.time()
    ###print result
    if(t%%10==0){
      cat("beta =", beta_mat[t,],"\n");cat("\n")
      cat("lambda =", lambda_mat[t,],"\n");cat("\n")
      cat("sigma_e =",sige_mat[t,],"\n");cat("\n")
      cat("siga_alpha =",siga_mat[t,],"\n");cat("\n")
      cat("current iteration",t,"is end","\n")
      cat("At",r,"repetition",t,"iteration spend",end_time-start_time,"seconds","\n");cat("\n")
    }
  }
  beta1_list[[r]]=beta_mat[500:iter,1][seq(1,length(beta_mat[500:iter,1]),10)]
  beta2_list[[r]]=beta_mat[500:iter,2][seq(1,length(beta_mat[500:iter,2]),10)]
  beta3_list[[r]]=beta_mat[500:iter,3][seq(1,length(beta_mat[500:iter,3]),10)]
  lambda_list[[r]]=lambda_mat[500:iter,][seq(1,length(lambda_mat[500:iter,]),10)]
  sige_list[[r]]=sige_mat[500:iter,][seq(1,length(sige_mat[500:iter,]),10)]
  siga_list[[r]]=siga_mat[500:iter,][seq(1,length(siga_mat[500:iter,]),10)]
  cat(r,"repetition of MCMC end!","\n")
}
ed_time=Sys.time()
cat("total",iter,"iterations spend",ed_time-st_time,"!","\n")

beta1_vec=c();beta2_vec=c();beta3_vec=c()
siga_vec=c();sige_vec=c();lambda_vec=c()

for(r in 1:R){
  beta1=beta1_list[[r]][-1]
  beta1_vec=c(beta1_vec,beta1)
  beta2=beta2_list[[r]][-1]
  beta2_vec=c(beta2_vec,beta2)
  beta3=beta3_list[[r]][-1]
  beta3_vec=c(beta3_vec,beta3)
  lambda=lambda_list[[r]][-1]
  lambda_vec=c(lambda_vec,lambda)
  sigmaa=siga_list[[r]][-1]
  siga_vec=c(siga_vec,sigmaa)
  sigmae=sige_list[[r]][-1]
  sige_vec=c(sige_vec,sigmae)
}

round(mean(beta1_vec),3);round(sd(beta1_vec),3)
round(mean(beta2_vec),3);round(sd(beta2_vec),3)
round(mean(beta3_vec),3);round(sd(beta3_vec),3)
round(mean(lambda_vec),3);round(sd(lambda_vec),3)
round(mean(sige_vec),3);round(sd(sige_vec),3)
round(mean(siga_vec),3);round(sd(siga_vec),3)

par(mfrow=c(2,3))

plot(density(beta1_vec),type="l",xlab="",main=expression(paste(beta[1])));abline(v=0.5,col="red")
plot(density(beta2_vec),type="l",xlab="",main=expression(paste(beta[2])));abline(v=0.5,col="red")
plot(density(beta3_vec),type="l",xlab="",main=expression(paste(beta[3])));abline(v=0.5,col="red")
plot(density(lambda_vec),type="l",xlab="",main=expression(lambda));abline(v=0.05,col="red")
plot(density(sige_vec),type="l",xlab="",main=expression(paste(sigma[epsilon]^2)));abline(v=1.25,col="red")
plot(density(siga_vec),type="l",xlab="",main=expression(paste(sigma[alpha]^2)));abline(v=0.5,col="red")
