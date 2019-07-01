rm(list=ls())
set.seed(2018311133)
library(mvtnorm)
n=30; G=50; R=50
gamma=c(-1.5,0.5,1)
beta=c(0.5,0.5,0.5)
delta=c(0.5,0.5)
sigma=matrix(c(1,0,0,1),ncol=2)
lambda=0.05
sig_alpha=0.5;mu=c(0.5,0)
z=matrix(0,ncol=n,nrow=G)
u=matrix(0,ncol=n,nrow=G)
C_g=matrix(0,ncol=n,nrow=n)
C_glist=list()

for(g in 1:G){
  mat=rmvnorm(n,mu,sigma)
  u1=runif(n,0,1); u2=runif(n,0,1)
  for(i in 1:n){
    for(j in 1:n){
      if(u1[i]>0.5 && u2[j]>0.5){
        C_g[i,j]=1
      } else if(u1[i]<0.5 && u2[j]<0.5){
        C_g[i,j]=1
      } else{
        C_g[i,j]=0
      }
      if(i==j){
        C_g[i,j]=0
      }
    }
    z[g,i]=mat[i,1]
    u[g,i]=mat[i,2]
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
  Wg[[g]]=w
}

Xg=list();yg=list()

for(g in 1:G){
  SS=diag(1,n)-lambda*Wg[[g]]
  x=rnorm(n,0,1)
  Xg[[g]]=x
  alpha=rnorm(1,0,sqrt(sig_alpha))
  XX=cbind(rep(1,n),Xg[[g]],Wg[[g]]%*%Xg[[g]])
  ZZ=cbind(z[g,],Wg[[g]]%*%z[g,])
  yg[[g]]=solve(SS)%*%(XX%*%(beta)+ZZ%*%delta+rep(1,n)*alpha+u[g,])
}

iter=5500
mini=c();maxi=c();Tau_g=c()
for(g in 1:G){
  for(i in 1:n){
    mini[i]=sum(Wg[[g]][i,])
    maxi[i]=sum(Wg[[g]][,i])
    Tau_g=min(mini[which.max(mini)],maxi[which.max(maxi)])
    tau=Tau_g[which.max(Tau_g)]
  }
}

### save 
beta1_list=list();beta2_list=list();beta3_list=list()
gamma1_list=list();gamma2_list=list();gamma3_list=list()
lambda_list=list();siga_list=list();sigu_list=list()
delta1_list=list();delta2_list=list()

st_time=Sys.time()
for(r in 1:R){
  c_1=1;c_2=1;c_3=1;c_4=1
  acc_1=0;acc_2=0;acc_3=0;acc_4=matrix(0,nrow=G,ncol=1)
  
  ### assign parameter in prior distribution ###
  gamma0=rep(0,3); Gamma0=5*diag(1,3)
  beta0=rep(0,3); Beta0=5*diag(1,3)
  rho0=5; eta0=1;
  delta0=c(0,0); Delta0=5*diag(1,2)
  M0=2
  nu0=5;omega0=1
  
  Z_mat=matrix(0,ncol=n,nrow=G) #save for Z
  gamma_mat=matrix(0,ncol=3,nrow=iter) #save for gamma
  lambda_mat=matrix(0,ncol=1,nrow=iter) #save for lambda
  beta_mat=matrix(0,ncol=3,nrow=iter) #save for beta
  sige_mat=matrix(0,ncol=1,nrow=iter) #save for sigma_e^2
  siga_mat=matrix(0,ncol=1,nrow=iter) #save for sigma_a^2
  delta_mat=matrix(0,ncol=2,nrow=iter) #save for delta
  mu_mat=matrix(0,ncol=n,nrow=iter) #save for mu
  sigu_mat=matrix(0,ncol=1,nrow=iter) #save for siga_u^2
  alpha_mat=matrix(0,nrow=iter,ncol=G) #save for alpha
  acc_rate1=matrix(0,nrow=iter,1) #save acceptance rate for gamma
  acc_rate2=matrix(0,nrow=iter,1) #save accpetance rate for lambda
  acc_rate3=matrix(0,nrow=iter,1) #save acceptance rate for delta
  acc_rate4=matrix(0,nrow=G,ncol=iter)
  
  ### starting value of draw ###
  gamma_mat[1,]=c(-1,0.5,1);
  sigu_mat[1,]=0.5; siga_mat[1,]=0.4
  beta_mat[1,]=c(0.3,0.3,0.4);mu_mat[1,]=rep(0.5,n)
  delta_mat[1,]=c(0.4,0.4)
  
  cat(r,"repetition of MCMC start!","\n")
  for(t in 2:iter){
    start_time=Sys.time()
    
    ### propose lambda*
    
    accept=0
    while(accept==0){
      if(t<3){
        lambda_1=rnorm(1,lambda_mat[t-1,],0.1^2)
      } else{
        lambda_1=rnorm(1,lambda_mat[t-1,],cov(as.matrix(lambda_mat[1:t-1,]))*2.38^2)*0.95+
          rnorm(1,lambda_mat[t-1,],0.1^2)*0.05
      }
      lambda_1=as.numeric(lambda_1)
      if(lambda_1>-1/tau && lambda_1<1/tau){
        accept=1
      }
    }
    pp_l=1
    ### propose gamma*
    
    if(t<6){
      gamma_1=rmvnorm(1,gamma_mat[t-1,],diag(1,3)*0.1^2/3)
    } else{
      gamma_1=rmvnorm(1,gamma_mat[t-1,],cov(as.matrix(gamma_mat[1:t-1,]))*2.38^2/3)*0.95+
        rmvnorm(1,gamma_mat[t-1,],diag(1,3)*0.1^2/3)*0.05
    }
    pp_G=1
    
    ### propose delta*
    
    accept=0
    while(accept==0){
      if(t<6){
        delta_1=rmvnorm(1,delta_mat[t-1,],diag(1,2)*0.1^2/2)
      } else{
        delta_1=rmvnorm(1,delta_mat[t-1,],cov(as.matrix(delta_mat[1:t-1,]))*2.38^2/2)*0.95+
          rmvnorm(1,delta_mat[t-1,],diag(1,2)*0.1^2/2)*0.05
      }
      if(delta_1[1]>0 && delta_1[2]>0){
        accept=1
      }
    }    
    
    ### M-H algorithm for sampling Z ###
    V=(sigu_mat[t-1,])*diag(1,n)+siga_mat[t-1,]*rep(1,n)%*%t(rep(1,n))

    for(g in 1:G){
      zz_1=Z_mat[g,]
      acc_4v=0
      for(v in 1:n){
        zz_1[v]=rnorm(1,0,1)*c_4+Z_mat[g,v]
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
          }
           else{
            psi_1=gamma_mat[t-1,1]+gamma_mat[t-1,2]*C_glist[[g]][i,v]-gamma_mat[t-1,3]*abs(zz_1[i]-zz_1[v])
            psi_2=gamma_mat[t-1,1]+gamma_mat[t-1,2]*C_glist[[g]][i,v]-gamma_mat[t-1,3]*abs(Z_mat[g,i]-Z_mat[g,v])
            p_1=exp(psi_1*Wg[[g]][i,v])/(1+exp(psi_1))
            p_2=exp(psi_2*Wg[[g]][i,v])/(1+exp(psi_2))
            pp=pp*(p_1/p_2)
           }
        }
        SS=diag(1,n)-lambda_mat[t-1,]*Wg[[g]]
        XX=cbind(rep(1,n),Xg[[g]],Wg[[g]]%*%Xg[[g]])
        ep=SS%*%(yg[[g]])-XX%*%beta_mat[t-1,]
        d1=zz_1*delta_mat[t-1,1]+Wg[[g]]%*%zz_1*delta_mat[t-1,2]
        d2=Z_mat[g,]*delta_mat[t-1,1]+Wg[[g]]%*%Z_mat[g,]*delta_mat[t-1,2]
        d3=mu_mat[t-1,]*delta_mat[t-1,1]+Wg[[g]]%*%mu_mat[t-1,]*delta_mat[t-1,2]
        like_Y1=exp(-0.5*t(ep-d1-d3)%*%solve(V)%*%(ep-d1-d3))
        like_Y2=exp(-0.5*t(ep-d2-d3)%*%solve(V)%*%(ep-d2-d3))
        pp=pp*(like_Y1/like_Y2)*(dnorm(zz_1[v])/dnorm(Z_mat[g,v]))
        pp=min(pp,1)
        
        if(runif(1,0,1)<=pp){
          Z_mat[g,v]=zz_1[v]
          acc_4v=acc_4v+1
        } 
        zz_1=Z_mat[g,]
      }
      if(acc_4v>=n/2){
        acc_4[g,]=acc_4[g,]+1
      } 
      acc_rate4[g,t]=acc_4[g,]/t
      
      ### M-H algorithm for sampling gamma and lambda
      
      pp=1
      for(i in 1:n){
        for(j in 1:n){
          psi_1=gamma_1[1]+gamma_1[2]*C_glist[[g]][i,j]-gamma_1[3]*abs(Z_mat[g,i]-Z_mat[g,j])
          psi_2=gamma_mat[t-1,1]+gamma_mat[t-1,2]*C_glist[[g]][i,j]-gamma_mat[t-1,3]*abs(Z_mat[g,i]-Z_mat[g,j])
          p_1=exp(psi_1*Wg[[g]][i,j])/(1+exp(psi_1))
          p_2=exp(psi_2*Wg[[g]][i,j])/(1+exp(psi_2))
          if(i==j){
            p_1=1
            p_2=1
          }
          pp=pp*(p_1/p_2)
        }
      }
      pp_G=pp_G*pp
      
      S_1=diag(1,n)-lambda_1*Wg[[g]]
      S_2=diag(1,n)-lambda_mat[t-1,]*Wg[[g]]
      XX=cbind(rep(1,n),Xg[[g]],Wg[[g]]%*%Xg[[g]])
      ZZ=cbind(Z_mat[g,],Wg[[g]]%*%Z_mat[g,])
      MM=cbind(mu_mat[t-1,],Wg[[g]]%*%mu_mat[t-1,])
      ep_1=S_1%*%yg[[g]]-XX%*%beta_mat[t-1,]
      ep_2=S_2%*%yg[[g]]-XX%*%beta_mat[t-1,]
      del=ZZ%*%delta_mat[t-1,]
      mu=MM%*%delta_mat[t-1,]
      like_1=det(S_1)*exp(-0.5*t(ep_1-del-mu)%*%solve(V)%*%(ep_1-del-mu))
      like_2=det(S_2)*exp(-0.5*t(ep_2-del-mu)%*%solve(V)%*%(ep_2-del-mu))
      pp_l=pp_l*(like_1/like_2)
    }
    pp_G=pp_G*(dmvnorm(gamma_1,gamma0,Gamma0)/dmvnorm(gamma_mat[t-1,],gamma0,Gamma0))
    pp_G=min(pp_G,1)
    
    if(runif(1,0,1)<=pp_G){
      gamma_mat[t,]=gamma_1
      acc_1=acc_1+1
    } else{
      gamma_mat[t,]=gamma_mat[t-1,]
    }
    acc_rate1[t,]=acc_1/t
    
    pp_l=min(pp_l,1)
    if(runif(1,0,1)<=pp_l){
      lambda_mat[t,]=lambda_1
      acc_2=acc_2+1
    } else{
      lambda_mat[t,]=lambda_mat[t-1,]
    }
    acc_rate2[t,]=acc_2/t
    
    if(mean(acc_rate4[,t])<0.4){
      c_4=c_4/1.1
    }
    if(mean(acc_rate4[,t])>0.6){
      c_4=c_4*1.1
    }
    ### sample of beta from posterior distribution
    
    XVX=matrix(0,ncol=length(beta0),nrow=length(beta0))
    XVY=matrix(0,nrow=length(beta0),ncol=1)
    for(g in 1:G){
      SS=diag(1,n)-lambda_mat[t,]*Wg[[g]]
      MM=cbind(mu_mat[t-1,],Wg[[g]]%*%mu_mat[t-1,])
      ZZ=cbind(Z_mat[g,],Wg[[g]]%*%Z_mat[g,])
      YY=SS%*%yg[[g]]-ZZ%*%delta_mat[t-1,]-MM%*%delta_mat[t-1,]
      XX=cbind(rep(1,n),Xg[[g]],Wg[[g]]%*%Xg[[g]])
      XVX=XVX+t(XX)%*%solve(V)%*%XX
      XVY=XVY+t(XX)%*%solve(V)%*%YY
    }
    B=solve(solve(Beta0)+XVX)
    nbeta=B%*%(solve(Beta0)%*%beta0+XVY)
    beta_mat[t,]=rmvnorm(1,nbeta,B)
    
    ### sampling of delta from posterior
    
    pp_d=1
    for(g in 1:G){
      S=diag(1,n)-lambda_mat[t,]*Wg[[g]]
      XX=cbind(rep(1,n),Xg[[g]],Wg[[g]]%*%Xg[[g]])
      ZZ=cbind(Z_mat[g,],Wg[[g]]%*%Z_mat[g,])
      MM=cbind(mu_mat[t-1,],Wg[[g]]%*%mu_mat[t-1,])
      ep=S%*%yg[[g]]-XX%*%beta_mat[t,]
      del1=ZZ%*%t(delta_1)
      del2=ZZ%*%delta_mat[t-1,]
      mu1=MM%*%t(delta_1)
      mu2=MM%*%delta_mat[t-1,]
      like_1=exp(-0.5*t(ep-del1-mu1)%*%solve(V)%*%(ep-del1-mu1))
      like_2=exp(-0.5*t(ep-del2-mu2)%*%solve(V)%*%(ep-del2-mu2))
      pp_d=pp_d*(like_1/like_2)
    }
    pp_d=pp_d*(dmvnorm(delta_1,delta0,Delta0)/dmvnorm(delta_mat[t-1,],delta0,Delta0))
    
    pp_d=min(pp_d,1)
    if(runif(1,0,1)<=pp_d){
      delta_mat[t,]=delta_1
      acc_3=acc_3+1
    } else{
      delta_mat[t,]=delta_mat[t-1,]
    }
    acc_rate3[t,]=acc_3/t
    
    ### sampling of mu_z from posterior
    
    new_mean=sum(Z_mat+mu_mat[t-1,1])/(n*G+(1/M0))
    new_var=1/(n*G+(1/M0))
    mu_mat[t,]=rnorm(1,new_mean,sqrt(new_var))
    
    ### sampling of sigmau^2 from posterior
    
    UG=0
    for(g in 1:G){
      SS=diag(1,n)-lambda_mat[t,]*Wg[[g]]
      XX=cbind(rep(1,n),Xg[[g]],Wg[[g]]%*%Xg[[g]])
      ZZ=cbind(Z_mat[g,],Wg[[g]]%*%Z_mat[g,])
      MM=cbind(mu_mat[t,],Wg[[g]]%*%mu_mat[t,])
      ep=SS%*%yg[[g]]-XX%*%beta_mat[t,]
      del=ZZ%*%delta_mat[t,]
      mu=MM%*%delta_mat[t,]
      ug=ep-del-mu
      UG=UG+t(ug)%*%ug
    }
    sigu_alpha=(nu0+n*G)/2
    sigu_beta=(omega0+UG)/2
    sigu_mat[t,]=1/rgamma(1,shape=sigu_alpha,rate=sigu_beta)
    
    ### sampling of alpha_g from posterior distribution
    
    Dg=(siga_mat[t-1,]^(-1)+(sigu_mat[t,])^(-1)*t(rep(1,n))%*%rep(1,n))^(-1)
    for(g in 1:G){
      SS=diag(1,n)-lambda_mat[t,]*Wg[[g]]
      ZZ=cbind(Z_mat[g,],Wg[[g]]%*%Z_mat[g,])
      MM=cbind(mu_mat[t,],Wg[[g]]%*%mu_mat[t,])
      YY=SS%*%yg[[g]]-ZZ%*%(delta_mat[t,])-MM%*%delta_mat[t,]
      XX=cbind(rep(1,n),Xg[[g]],Wg[[g]]%*%Xg[[g]])
      al_hat=(sigu_mat[t,]^(-1))*Dg*(rep(1,n)%*%(YY-XX%*%beta_mat[t,]))
      alpha_mat[t,g]=rnorm(1,al_hat,sqrt(Dg))
    }
    
    ### sampling of sigma_alpha from posterior distribution
    
    inv_alpha=(rho0+G)/2
    inv_beta=(eta0+sum((alpha_mat[t,])^2))/2
    siga_mat[t,]=1/rgamma(1,shape=inv_alpha,rate=inv_beta)
    
    end_time=Sys.time()
    ###print result
    cat("gamma =", round(gamma_mat[t,],3),"\n")
    cat("beta =", round(beta_mat[t,],3),"\n")
    cat("lambda =", round(lambda_mat[t,],3),"\n")
    cat("delta =",round(delta_mat[t,],3),"\n")
    cat("sigma_u =",round(sigu_mat[t,],3),"\n")
    cat("mu_z =",round(mu_mat[t,1],3),"\n")
    cat("siga_alpha =",round(siga_mat[t,],3),"\n")
    cat("gamma acceptance rate=",round(acc_rate1[t,],2),"\n")
    cat("lambda acceptance rate=",round(acc_rate2[t,],2),"\n")
    cat("delta acceptance rate=",round(acc_rate3[t,],2),"\n")
    cat("current iteration",t,"is end","\n")
    cat("At",r,"repetition",t,"iteration spend",end_time-start_time,"seconds","\n");cat("\n")
  }
  beta1_list[[r]]=beta_mat[500:iter,1][seq(1,length(beta_mat[500:iter,1]),10)]
  beta2_list[[r]]=beta_mat[500:iter,2][seq(1,length(beta_mat[500:iter,2]),10)]
  beta3_list[[r]]=beta_mat[500:iter,3][seq(1,length(beta_mat[500:iter,3]),10)]
  gamma1_list[[r]]=gamma_mat[500:iter,1][seq(1,length(gamma_mat[500:iter,1]),10)]
  gamma2_list[[r]]=gamma_mat[500:iter,2][seq(1,length(gamma_mat[500:iter,2]),10)]
  gamma3_list[[r]]=gamma_mat[500:iter,3][seq(1,length(gamma_mat[500:iter,3]),10)]
  lambda_list[[r]]=lambda_mat[500:iter,][seq(1,length(lambda_mat[500:iter,]),10)]
  delta1_list[[r]]=delta_mat[500:iter,1][seq(1,length(delta_mat[500:iter,1]),10)]
  delta2_list[[r]]=delta_mat[500:iter,2][seq(1,length(delta_mat[500:iter,2]),10)]
  mu_list[[r]]=mu_mat[500:iter,][seq(1,length(mu_mat[500:iter,]),10)]
  sigu_list[[r]]=sigu_mat[500:iter,][seq(1,length(sigu_mat[500:iter,]),10)]
  siga_list[[r]]=siga_mat[500:iter,][seq(1,length(siga_mat[500:iter,]),10)]
  cat(r,"repetition of MCMC end!","\n")
}
ed_time=Sys.time()
cat("total",iter,"iterations spend",ed_time-st_time,"!","\n")

beta1_vec=c();beta2_vec=c();beta3_vec=c()
gamma1_vec=c();gamma2_vec=c();gamma3_vec=c()
siga_vec=c();muz_vec=c();lambda_vec=c()
delta1_vec=c();delta2_vec=c();sigu_vec=c()

for(r in 1:R){
  beta1=beta1_list[[r]][-1]
  beta1_vec=c(beta1_vec,beta1)
  beta2=beta2_list[[r]][-1]
  beta2_vec=c(beta2_vec,beta2)
  beta3=beta3_list[[r]][-1]
  beta3_vec=c(beta3_vec,beta3)
  lambda=lambda_list[[r]][-1]
  lambda_vec=c(lambda_vec,lambda)
  gamma1=gamma1_list[[r]][-1]
  gamma1_vec=c(gamma1_vec,gamma1)
  gamma2=gamma2_list[[r]][-1]
  gamma2_vec=c(gamma2_vec,gamma2)
  gamma3=gamma3_list[[r]][-1]
  gamma3_vec=c(gamma3_vec,gamma3)
  sigmaa=siga_list[[r]][-1]
  siga_vec=c(siga_vec,sigmaa)
  sigmau=sigu_list[[r]][-1]
  sigu_vec=c(sigu_vec,sigmau)
  muz=mu_list[[r]][-1]
  muz_vec=c(muz_vec,muz)
  delta1=delta1_list[[r]][-1]
  delta1_vec=c(delta1_vec,delta1)
  delta2=delta2_list[[r]][-1]
  delta2_vec=c(delta2_vec,delta2)
}

round(mean(beta1_vec),3);round(sd(beta1_vec),3)
round(mean(beta2_vec),3);round(sd(beta2_vec),3)
round(mean(beta3_vec),3);round(sd(beta3_vec),3)
round(mean(gamma1_vec),3);round(sd(gamma1_vec),3)
round(mean(gamma2_vec),3);round(sd(gamma2_vec),3)
round(mean(gamma3_vec),3);round(sd(gamma3_vec),3)
round(mean(lambda_vec),3);round(sd(lambda_vec),3)
round(mean(sigu_vec),3);round(sd(sigu_vec),3)
round(mean(muz_vec),3);round(sd(muz_vec),3)
round(mean(siga_vec),3);round(sd(siga_vec),3)
round(mean(delta1_vec),3);round(sd(delta1_vec),3)
round(mean(delta2_vec),3);round(sd(delta2_vec),3)
par(mfrow=c(2,3))

plot(density(beta1_vec),type="l",xlab="",main=expression(paste(beta[1])))
plot(density(beta2_vec),type="l",xlab="",main=expression(paste(beta[2])))
plot(density(beta3_vec),type="l",xlab="",main=expression(paste(beta[3])))
plot(density(gamma1_vec),type="l",xlab="",main=expression(paste(gamma[1])))
plot(density(gamma2_vec),type="l",xlab="",main=expression(paste(gamma[2])))
plot(density(gamma3_vec),type="l",xlab="",main=expression(paste(gamma[3])))

plot(density(lambda_vec),type="l",xlab="",main=expression(lambda))
plot(density(delta1_vec),type="l",xlab="",main=expression(paste(delta[1])))
plot(density(delta2_vec),type="l",xlab="",main=expression(paste(delta[2])))
plot(density(sigu_vec),type="l",xlab="",main=expression(paste(sigma[u]^2)))
plot(density(muz_vec),type="l",xlab="",main=expression(paste(mu[z])))
plot(density(siga_vec),type="l",xlab="",main=expression(paste(sigma[alpha]^2)))

