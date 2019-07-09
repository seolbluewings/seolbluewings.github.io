rm(list=ls())
set.seed(2081311133)
library(mvtnorm)
WC=read.csv("C://seolbluewings.github.io/Bayesian/WC_data.csv",header=T)
y=WC[,4:5]; x=WC[,c(4:5,7:13,15,18)]
Xg=list();yg=list();zg=list()
G=128

### allocate group ###
### calculate group size ###
mg=c()
for(g in 1:G){
    yg[[g]]=as.vector(y[WC$group==g,2])
    Xg[[g]]=as.data.frame(x[WC$group==g,-c(1:2)])
    mg[g]=length(yg[[g]])
    zg[[g]]=as.vector(x[WC$group==g,2])
}

Z_list=list()
Cg1_list=list(); Cg2_list=list(); Cg3_list=list(); Cg4_list=list()
for(g in 1:G){
  z_mat=zg[[g]]
  C_g1=matrix(0,ncol=mg[g],nrow=mg[g])
  C_g2=matrix(0,ncol=mg[g],nrow=mg[g])
  C_g3=matrix(0,ncol=mg[g],nrow=mg[g])
  C_g4=matrix(0,ncol=mg[g],nrow=mg[g])
  for(i in 1:mg[g]){
    for(j in 1:mg[g]){
      #observed characteristic Shot
      if(Xg[[g]][i,]$Shot>=2 && Xg[[g]][j,]$Shot>=2){
        C_g1[i,j]=1
      } else if(Xg[[g]][i,]$Shot==0 && Xg[[g]][j,]$Shot==0){
        C_g1[i,j]=1
      } else{
        C_g1[i,j]=0
      }
      
      #observed characteristic Dribble
      if(Xg[[g]][i,]$Dribble>=2 && Xg[[g]][j,]$Dribble>=2){
        C_g2[i,j]=1
      } else if(Xg[[g]][i,]$Dribble==0 && Xg[[g]][j,]$Dribble==0){
        C_g2[i,j]=1
      } else{
        C_g2[i,j]=0
      }
      #observed characteristic Clearance
      if(Xg[[g]][i,]$Clearance>=2 && Xg[[g]][j,]$Clearance>=2){
        C_g3[i,j]=1
      } else if(Xg[[g]][i,]$Clearance==0 && Xg[[g]][j,]$Clearance==0){
        C_g3[i,j]=1
      } else{
        C_g3[i,j]=0
      }
      
      if(Xg[[g]][i,]$Duel>=2 && Xg[[g]][j,]$Duel>=2){
        C_g4[i,j]=1
      } else if(Xg[[g]][i,]$Duel==0 && Xg[[g]][j,]$Duel==0){
        C_g4[i,j]=1
      } else{
        C_g4[i,j]=0
      }
    }
    
  }
  Z_list[[g]]=z_mat
  Cg1_list[[g]]=C_g1; Cg2_list[[g]]=C_g2
  Cg3_list[[g]]=C_g3; Cg4_list[[g]]=C_g4
}

iter=5000
R=30
for(g in 1:G){
  Xg[[g]]=as.matrix(Xg[[g]])
}

### save 
beta1_list=list(); beta2_list=list(); beta3_list=list()
beta4_list=list(); beta5_list=list(); beta6_list=list()
beta7_list=list(); beta8_list=list(); beta9_list=list()
beta10_list=list(); beta11_list=list(); beta12_list=list()
beta13_list=list(); beta14_list=list(); beta15_list=list()
beta16_list=list(); beta17_list=list(); beta18_list=list()
beta19_list=list()
gamma1_list=list(); gamma2_list=list(); gamma3_list=list()
gamma4_list=list(); gamma5_list=list(); gamma6_list=list()
lambda_list=list()
sige_list=list();siga_list=list();sigez_list=list()

st_time=Sys.time()
for(r in 1:R){
  c_1=1;c_2=1;c_3=1;c_4=1
  acc_1=0;acc_2=0;acc_3=0;acc_4=matrix(0,nrow=G,ncol=1)
  
  ### assign parameter in prior distribution ###
  gamma0=rep(0,6); Gamma0=5*diag(1,6)
  beta0=rep(0,2*dim(Xg[[1]])[2]+1); Beta0=5*diag(1,2*dim(Xg[[1]])[2]+1)
  rho0=5; eta0=1; sig0=c(0,0)
  
  Zlist=list() #save for Z
  gamma_mat=matrix(0,ncol=6,nrow=iter) #save for gamma
  lambda_mat=matrix(0,ncol=1,nrow=iter) #save for lambda
  beta_mat=matrix(0,ncol=2*dim(Xg[[1]])[2]+1,nrow=iter) #save for beta
  sige_mat=matrix(0,ncol=1,nrow=iter) #save for sigma_e^2
  siga_mat=matrix(0,ncol=1,nrow=iter) #save for sigma_a^2
  sigez_mat=matrix(0,ncol=1,nrow=iter) #save for sigma_ez^2
  alpha_mat=matrix(0,nrow=iter,ncol=G) #save for alpha
  acc_rate1=matrix(0,nrow=iter,1) #acceptance rate gamma
  acc_rate2=matrix(0,nrow=iter,1) #accpetance rate lambda
  acc_rate3=matrix(0,nrow=iter,1) #acceptance rate sigma
  acc_rate4=matrix(0,nrow=G,ncol=iter)
  
  ### starting value of draw ###
  gamma_mat[1,]=c(-1,0.5,0.5,0.5,0.5,1);
  sige_mat[1,]=1; sigez_mat[1,]=0.5; siga_mat[1,]=0.3
  beta_mat[1,]=rep(0,2*dim(Xg[[1]])[2]+1)
  
  cat(r,"repetition of MCMC start!","\n")
  for(t in 2:iter){
    start_time=Sys.time()
    
    
    Wg=list()
    
    for(g in 1:G){
      w=matrix(0,ncol=mg[g],nrow=mg[g])
      p=matrix(0,ncol=mg[g],nrow=mg[g])
      for(i in 1:mg[g]){
        for(j in 1:mg[g]){
          psi=gamma_mat[t-1,1]+gamma_mat[t-1,2]*Cg1_list[[g]][i,j]
          +gamma_mat[t-1,3]*Cg2_list[[g]][i,j]
          +gamma_mat[t-1,4]*Cg3_list[[g]][i,j]
          +gamma_mat[t-1,5]*Cg4_list[[g]][i,j]
          -gamma_mat[t-1,6]*abs(Z_list[[g]][i]-Z_list[[g]][j])
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
    
    mini=c();maxi=c();Tau_g=c()
    for(g in 1:G){
      for(i in 1:mg[g]){
        mini[i]=sum(Wg[[g]][i,])
        maxi[i]=sum(Wg[[g]][,i])
        Tau_g=min(mini[which.max(mini)],maxi[which.max(maxi)])
        tau=Tau_g[which.max(Tau_g)]
      }
    }
    
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
    pp_l=1
    ### propose gamma*
    
    if(t<6){
      gamma_1=rmvnorm(1,gamma_mat[t-1,],diag(1,6)*0.1^2/3)
    } else{
      gamma_1=rmvnorm(1,gamma_mat[t-1,],cov(as.matrix(gamma_mat[1:t-1,]))*2.38^2/3)*0.95+
        rmvnorm(1,gamma_mat[t-1,],diag(1,6)*0.1^2/3)*0.05
    }
    pp_G=1
    
    ### M-H algorithm for sampling Z ###
    #V=(sige_mat[t-1,]-sigez_mat[t-1,]^2)*diag(1,mg[g])+siga_mat[t-1,]*rep(1,mg[g])%*%t(rep(1,mg[g]))
    for(g in 1:G){
      Z_vec=rep(0,mg[g])
      Zlist[[g]]=Z_vec
      zz_1=Z_vec
      acc_4v=0
      for(v in 1:mg[g]){
        zz_1[v]=rnorm(1,0,1)*c_4+Z_vec[v]
        pp=1
        for(i in 1:mg[g]){
          if(i==v){
            for(j in 1:mg[g]){
              if(j==v){
                p_1=1
                p_2=1
              } else{
                psi_1=gamma_mat[t-1,1]+gamma_mat[t-1,2]*Cg1_list[[g]][i,j]
                +gamma_mat[t-1,3]*Cg2_list[[g]][i,j]
                +gamma_mat[t-1,4]*Cg3_list[[g]][i,j]
                +gamma_mat[t-1,5]*Cg4_list[[g]][i,j]
                -gamma_mat[t-1,6]*abs(zz_1[i]-zz_1[j])
                psi_2=gamma_mat[t-1,1]+gamma_mat[t-1,2]*Cg1_list[[g]][i,j]
                +gamma_mat[t-1,3]*Cg2_list[[g]][i,j]
                +gamma_mat[t-1,4]*Cg3_list[[g]][i,j]
                +gamma_mat[t-1,5]*Cg4_list[[g]][i,j]
                -gamma_mat[t-1,6]*abs(Zlist[[g]][i]-Zlist[[g]][j])
                
                p_1=exp(psi_1*Wg[[g]][i,j])/(1+exp(psi_1))
                p_2=exp(psi_2*Wg[[g]][i,j])/(1+exp(psi_2))
              }
              pp=pp*(p_1/p_2)
            }
          } else{
            psi_1=gamma_mat[t-1,1]+gamma_mat[t-1,2]*Cg1_list[[g]][i,j]
            +gamma_mat[t-1,3]*Cg2_list[[g]][i,j]
            +gamma_mat[t-1,4]*Cg3_list[[g]][i,j]
            +gamma_mat[t-1,5]*Cg4_list[[g]][i,j]
            -gamma_mat[t-1,6]*abs(zz_1[i]-zz_1[v])
            psi_2=gamma_mat[t-1,1]+gamma_mat[t-1,2]*Cg1_list[[g]][i,j]
            +gamma_mat[t-1,3]*Cg2_list[[g]][i,j]
            +gamma_mat[t-1,4]*Cg3_list[[g]][i,j]
            +gamma_mat[t-1,5]*Cg4_list[[g]][i,j]
            -gamma_mat[t-1,6]*abs(Zlist[[g]][i]-Zlist[[g]][v])
            p_1=exp(psi_1*Wg[[g]][i,v])/(1+exp(psi_1))
            p_2=exp(psi_2*Wg[[g]][i,v])/(1+exp(psi_2))
            pp=pp*(p_1/p_2)
          }
        }
        V=(sige_mat[t-1,]-sigez_mat[t-1,]^2)*diag(1,mg[g])+siga_mat[t-1,]*rep(1,mg[g])%*%t(rep(1,mg[g]))
        SS=diag(1,mg[g])-lambda_mat[t-1,]*Wg[[g]]
        XX=cbind(rep(1,mg[g]),Xg[[g]],Wg[[g]]%*%Xg[[g]])
        ep=SS%*%(yg[[g]])-XX%*%beta_mat[t-1,]
        like_Y1=exp(-0.5*t(ep-sigez_mat[t-1,]*zz_1)%*%solve(V)%*%(ep-sigez_mat[t-1,]*zz_1))
        like_Y2=exp(-0.5*t(ep-sigez_mat[t-1,]*Zlist[[g]])%*%solve(V)%*%(ep-sigez_mat[t-1,]*Zlist[[g]]))
        pp=pp*(like_Y1/like_Y2)*(dmvnorm(zz_1[v])/dmvnorm(Zlist[[g]][v]))
        pp=min(pp,1)
        if(runif(1,0,1)<=pp){
          Zlist[[g]][v]=zz_1[v]
          acc_4v=acc_4v+1
        }
        zz_1=Zlist[[g]]
      }
      if(acc_4v>=mg[g]/2){
        acc_4[g,]=acc_4[g,]+1
      } 
      acc_rate4[g,t]=acc_4[g,]/t
      
      ### M-H algorithm for sampling gamma and lambda
      pp=1
      for(i in 1:mg[g]){
        for(j in 1:mg[g]){
          psi_1=gamma_1[1]+gamma_1[2]*Cg1_list[[g]][i,j]
          +gamma_1[3]*Cg2_list[[g]][i,j]
          +gamma_1[4]*Cg3_list[[g]][i,j]
          +gamma_1[5]*Cg4_list[[g]][i,j]
          -gamma_1[6]*abs(Zlist[[g]][i]-Zlist[[g]][j])
          psi_2=gamma_mat[t-1,1]+gamma_mat[t-1,2]*Cg1_list[[g]][i,j]
          +gamma_mat[t-1,3]*Cg2_list[[g]][i,j]
          +gamma_mat[t-1,4]*Cg3_list[[g]][i,j]
          +gamma_mat[t-1,5]*Cg4_list[[g]][i,j]
          -gamma_mat[t-1,6]*abs(Zlist[[g]][i]-Zlist[[g]][j])
          
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
      
      S_1=diag(1,mg[g])-lambda_1*Wg[[g]]
      S_2=diag(1,mg[g])-lambda_mat[t-1,]*Wg[[g]]
      XX=cbind(rep(1,mg[g]),Xg[[g]],Wg[[g]]%*%Xg[[g]])
      V=(sige_mat[t-1,]-sigez_mat[t-1,]^2)*diag(1,mg[g])+siga_mat[t-1,]*rep(1,mg[g])%*%t(rep(1,mg[g]))
      ep_1=S_1%*%yg[[g]]-XX%*%beta_mat[t-1,]
      ep_2=S_2%*%yg[[g]]-XX%*%beta_mat[t-1,]
      like_1=det(S_1)*exp(-0.5*t(ep_1-sigez_mat[t-1,]*Zlist[[g]])%*%solve(V)%*%(ep_1-sigez_mat[t-1,]*Zlist[[g]]))
      like_2=det(S_2)*exp(-0.5*t(ep_2-sigez_mat[t-1,]*Zlist[[g]])%*%solve(V)%*%(ep_2-sigez_mat[t-1,]*Zlist[[g]]))
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
      V=(sige_mat[t-1,]-sigez_mat[t-1,]^2)*diag(1,mg[g])+siga_mat[t-1,]*rep(1,mg[g])%*%t(rep(1,mg[g]))
      SS=diag(1,mg[g])-lambda_mat[t,]*Wg[[g]]
      YY=SS%*%yg[[g]]-sigez_mat[t-1,]*Zlist[[g]]
      XX=cbind(rep(1,mg[g]),Xg[[g]],Wg[[g]]%*%as.matrix(Xg[[g]]))
      XVX=XVX+t(XX)%*%solve(V)%*%XX
      XVY=XVY+t(XX)%*%solve(V)%*%YY
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

    pp_sig=1
    for(g in 1:G){
      V1=(sige_1-sigez_1^2)*diag(1,mg[g])+siga_mat[t-1,]*rep(1,mg[g])%*%t(rep(1,mg[g]))
      V2=(sige_mat[t-1,]-sigez_mat[t-1,]^2)*diag(1,mg[g])+siga_mat[t-1,]*rep(1,mg[g])%*%t(rep(1,mg[g]))
      SS=diag(1,mg[g])-lambda_mat[t,]*Wg[[g]]
      XX=cbind(rep(1,mg[g]),Xg[[g]],Wg[[g]]%*%Xg[[g]])
      ep=SS%*%yg[[g]]-XX%*%beta_mat[t-1,]
      like_1=det(V1)^(-0.5)*exp(-0.5*t(ep-sigez_1*Zlist[[g]])%*%solve(V1)%*%(ep-sigez_1*Zlist[[g]]))
      like_2=det(V2)^(-0.5)*exp(-0.5*t(ep-sigez_mat[t-1,]*Zlist[[g]])%*%solve(V2)%*%(ep-sigez_mat[t-1,]*Zlist[[g]]))
      pp_sig=pp_sig*(like_1/like_2)
    }
    
    pp_sig=pp_sig*(dmvnorm(sig,sig0,diag(1,2))/dmvnorm(c(sige_mat[t-1,],sigez_mat[t-1,]),sig0,diag(1,2)))
    pp_sig=min(pp_sig,1)
    
    if(runif(1,0,1)<=pp_sig){
      sige_mat[t,]=sige_1
      sigez_mat[t,]=sigez_1
      acc_3=acc_3+1
    } else{
      sige_mat[t,]=sige_mat[t-1,]
      sigez_mat[t,]=sigez_mat[t-1,]
    }
    acc_rate3[t,]=acc_3/t
    
    ### sampling of alpha_g from posterior distribution
    
    for(g in 1:G){
      dd=(siga_mat[t-1,]^(-1)+(sige_mat[t,]-sigez_mat[t,]^2)^(-1)*t(rep(1,mg[g]))%*%rep(1,mg[g]))^(-1)
      SS=diag(1,mg[g])-lambda_mat[t,]*Wg[[g]]
      YY=SS%*%yg[[g]]-sigez_mat[t,]*Zlist[[g]]
      XX=cbind(rep(1,mg[g]),Xg[[g]],Wg[[g]]%*%Xg[[g]])
      XB=XX%*%beta_mat[t,]
      alpha_mat[t,g]=rnorm(1,(sige_mat[t,]-sigez_mat[t,]^2)^(-1)*dd*(rep(1,mg[g])%*%(YY-XB)),sqrt(dd))
    }
    
    ### sampling of sigma_alpha from posterior distribution
    inv_alpha=(rho0+G)/2
    inv_beta=(eta0+sum((alpha_mat[t,])^2))/2
    siga_mat[t,]=1/rgamma(1,shape=inv_alpha,rate=inv_beta)
    
    end_time=Sys.time()
    ###print result
    if(t%%10==0){
      cat("Intercept =",round(beta_mat[t,1],3),"\n")
      cat("Own Xg =", round(beta_mat[t,2],3),"\n")
      cat("Own Goal =", round(beta_mat[t,3],3),"\n")
      cat("Own Shot =", round(beta_mat[t,4],3),"\n")
      cat("Own Shot Assitst =", round(beta_mat[t,5],3),"\n")
      cat("Own Cross =", round(beta_mat[t,6],3),"\n")
      cat("Own Clearance =", round(beta_mat[t,7],3),"\n")
      cat("Own Dribble =", round(beta_mat[t,8],3),"\n")
      cat("Own Duel =", round(beta_mat[t,9],3),"\n")
      cat("Own Pressure =", round(beta_mat[t,10],3),"\n")
      cat("Peer Xg =", round(beta_mat[t,11],3),"\n")
      cat("Peer Goal =", round(beta_mat[t,12],3),"\n")
      cat("Peer Shot =", round(beta_mat[t,13],3),"\n")
      cat("Peer Shot Assist =", round(beta_mat[t,14],3),"\n")
      cat("Peer Cross =", round(beta_mat[t,15],3),"\n")
      cat("Peer Clearance =", round(beta_mat[t,16],3),"\n")
      cat("Peer Dribble =", round(beta_mat[t,17],3),"\n")
      cat("Peer Duel =", round(beta_mat[t,18],3),"\n")
      cat("Peer Pressure =", round(beta_mat[t,19],3),"\n")
      cat("sigma_e =",round(sige_mat[t,],3),"\n")
      cat("sigma_ez =",round(sigez_mat[t,],3),"\n")
      cat("gamma acceptance rate=",round(acc_rate1[t,],2),"\n")
      cat("lambda acceptance rate=",round(acc_rate2[t,],2),"\n")
      cat("sigma acceptance rate=",round(acc_rate3[t,],2),"\n")
      cat("lambda =", round(lambda_mat[t,],3),"\n")
      cat("current iteration",t,"is end","\n")
      cat("At",r,"repetition",t,"iteration spend",end_time-start_time,"seconds","\n");cat("\n")
    }
    
  }
  beta1_list[[r]]=beta_mat[500:iter,1][seq(1,length(beta_mat[500:iter,1]),10)]
  beta2_list[[r]]=beta_mat[500:iter,2][seq(1,length(beta_mat[500:iter,2]),10)]
  beta3_list[[r]]=beta_mat[500:iter,3][seq(1,length(beta_mat[500:iter,3]),10)]
  beta4_list[[r]]=beta_mat[500:iter,4][seq(1,length(beta_mat[500:iter,4]),10)]
  beta5_list[[r]]=beta_mat[500:iter,5][seq(1,length(beta_mat[500:iter,5]),10)]
  beta6_list[[r]]=beta_mat[500:iter,6][seq(1,length(beta_mat[500:iter,6]),10)]
  beta7_list[[r]]=beta_mat[500:iter,7][seq(1,length(beta_mat[500:iter,7]),10)]
  beta8_list[[r]]=beta_mat[500:iter,8][seq(1,length(beta_mat[500:iter,8]),10)]
  beta9_list[[r]]=beta_mat[500:iter,9][seq(1,length(beta_mat[500:iter,9]),10)]
  beta10_list[[r]]=beta_mat[500:iter,10][seq(1,length(beta_mat[500:iter,10]),10)]
  beta12_list[[r]]=beta_mat[500:iter,11][seq(1,length(beta_mat[500:iter,11]),10)]
  beta12_list[[r]]=beta_mat[500:iter,12][seq(1,length(beta_mat[500:iter,12]),10)]
  beta13_list[[r]]=beta_mat[500:iter,13][seq(1,length(beta_mat[500:iter,13]),10)]
  beta14_list[[r]]=beta_mat[500:iter,14][seq(1,length(beta_mat[500:iter,14]),10)]
  beta15_list[[r]]=beta_mat[500:iter,15][seq(1,length(beta_mat[500:iter,15]),10)]
  beta16_list[[r]]=beta_mat[500:iter,16][seq(1,length(beta_mat[500:iter,16]),10)]
  beta17_list[[r]]=beta_mat[500:iter,17][seq(1,length(beta_mat[500:iter,17]),10)]
  beta18_list[[r]]=beta_mat[500:iter,18][seq(1,length(beta_mat[500:iter,18]),10)]
  beta19_list[[r]]=beta_mat[500:iter,19][seq(1,length(beta_mat[500:iter,19]),10)]
  
  gamma1_list[[r]]=gamma_mat[500:iter,1][seq(1,length(gamma_mat[500:iter,1]),10)]
  gamma2_list[[r]]=gamma_mat[500:iter,2][seq(1,length(gamma_mat[500:iter,2]),10)]
  gamma3_list[[r]]=gamma_mat[500:iter,3][seq(1,length(gamma_mat[500:iter,3]),10)]
  gamma4_list[[r]]=gamma_mat[500:iter,4][seq(1,length(gamma_mat[500:iter,4]),10)]
  gamma5_list[[r]]=gamma_mat[500:iter,5][seq(1,length(gamma_mat[500:iter,5]),10)]
  gamma6_list[[r]]=gamma_mat[500:iter,6][seq(1,length(gamma_mat[500:iter,6]),10)]
  
  lambda_list[[r]]=lambda_mat[500:iter,][seq(1,length(lambda_mat[500:iter,]),10)]
  sigez_list[[r]]=sigez_mat[500:iter,][seq(1,length(sigez_mat[500:iter,]),10)]
  sige_list[[r]]=sige_mat[500:iter,][seq(1,length(sige_mat[500:iter,]),10)]
  siga_list[[r]]=siga_mat[500:iter,][seq(1,length(siga_mat[500:iter,]),10)]
  ed_time=Sys.time()
}
ed_time=Sys.time()
cat("total",iter,"iterations spend",ed_time-st_time,"!","\n")

beta1_vec=c();beta2_vec=c();beta3_vec=c()
beta4_vec=c();beta5_vec=c();beta6_vec=c()
beta7_vec=c();beta8_vec=c();beta9_vec=c()
beta10_vec=c();beta11_vec=c();beta12_vec=c()
beta13_vec=c();beta14_vec=c();beta15_vec=c()
beta16_vec=c();beta17_vec=c();beta18_vec=c();beta19_vec=c()

gamma1_vec=c();gamma2_vec=c();gamma3_vec=c()
gamma4_vec=c();gamma5_vec=c();gamma6_vec=c()
sigez_vec=c();sige_vec=c();lambda_vec=c();siga_vec=c()

for(r in 1:R){
  beta1=beta1_list[[r]][-1]
  beta1_vec=c(beta1_vec,beta1)
  beta2=beta2_list[[r]][-1]
  beta2_vec=c(beta2_vec,beta2)
  beta3=beta3_list[[r]][-1]
  beta3_vec=c(beta3_vec,beta3)
  beta4=beta4_list[[r]][-1]
  beta4_vec=c(beta4_vec,beta4)
  beta5=beta5_list[[r]][-1]
  beta5_vec=c(beta5_vec,beta5)
  beta6=beta6_list[[r]][-1]
  beta6_vec=c(beta6_vec,beta6)
  
  beta7=beta7_list[[r]][-1]
  beta7_vec=c(beta7_vec,beta7)
  beta8=beta8_list[[r]][-1]
  beta8_vec=c(beta8_vec,beta8)
  beta9=beta9_list[[r]][-1]
  beta9_vec=c(beta9_vec,beta9)
  beta10=beta10_list[[r]][-1]
  beta10_vec=c(beta10_vec,beta10)
  beta11=beta11_list[[r]][-1]
  beta11_vec=c(beta11_vec,beta11)
  beta12=beta12_list[[r]][-1]
  beta12_vec=c(beta12_vec,beta12)
  
  beta13=beta13_list[[r]][-1]
  beta13_vec=c(beta13_vec,beta13)
  beta14=beta14_list[[r]][-1]
  beta14_vec=c(beta14_vec,beta14)
  beta15=beta15_list[[r]][-1]
  beta15_vec=c(beta15_vec,beta15)
  beta16=beta16_list[[r]][-1]
  beta16_vec=c(beta16_vec,beta16)
  beta17=beta17_list[[r]][-1]
  beta17_vec=c(beta17_vec,beta17)
  beta18=beta18_list[[r]][-1]
  beta18_vec=c(beta18_vec,beta18)
  beta19=beta19_list[[r]][-1]
  beta19_vec=c(beta19_vec,beta19)
  
  lambda=lambda_list[[r]][-1]
  lambda_vec=c(lambda_vec,lambda)
  
  gamma1=gamma1_list[[r]][-1]
  gamma1_vec=c(gamma1_vec,gamma1)
  gamma2=gamma2_list[[r]][-1]
  gamma2_vec=c(gamma2_vec,gamma2)
  gamma3=gamma3_list[[r]][-1]
  gamma3_vec=c(gamma3_vec,gamma3)
  gamma4=gamma4_list[[r]][-1]
  gamma4_vec=c(gamma4_vec,gamma4)
  gamma5=gamma5_list[[r]][-1]
  gamma5_vec=c(gamma5_vec,gamma5)
  gamma6=gamma6_list[[r]][-1]
  gamma6_vec=c(gamma6_vec,gamma6)
  
  sigmaez=sigez_list[[r]][-1]
  sigez_vec=c(sigez_vec,sigmaez)
  sigmae=sige_list[[r]][-1]
  sige_vec=c(sige_vec,sigmae)
  sigmaa=siga_list[[r]][-1]
  siga_vec=c(siga_vec,sigmaa)
}

round(mean(beta1_vec),3);round(sd(beta1_vec),3)
round(mean(beta2_vec),3);round(sd(beta2_vec),3)
round(mean(beta3_vec),3);round(sd(beta3_vec),3)
round(mean(beta4_vec),3);round(sd(beta4_vec),3)
round(mean(beta5_vec),3);round(sd(beta5_vec),3)
round(mean(beta6_vec),3);round(sd(beta6_vec),3)
round(mean(beta7_vec),3);round(sd(beta7_vec),3)
round(mean(beta8_vec),3);round(sd(beta8_vec),3)
round(mean(beta9_vec),3);round(sd(beta9_vec),3)
round(mean(beta10_vec),3);round(sd(beta10_vec),3)
round(mean(beta11_vec),3);round(sd(beta11_vec),3)
round(mean(beta12_vec),3);round(sd(beta12_vec),3)
round(mean(beta13_vec),3);round(sd(beta13_vec),3)
round(mean(beta14_vec),3);round(sd(beta14_vec),3)
round(mean(beta15_vec),3);round(sd(beta15_vec),3)
round(mean(beta16_vec),3);round(sd(beta16_vec),3)
round(mean(beta17_vec),3);round(sd(beta17_vec),3)
round(mean(beta18_vec),3);round(sd(beta18_vec),3)
round(mean(beta19_vec),3);round(sd(beta19_vec),3)

round(mean(gamma1_vec),3);round(sd(gamma1_vec),3)
round(mean(gamma2_vec),3);round(sd(gamma2_vec),3)
round(mean(gamma3_vec),3);round(sd(gamma3_vec),3)
round(mean(gamma4_vec),3);round(sd(gamma4_vec),3)
round(mean(gamma5_vec),3);round(sd(gamma5_vec),3)
round(mean(gamma6_vec),3);round(sd(gamma6_vec),3)

round(mean(lambda_vec),3);round(sd(lambda_vec),3)
round(mean(sige_vec),3);round(sd(sige_vec),3)
round(mean(sigez_vec),3);round(sd(sigez_vec),3)
round(mean(siga_vec),3);round(sd(siga_vec),3)

par(mfrow=c(2,5))

plot(density(beta1_vec),type="l",xlab="",main=expression(paste(beta[1])));abline(v=0.5,col="red")
plot(density(beta2_vec),type="l",xlab="",main=expression(paste(beta[2])));abline(v=0.5,col="red")
plot(density(beta3_vec),type="l",xlab="",main=expression(paste(beta[3])));abline(v=0.5,col="red")
plot(density(gamma1_vec),type="l",xlab="",main=expression(paste(gamma[1])));abline(v=-1.5,col="red")
plot(density(gamma2_vec),type="l",xlab="",main=expression(paste(gamma[2])));abline(v=0.5,col="red")
plot(density(gamma3_vec),type="l",xlab="",main=expression(paste(gamma[3])));abline(v=1.0,col="red")
plot(density(lambda_vec),type="l",xlab="",main=expression(lambda));abline(v=0.05,col="red")
plot(density(sige_vec),type="l",xlab="",main=expression(paste(sigma[epsilon]^2)));abline(v=1.25,col="red")
plot(density(sigez_vec),type="l",xlab="",main=expression(paste(sigma[paste(epsilon,z)])));abline(v=0.5,col="red")
plot(density(siga_vec),type="l",xlab="",main=expression(paste(sigma[alpha]^2)));abline(v=0.5,col="red")

plot(beta1_vec,type="l",ylab="",main=expression(paste(beta[1])),xlab="");abline(h=0.5,col="red")
plot(beta2_vec,type="l",ylab="",main=expression(paste(beta[2])),xlab="");abline(h=0.5,col="red")
plot(beta3_vec,type="l",ylab="",main=expression(paste(beta[3])),xlab="");abline(h=0.5,col="red")
plot(gamma1_vec,type="l",ylab="",main=expression(paste(gamma[1])),xlab="");abline(h=-1.5,col="red")
plot(gamma2_vec,type="l",ylab="",main=expression(paste(gamma[2])),xlab="");abline(h=0.5,col="red")
plot(gamma3_vec,type="l",ylab="",main=expression(paste(gamma[3])),xlab="");abline(h=1.0,col="red")
plot(lambda_vec,type="l",ylab="",main=expression(lambda));abline(h=0.05,col="red")
plot(sige_vec,type="l",xlab="",main=expression(paste(sigma[epsilon]^2)));abline(h=1.25,col="red")
plot(sigez_vec,type="l",xlab="",main=expression(paste(sigma[paste(epsilon,z)])));abline(h=0.5,col="red")
plot(siga_vec,type="l",xlab="",main=expression(paste(sigma[alpha]^2)));abline(h=0.5,col="red")
