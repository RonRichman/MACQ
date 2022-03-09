##################################################################
#########  Marginal Attribution by Conditioning on Quantiles MACQ
#########  Authors: Merz, Richman, Tsanakas, Wüthrich
#########  Version March, 2022
##################################################################

path2 <- ""

##### load necessary packages
library(keras)
library(locfit)
library(tensor)
library(plot3D)
RNGversion("3.5.0")

##### versions of loaded packages: sessionInfo()
## plot3D_1.1.1     tensor_1.5       locfit_1.5-9.1   keras_2.2.0   
## Rcpp_1.0.2       lattice_0.20-35  zeallot_0.1.0    grid_3.5.0      
## R6_2.2.2         jsonlite_1.5     magrittr_1.5     tfruns_1.4      
## misc3d_0.8-4     whisker_0.3-2    Matrix_1.2-14    reticulate_1.10 
## compiler_3.5.0   base64enc_0.1-3  tensorflow_2.6.0

##### define regression function 
q0 <- 7

Design = layer_input(shape = c(q0), dtype = 'float32', name = 'Design') 
Term1  = Design %>% layer_lambda(function(x) k_square(x[,1])/2+k_sin(x[,2]))
Term2  = Design %>% layer_lambda(function(x) x[,3] * k_sin(x[,4])/2)
Term3  = Design %>% layer_lambda(function(x) - x[,5] * x[,6]/2)
Response = list(Term1, Term2, Term3) %>% layer_add()              
model <- keras_model(inputs = c(Design), outputs = c(Response))

##### generating data
N0 <- 10000 
set.seed(100)
dat    <- data.frame(array(rnorm(N0*q0),dim=c(N0,q0)))
dat$mu <- as.vector(model %>% predict(as.matrix(dat)))
dat$Y  <- rnorm(n=N0, mean=dat$mu, sd=1)
dat$U  <- rank(dat$mu)/nrow(dat)
featuresX <- paste("X",c(1:7), sep="")
XX     <- as.matrix(dat[,featuresX])

##### set quantile grid
alpha <- c(1:99)/100
qant  <- quantile((dat$mu), probs = alpha)

##### define and calculate gradient
grad = Response %>% layer_lambda(function(x) k_gradients(model$outputs, model$inputs))
model.grad <- keras_model(inputs = c(Design), outputs = c(grad))
##### calculate gradient
theta1 <- data.frame(model.grad %>% predict(XX))
names(theta1) <- featuresX
##### directional derivatives
Xtheta1 <- theta1 * XX
names(Xtheta1) <- featuresX

##### prepare for reference point optimization 1st order terms
for (jj in 1:q0){
  spline0 <- predict(locfit(theta1[, jj] ~ dat$U , alpha=0.1, deg=2), newdata=alpha)
  if (jj==1){
    grad1.mean <- spline0
    }else{
    grad1.mean <- cbind(grad1.mean,spline0)
    }}
#
grad1.Xmean <- predict(locfit(rowSums(Xtheta1) ~ dat$U , alpha=0.1, deg=2), newdata=alpha) 

##### define and calculate Hessian
for (jj in 1:q0){
 grads_jj = grad %>% layer_lambda(function(x) x[,jj])
 model.grads_jj <- keras_model(inputs = c(Design), outputs = c(grads_jj))
 Hessian = grads_jj %>% layer_lambda(function(x) k_gradients(model.grads_jj$outputs, model.grads_jj$inputs))
 model.Hessian <- keras_model(inputs = c(Design), outputs = c(Hessian))
 theta2_jj <- as.matrix(model.Hessian %>% predict(XX))
 if (jj==1){
      theta2 <- theta2_jj
      }else{
      theta2 <- cbind(theta2,theta2_jj)
      }}
#      
theta2.tensor <- array(theta2, dim=c(nrow(theta2), q0, q0))

##### prepare reference point optimization 2nd order terms
for (jj in 1:q0^2){
  spline0 <- predict(locfit(theta2[, jj] ~ dat$U , alpha=0.1, deg=2), newdata=alpha)
  if (jj==1){
    grad2.mean <- spline0
    }else{
    grad2.mean <- cbind(grad2.mean,spline0)
    }}
#
Xtheta2 <- array(NA, dim=c(nrow(theta2), q0))
for (jj in 1:q0){Xtheta2[, jj] <- rowSums(theta2.tensor[,,jj] * XX)}
for (jj in 1:q0){
  spline0 <- predict(locfit(Xtheta2[, jj] ~ dat$U , alpha=0.1, deg=2), newdata=alpha)
  if (jj==1){
    grad2.Xmean <- spline0
    }else{
    grad2.Xmean <- cbind(grad2.Xmean,spline0)
    }}
##
grad2.XXmean <- predict(locfit(rowSums(Xtheta2 * XX) ~ dat$U , alpha=0.1, deg=2), newdata=alpha) 


### reference point initialization
T0 <- 500
G <- array(NA, dim=c(T0+1,1))
const.alpha <- as.vector(quantile((dat$mu),probs=alpha))-grad1.Xmean+grad2.XXmean/2
weight.alpha <- t(grad1.mean) - t(grad2.Xmean)
at <- t(rep(0.0001, q0))
grad2.tensor.mean <- array(grad2.mean, dim=c(nrow(grad2.mean),q0,q0))
aa <- tensor(at, grad2.tensor.mean,2,2)[1,,] %*% t(at)/2
theta.at <- as.vector(model %>% predict(at)) 
G[1,1] <- sum((const.alpha - theta.at + (at %*% weight.alpha)[1,] + aa[,1])^2)

### reference point optimization
eta <- .01
for (t0 in 1:T0){
  dd <- 2 * (const.alpha - theta.at + (at %*% weight.alpha)[1,] + aa[,1])
  grad.theta.at = t(as.vector(model.grad %>% predict(at)))
  grad.at1 <- - sum(dd) * grad.theta.at
  grad.11 <- weight.alpha + t(tensor(at, grad2.tensor.mean,2,2)[1,,])/2
  grad.at2 <- dd %*% t(grad.11)
  grad.at <- grad.at1 + grad.at2
  at <- at - eta * grad.at/sqrt(sum(grad.at^2))
  theta.at <- as.vector(model %>% predict(at)) 
  aa <- tensor(at, grad2.tensor.mean,2,2)[1,,] %*% t(at)/2
  G[t0+1,1] <- sum((const.alpha - theta.at + (at %*% weight.alpha)[1,] + aa[,1])^2)
   }
##
plot(x=c(0:T0), y=G[,1], cex.lab=1.5, type='l', ylab="objective function G(a)", xlab="algorithmic time t", main=list("gradient descent decay in G", cex=1.5))


### shift the features to the new reference point at
at <- data.frame(at)
names(at) <- featuresX
round(at,3)
(theta.at <- as.vector(model %>% predict(as.matrix(at))))
XX.at <- XX - at[col(XX)]

### new directional derivatives w.r.t. new reference point at
Xtheta1 <- theta1 * XX.at
names(Xtheta1) <- featuresX

### new Hessians w.r.t. new reference point at
for (jj in 1:q0){
 grads_jj = grad %>% layer_lambda(function(x) x[,jj])
 model.grads_jj = keras_model(inputs = c(Design), outputs = c(grads_jj))
 Hessian = grads_jj %>% layer_lambda(function(x) k_gradients(model.grads_jj$outputs, model.grads_jj$inputs))
 model.Hessian <- keras_model(inputs = c(Design), outputs = c(Hessian))
 theta2_jj = as.matrix(model.Hessian %>% predict(XX))
 theta2_jj <- theta2_jj * XX.at
 theta2_jj <- theta2_jj * XX.at[,jj]
 names(theta2_jj) <- paste(featuresX, jj, sep="")
  if (jj==1){
      theta2 <- theta2_jj
      }else{
      theta2 <- cbind(theta2,theta2_jj)
      }}

### plot 1st and 2nd order contributions      
first.order <- rowSums(Xtheta1)
second.order <- rowSums(theta2)
second.order.diag <- rowSums(theta2[,(c(1:q0)-1)*q0+c(1:q0)])
spline1 <- predict(locfit(first.order ~ dat$U , alpha=0.1, deg=2), newdata=alpha)
spline2 <- predict(locfit(second.order ~ dat$U , alpha=0.1, deg=2), newdata=alpha)
spline2.diag <- predict(locfit(second.order.diag ~ dat$U , alpha=0.1, deg=2), newdata=alpha)
#
plot.yes <- 1
plot.yes <- 0
filey1 <- paste(path2, "linear_square_approximation.pdf", sep="")
if (plot.yes==1){pdf(file=filey1)}
col0 <- c("orange", "cyan", "red")
plot(x=alpha, y=theta.at+spline1, ylim=range(0, qant, theta.at+spline1-spline2/2),type='l', lwd=2, col=col0[1],cex.lab=1.5,xlab="quantile level alpha", ylab="mean mu(x)", cex=1.5, main=list("1st and 2nd order contributions", cex=1.5))
polygon(x=c(alpha, rev(alpha)), y=c(theta.at+spline1-spline2/2, rev(theta.at+spline1-spline2.diag/2)), col=adjustcolor("cyan",alpha.f=0.1)) 
points(x=alpha, y=qant, pch=19)
lines(x=alpha, y=theta.at+spline1, lwd=2, col=col0[1])
lines(x=alpha, y=theta.at+spline1-spline2/2, lwd=3, col=col0[3])
lines(x=alpha, y=theta.at+spline1-spline2.diag/2, lwd=2, col=col0[2])
#abline(h=mu0)
abline(v=c(0:5)*.2, col="darkgray", lty=2)
legend(x="bottomright", pch=c(-1,-1,-1, 1 ), bg="white", col=c(col0,"black"), lty=c(1,1,1,-1), lwd=c(2,3,2,-1), legend=c("first order contributions C_1", "second order without interactions C_2", "full second order contributions C_22","empirical quantiles"))
if (plot.yes==1){dev.off()}

### plot attributions excluding interaction terms      
Xtheta2 <- Xtheta1 - theta2[,(c(1:q0)-1)*q0+c(1:q0)]/2
for (jj in 1:q0){
  spline2 <- predict(locfit(Xtheta2[, jj] ~ dat$U , alpha=0.1, deg=2), newdata=alpha) 
  if (jj==1){
    grad2.Xmean <- spline2
    }else{
    grad2.Xmean <- cbind(grad2.Xmean,spline2)
    }}
grad2.Xmean <- data.frame(grad2.Xmean)
names(grad2.Xmean) <- featuresX
#
plot.yes <- 1
plot.yes <- 0
filey1 <- paste(path2, "expected_XXgradients.pdf", sep="")
if (plot.yes==1){pdf(file=filey1)}
col0 <- rev(rainbow(n=length(featuresX), start=0.15, end=1))
plot(x=alpha, grad2.Xmean[,1], ylim=range(c(-2,2)), type='l', lwd=1, col=col0[1],cex.lab=1.5,xlab="quantile level alpha", ylab="sensitivities", cex=1.5, main=list("attributions S_j-T_jj/2", cex=1.5))
abline(h=0)
for (jj in rev(1:(q0))){
    lines(x=alpha, y=grad2.Xmean[,jj], col=col0[jj], lwd=2)
    }
abline(v=c(0:5)*.2, col="darkgray", lty=2)
legend(x="bottomright", pch=rep(-1, q0), bg="white", col=col0, lty=rep(1,q0), lwd=2, legend=featuresX)
if (plot.yes==1){dev.off()}

### plot interaction terms (off-diagonal terms)
for (jj in 1:q0^2){
  spline22 <- predict(locfit(theta2[, jj] ~ dat$U , alpha=0.1, deg=2), newdata=alpha)
  if (jj==1){
    grad22.Xmean <- spline22
    }else{
    grad22.Xmean <- cbind(grad22.Xmean,spline22)
    }}
grad22.Xoff <- data.frame(-grad22.Xmean/2)

# select the significant interactions, i.e., above threshold rho
rho <- 0
interactions <- array(0,q0^2)
for (jj in 1:q0){for (kk in 1:q0){
   if (jj<kk){grad22.Xoff[,(jj-1)*q0+kk] <- 2*grad22.Xoff[,(jj-1)*q0+kk]
              if (max(abs(grad22.Xoff[,(jj-1)*q0+kk]))>rho){interactions[(jj-1)*q0+kk] <- 1}
              }else{
             grad22.Xoff[,(jj-1)*q0+kk] <- 0}
             } }
names(grad22.Xoff) <- names(theta2)
#
HH <- c(paste(featuresX, "-", featuresX[1], sep=""))
for (jj in 2:q0){HH <- c(HH, c(paste(featuresX, "-", featuresX[jj], sep="")))}
#
plot.yes <- 1
plot.yes <- 0
filey1 <- paste(path2, "expected_offdiagonals.pdf", sep="")
if (plot.yes==1){pdf(file=filey1)}
col0 <- rev(rainbow(n=sum(interactions), start=0.6, end=1))
plot(x=alpha, grad22.Xoff[,1], ylim=range(grad22.Xoff), type='l', lwd=1, col="white",cex.lab=1.5,xlab="quantile level alpha", ylab="interaction contributions", cex=1.5, main=list("interaction terms -T_jk", cex=1.5))
namess <- ""
for (jj in 1:q0){for (kk in 1:q0){
     if (interactions[(jj-1)*q0+kk]==1){
      lines(x=alpha, y=grad22.Xoff[,(jj-1)*q0+kk], col=col0[sum(interactions[1:((jj-1)*q0+kk)])], lwd=2)
      namess <- c(namess, HH[(jj-1)*q0+kk])
    }}}
namess <- namess[-1]    
abline(h=c(-rho, 0, rho), lty=c(2,1,2))
abline(v=c(0:5)*.2, col="darkgray", lty=2)
legend(x="topright", pch=rep(-1, sum(interactions)), bg="white", col=col0, lty=rep(1,sum(interactions)), lwd=rep(2,sum(interactions)), legend=namess)
if (plot.yes==1){dev.off()}


### plots individual marginal contributions
set.seed(200)
ll <- sample(size=1000, x=c(1:nrow(dat)))
#
jj <- 2
label <- featuresX[jj]
col0 <- (rainbow(n=length(unique(dat[ll,label])), start=0, end=.5))
col1 <- col0[as.integer(as.factor(dat[ll,label]))]
#
ylim0 <- range(Xtheta2[ll,label])
if (jj==2){ylim0 <- c(-2,2)}
spline1 <- predict(locfit(Xtheta2[, label] ~ dat$U , alpha=0.1, deg=2), newdata=alpha)
spline2 <- predict(locfit(Xtheta2[, label]^2 ~ dat$U , alpha=0.1, deg=2), newdata=alpha)
spline2 <- sqrt(spline2-spline1^2)
#                                                           
plot.yes <- 1
plot.yes <- 0
filey <- paste(path2, "IndSensitivity_",label,".pdf", sep="")
if (plot.yes==1){pdf(file=filey)} 
par(fig=c(0,.9,0,1), new=FALSE)
plot(x=dat[ll,]$U, y=Xtheta2[ll, label], pch=19, cex=1,  col=col1, ylim=ylim0, xlab="quantile level alpha", ylab=paste("individual second order contributions", sep=""), main=list(paste("inidividual marginal contributions: ",label, sep=""), cex=1.5), cex.lab=1.5)
abline(h=0, col="black", lty=1, lwd=1)
lines(x=alpha, spline1, col="black", lwd=2)
lines(x=alpha, y=spline1+spline2, col="black", lwd=1, lty=2)
lines(x=alpha, y=spline1-spline2, col="black", lwd=1, lty=2)
abline(v=c(0:5)*.2, col="darkgray", lty=2)
colkey(col = col0, clim=range(dat[ll,label]), add=TRUE, line.clab=.5)
if (plot.yes==1){dev.off()} 





