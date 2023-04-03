# posteriorRho, postRav, and postRav.Density functions are taken from DMC, which
# can be found here: osf.io/pbwx8/
posteriorRho <- function(r, n, npoints=100, kappa=1)
  # Code provided by Dora Matzke, March 2016, from Alexander Ly
  # Reformatted into a single funciton. kappa=1 implies uniform prior.
  # Picks smart grid of npoints points concentrating around the density peak. 
  # Returns approxfun for the unnormalized density. 
{
  
  .bf10Exact <- function(n, r, kappa=1) {
    # Ly et al 2015
    # This is the exact result with symmetric beta prior on rho
    # with parameter alpha. If kappa = 1 then uniform prior on rho
    #
    if (n <= 2){
      return(1)
    } else if (any(is.na(r))){
      return(NaN)
    }
    # TODO: use which
    check.r <- abs(r) >= 1 # check whether |r| >= 1
    if (kappa >= 1 && n > 2 && check.r) {
      return(Inf)
    }
    
    log.hyper.term <- log(hypergeo::genhypergeo(U=c((n-1)/2, (n-1)/2), 
                                                L=((n+2/kappa)/2), z=r^2))
    log.result <- log(2^(1-2/kappa))+0.5*log(pi)-lbeta(1/kappa, 1/kappa)+
      lgamma((n+2/kappa-1)/2)-lgamma((n+2/kappa)/2)+log.hyper.term
    real.result <- exp(Re(log.result))
    return(real.result)
  }
  
  .jeffreysApproxH <- function(n, r, rho) {	
    result <- ((1 - rho^(2))^(0.5*(n - 1)))/((1 - rho*r)^(n - 1 - 0.5))
    return(result)
  }
  
  .bf10JeffreysIntegrate <- function(n, r, kappa=1) {
    # Jeffreys' test for whether a correlation is zero or not
    # Jeffreys (1961), pp. 289-292
    # This is the exact result, see EJ
    ##
    if (n <= 2){
      return(1)
    } else if ( any(is.na(r)) ){
      return(NaN)
    }
    
    # TODO: use which
    if (n > 2 && abs(r)==1) {
      return(Inf)
    }
    hyper.term <- Re(hypergeo::genhypergeo(U=c((2*n-3)/4, (2*n-1)/4), L=(n+2/kappa)/2, z=r^2))
    log.term <- lgamma((n+2/kappa-1)/2)-lgamma((n+2/kappa)/2)-lbeta(1/kappa, 1/kappa)
    result <- sqrt(pi)*2^(1-2/kappa)*exp(log.term)*hyper.term
    return(result)
  }
  
  
  # 1.0. Built-up for likelihood functions
  .aFunction <- function(n, r, rho) {
    hyper.term <- Re(hypergeo::genhypergeo(U=c((n-1)/2, (n-1)/2), L=(1/2), z=(r*rho)^2))
    result <- (1-rho^2)^((n-1)/2)*hyper.term
    return(result)
  }
  
  .bFunction <- function(n, r, rho) {
    hyper.term <- Re(hypergeo::genhypergeo(U=c(n/2, n/2), L=(3/2), z=(r*rho)^2))
    log.term <- 2*(lgamma(n/2)-lgamma((n-1)/2))+((n-1)/2)*log(1-rho^2)
    result <- 2*r*rho*exp(log.term)*hyper.term
    return(result)
  }
  
  .hFunction <- function(n, r, rho) {
    result <- .aFunction(n, r, rho) + .bFunction(n, r, rho)
    return(result)
  }
  
  .scaledBeta <- function(rho, alpha, beta){
    result <- 1/2*dbeta((rho+1)/2, alpha, beta)
    return(result)
  }
  
  .priorRho <- function(rho, kappa=1) {
    .scaledBeta(rho, 1/kappa, 1/kappa)
  }
  
  fisherZ <- function(r) log((1+r)/(1-r))/2
  
  inv.fisherZ <- function(z) {K <- exp(2*z); (K-1)/(K+1)}
  
  
  # Main body
  
  # Values spaced around mode
  qs <- qlogis(seq(0,1,length.out=npoints+2)[-c(1,npoints+2)])
  rho <- c(-1,inv.fisherZ(fisherZ(r)+qs/sqrt(n)),1)
  # Get heights
  if (!is.na(r) && !r==0) {
    d <- .bf10Exact(n, r, kappa)*.hFunction(n, r, rho)*.priorRho(rho, kappa)
  } else if (!is.na(r) && r==0) {
    d <- .bf10JeffreysIntegrate(n, r, kappa)*
      .jeffreysApproxH(n, r, rho)*.priorRho(rho, kappa)
  } else return(NA)
  # Unnormalized approximation funciton for density
  approxfun(rho,d)
}

# Estimate population correlation using analytical posterior as described in 
# Ly, Marsman, & Wagenmakers, 2017, Statistica Neerlandica
postRav <- function(r, n, spacing=.01, kappa=1,npoints=100,save=FALSE)
  # r is a vector, returns average density. Can also save unnormalized pdfs
{
  funs <- sapply(r,posteriorRho,n=n,npoints=npoints,kappa=kappa)
  rho <- seq(-1,1,spacing)
  result <- apply(matrix(unlist(lapply(funs,function(x){
    out <- x(rho); out/sum(out)
  })),nrow=length(rho)),1,mean)
  names(result) <- seq(-1,1,spacing)
  attr(result,"n") <- n
  attr(result,"kappa") <- kappa
  if (save) attr(result,"updfs") <- funs
  result
}
postRav.Density <- function(result)
  # Produces density class object
{
  x.vals <- as.numeric(names(result))
  result <- result/(diff(range(x.vals))/length(x.vals))
  out <- list(x=x.vals,y=result,has.na=FALSE,
              data.name="postRav",call=call("postRav"),
              bw=mean(diff(x.vals)),n=attr(result,"n")) 
  class(out) <- "density"
  out
}

# Compute the posterior density for population
pop_correct_MCMC <- function(post_samples, n_subj, kappa) {
  # Scale because the method fails when values are close to bounds -1 or 1
  dens.r <- postRav(n=n_subj, r=post_samples*.97,spacing=.01,kappa=kappa, npoints = 100)
  
  # Sample from the population-corrected PDF
  corrected_samples <- approx(
    cumsum(dens.r)/sum(dens.r),
    seq(-1,1, length.out=length(dens.r))*1.06, # rescale to corect scaling
    runif(10000)
    )
  # set anything that goes out of bounds to bound
  corrected_samples$y[corrected_samples$y < -1] <- -1
  corrected_samples$y[corrected_samples$y > 1] <- 1
  return(corrected_samples$y)
}