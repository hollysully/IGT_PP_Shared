# Package for fitting .stan
library(rstan)
library(hBayesDM)
library(bayesplot)
library(here)

# the here package relates somehow to using "projects", which I set up when putting this on github
# open a new session of R, & in the top right hand corner, open the project file:
# /Users/tuo09169/Dropbox/1_Comp_Modelling/1_IGT_PlayPass/IGT_PP_Shared/IGT_PP_Shared.Rproj
# then, declare where the script is relative to the project root directory
here::i_am("1_SIGT_PP/Code/R/1_Analyses/fitting_ORL_SIGT_PP.R")

# the here() command tells you where the project root directory is
here()


########################################################################################## 
# first, compile model
########################################################################################## 
orl_pp <- stan_model(here("1_SIGT_PP", "Code", "Stan", "igt_orl.stan"))


########################################################################################## 
# fit data in a single task model, looping through each task
########################################################################################## 

# Read in stan-ready data for running the play or pass model on a single task 

for (lis in 1:2) {
  if (lis == 1) {
    stan_dat <- readRDS(here("1_SIGT_PP", "Data", "1_Preprocessed", "IGT_pre.rds"))
    dat_name <- "IGT_fitted"
  } else if (lis == 2) {
    stan_dat <- readRDS(here("1_SIGT_PP", "Data", "1_Preprocessed", "SIGT_v1_pre.rds"))
    dat_name <- "SIGT_v1_fitted"
  }
  
  
# for (lis in 3:4) {
#   if (lis == 3) {
#     stan_dat <- readRDS(here("1_SIGT_PP", "Data", "1_Preprocessed", "SIGT_v2_pre.rds"))
#     dat_name <- "SIGT_v2_fitted"
#   } else if (lis == 4) {
#     stan_dat <- readRDS(here("1_SIGT_PP", "Data", "1_Preprocessed", "SIGT_v3_pre.rds"))
#     dat_name <- "SIGT_v3_fitted"
#   }



################## If you have already compiled the model and saved as an RDS file, 
################## skip to line 43 and read in the RDS file

# Fit model
fit_sep <- sampling(orl_pp, 
                    data   = stan_dat, 
                    iter   = 5000, 
                    warmup = 1000,
                    chains = 6, 
                    cores  = 4,
                    seed   = 43210)


# save each stan ready data list as .rds
saveRDS(fit_sep, file = here("1_SIGT_PP", "Data", "2_Fitted", paste0(dat_name, ".rds")))
}










# to read the .rds file back into R later, you do:
fit_sep <- readRDS(filename)

# Extract parameters
pars <- extract(fit_sep)

# check 𝑅̂ (Rhat) is an index of the convergence of the chains. 
# 𝑅̂ values close to 1.00 would indicate that MCMC chains are converged to stationary target distributions.  
rhat(fit_sep)

# Plotting the "traceplots" 
# when using HbayesDM, these should look like furry caterpillars, where chains are similar
# but here, I think we are seeing the posterior distribution for the chains, so we want to see narrow distributions
plot(fit_sep, type = "trace")

  

# compute posterior means for different pars for each person, and then combine into 
# data.frame with another column indicating their respective IDs
IGT_PP_sep_sess1 <- data.frame(subjID  = stan_dat$subjID, 
                               Fit_sep_sess1_Arew = colMeans(pars$Arew[,]), # posterior means for session 1
                               Fit_sep_sess1_Apun = colMeans(pars$Apun[,]),
                               Fit_sep_sess1_K = colMeans(pars$K[,]),
                               Fit_sep_sess1_betaF = colMeans(pars$betaF[,]),
                               Fit_sep_sess1_betaP = colMeans(pars$betaP[,]))


# posterior means for parameter mu's
Fit_sep_sess1_mu_Arew <- mean(pars$mu_Arew) # 0.002426064
Fit_sep_sess1_mu_Apun <- mean(pars$mu_Apun) # 0.001275201
Fit_sep_sess1_mu_K <- mean(pars$mu_K)  # 0.06413485
Fit_sep_sess1_mu_betaF <- mean(pars$mu_betaF)  # 0.6253355
Fit_sep_sess1_mu_betaP <- mean(pars$mu_betaP)  # 0.6657244


#save the pars dataset
write.csv(IGT_PP_sep_sess1, here("Data", "2_Fitted", "IGT_PP_sep_sess1_IndPars.csv"))







########################################################################################## 
# fit Sess 2 data in a single timepoint model
########################################################################################## 

# Read in stan-ready data for single sess play pass ORL model
stan_dat <- readRDS(here("Data", "1_Preprocessed", "Sess2.rds"))

#explore the stan_dat
#stan_dat$Tsubj

################## If you have already compiled the model and saved as an RDS file, 
################## skip to line 43 and read in the RDS file

# Compile model
orl_pp_sep2 <- stan_model("Code", "Stan", "igt_orl.stan")

# Fit model
fit_sep <- sampling(orl_pp_sep2, 
                    data   = stan_dat, 
                    iter   = 5000, 
                    warmup = 1000,
                    chains = 6, 
                    cores  = 4,
                    seed   = 43210)




#save the fitted model as an .rds file
filename = here("Data", "2_Fitted", "orl_pp_sess2.rds")
saveRDS(fit_sep, filename)

# to read the .rds file back into R later, you do:
fit_sep <- readRDS(filename)

# Extract parameters
pars <- extract(fit_sep)

# check 𝑅̂ (Rhat) is an index of the convergence of the chains. 
# 𝑅̂ values close to 1.00 would indicate that MCMC chains are converged to stationary target distributions.  
rhat(fit_sep)

# Plotting the "traceplots" 
# when using HbayesDM, these should look like furry caterpillars, where chains are similar
# but here, I think we are seeing the posterior distribution for the chains, so we want to see narrow distributions
plot(fit_sep, type = "trace")



# compute posterior means for different pars for each person, and then combine into 
# data.frame with another column indicating their respective IDs
IGT_PP_sep_sess2 <- data.frame(subjID  = stan_dat$subjID, 
                               Fit_sep_sess2_Arew = colMeans(pars$Arew[,]), # posterior means for session 2
                               Fit_sep_sess2_Apun = colMeans(pars$Apun[,]),
                               Fit_sep_sess2_K = colMeans(pars$K[,]),
                               Fit_sep_sess2_betaF = colMeans(pars$betaF[,]),
                               Fit_sep_sess2_betaP = colMeans(pars$betaP[,]))


# posterior means for parameter mu's
Fit_sep_sess2_mu_Arew = mean(pars$mu_Arew) # 0.001953081
Fit_sep_sess2_mu_Apun = mean(pars$mu_Apun) # 0.001420647
Fit_sep_sess2_mu_K = mean(pars$mu_K)  # 0.04923301
Fit_sep_sess2_mu_betaF = mean(pars$mu_betaF)  # 0.5428928
Fit_sep_sess2_mu_betaP = mean(pars$mu_betaP)  # 1.110543



#save the pars dataset
write.csv(IGT_PP_sep_sess2,here("Data", "2_Fitted", "IGT_PP_sep_sess2_IndPars.csv"))


