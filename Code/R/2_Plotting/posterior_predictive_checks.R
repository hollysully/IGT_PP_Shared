# Package for fitting .stan
library(rstan)
library(hBayesDM)
library(bayesplot)
library(dplyr)
library(ggplot2)
library(foreach)
library(tidybayes)
library(patchwork)
library(abind)
library(zoo)
library(here)


# to read the .rds files back into R later, you do:
fit_sep1 <- readRDS(here("Data", "2_Fitted", "orl_pp_sess1.rds"))

fit_sep2 <- readRDS(here("Data", "2_Fitted", "orl_pp_sess2.rds"))


# Extract parameters
pars <- extract(fit_sep1)


# extract predicted choice behavior
post_pred <- pars$y_pred


# import session 1 stan-ready data
stan_dat1 <- readRDS(here("Data", "1_Preprocessed", "Sess1.rds"))


# Set color scheme for bayesplot (it is a global setting)
color_scheme_set("viridisC") 


# For binding together posterior predictions
acomb <- function(...) abind(..., along=3)


subj_list <- stan_dat1["subjID"]


subj_list <- as.data.frame(stan_dat1["subjID"])

igt_plots <- foreach(stim=1:4) %do% {
  #subj <- subj_list[i]
  print(stim) #so far, cycling through the stim list
  y <- foreach(i=seq_along(subj_list), .combine = "rbind") %do% {
    # in line below, -1 recodes the play & pass values to 0 and 1
    # that, in the stan-ready data, were 1 for play or 2 for pass
    stan_dat1$ydata[i,stan_dat1$stim[i,]==stim]-1
  } %>%
    colMeans()
}






# NOTE: this `i=seq_along(subj_list)` loop was used in my code because I was
# showing plots of different groups of subjects. We do not necessarily need to 
# do this, so could remove the outer loop if we want to. 
igt_plots <- foreach(i=seq_along(subj_list)) %do% {
  foreach(stim=1:4) %do% {
    # Subject numbers in each group
    subjs <- subj_list[[i]]
    
    # Group-level means in true responses
    # NOTE: for my application, subjects were just numbered 1, 2, 3, ...,
    # Here, they have actual IDs, which I think is where the error is coming from.
    # e.g., if subj = 2049, the `[subj,stan_dat1$stim[subj,]==stim]` indexing
    # is then trying to grab an index that does not exist because 
    # `dim(stan_dat1$ydata)` is only `(49, 120)`. One potential fix could be to 
    # iterate over `seq_along(subjs)` as opposed to just `subjs`.
    y <- foreach(subj=subjs, .combine = "rbind") %do% {
      # ydata is NxT matrix where rows = subjects and columns = trials
      # below grabs the response on trials where subject was presented 
      # a given stimulus
      stan_dat1$ydata[subj,stan_dat1$stim[subj,]==stim]-1
    } %>%
      colMeans()
    
    # Group-level posterior predictions for each trial
    yrep <- foreach(subj=subjs, .combine = "acomb") %do% {
      # below follows same logic as above, but with posterior predictions
      # instead of actual data
      pars$y_pred[,subj,stan_dat1$stim[subj,]==stim]-1
    } %>%
      apply(., c(1,2), mean)
    # Make plot
    tmp_plot <- ppc_intervals(
      y = y,
      yrep = yrep,
      x = 1:length(y),
      prob = 0.50, 
      prob_outer = .80
    ) +
      ylim(0,1) +
      theme_minimal(base_size = 14) +
      theme(panel.grid = element_blank(),
            legend.position = "none", 
            axis.title.x = element_blank(),
            axis.title.y = element_blank(), 
            title = element_blank())
    return(tmp_plot)  
  }
}

# patchwork grid to show the figures
(igt_plots[[1]][[1]] | igt_plots[[1]][[2]]) / 
  (igt_plots[[1]][[3]] | igt_plots[[1]][[4]])
