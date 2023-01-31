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



# to read the .rds files back into R later, you do:
fit_sep1 <- readRDS("/Users/tuo09169/Dropbox/1_Comp_Modelling/1_IGT_PlayPass/IGT_PP_Shared/Data/2_Fitted/orl_pp_sess1.rds")

fit_sep2 <- readRDS("/Users/tuo09169/Dropbox/1_Comp_Modelling/1_IGT_PlayPass/IGT_PP_Shared/Data/2_Fitted/orl_pp_sess2.rds")


# Extract parameters
pars <- extract(fit_sep1)


# extract predicted choice behavior
post_pred <- pars$y_pred


# import session 1 raw data
raw_dat1 <- readRDS("/Users/tuo09169/Dropbox/1_Comp_Modelling/1_IGT_PlayPass/IGT_PP_Shared/Data/1_Preprocessed/Sess1.rds")


# Set color scheme for bayesplot (it is a global setting)
color_scheme_set("viridisC") ## this one is good! 


subj_list <- raw_dat1["subjID"]

igt_plots <- foreach(i=seq_along(subj_list)) %do% {
  foreach(cond=1:4) %do% {

    # Subject numbers in each group
    subjs <- subj_list[[i]]

    # Group-level means in true responses
    y <- foreach(subj=subjs, .combine = "rbind") %do% {
      raw_dat1$ydata[subj,raw_dat1$stim[subj,]==cond]-1
    } %>%
      colMeans()

    # Group-level posterior predictions for each trial
    yrep <- foreach(subj=subjs, .combine = "acomb") %do% {
      pars$y_pred[,subj,raw_dat1$stim[subj,]==cond]-1
    } %>%
      apply(., c(1,2), mean)
    # Make plot
    tmp_plot <- ppc_intervals(
      y = y,
      yrep = yrep,
      x = 1:length(y),
      prob = 0.99, 
      prob_outer = .99
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



