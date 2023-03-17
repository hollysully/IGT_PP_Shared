library(dplyr)
library(ggplot2)
library(foreach)
library(hBayesDM)
library(ggplot2)
library(bayesplot)
library(tidybayes)
library(patchwork)
library(abind)
library(zoo)
library(rstan)

setwd("~/Dropbox/Box/LAP/Blair/")

raw_dat <- readRDS("Data/1_Preprocessed/stan_ready_PA_item_fullInfo_Q12_v1.rds")
fit <- readRDS("Data/2_Fitted/fit_igt_orl_item_Q12_2theta_multiGRM_v3_iter1500_warm500.rds")
pars <- rstan::extract(fit)

### Predicted vs true sum scores ###
survey_names <- c("SDQ Hyper", "SDQ Conduct", "SDQ Prosocial", 
                  "ICU", "Conners", "RPQ", "ARI", "AUDIT", "CUDIT", 
                  "MFQ", "SDQ Emo Prob", "SCARED Panic/Somatic",
                  "SCARED GAD", "SCARED Separation", "SCARED Social", 
                  "SCARED School Avoid")
color_scheme <- c(rep("red", 9), rep("blue", 7))

survey_plots <- foreach(i=seq_along(survey_names)) %do% {
  # For breaking up subscales of SCARED
  if (survey_names[i] %in% c("SCARED Panic/Somatic",
                             "SCARED GAD", "SCARED Separation", 
                             "SCARED Social", 
                             "SCARED School Avoid")) {
    use_items <- switch(survey_names[i],
                        "SCARED Panic/Somatic" = c(1,6,9,12,15,18,19,
                                                   22,24,27,30,34,38),
                        "SCARED GAD"           = c(5,7,14,21,23,28,33,
                                                   35,37), 
                        "SCARED Separation"    = c(4,8,13,16,20,25,29,31), 
                        "SCARED Social"        = c(3,10,26,32,39,40,41), 
                        "SCARED School Avoid"  = c(2,11,17,36))
    # for extracting SCARED data 
    which_survey <- 12
  } else {
    which_survey <- i
    use_items <- 1:dim(raw_dat[[paste0("y", which_survey)]])[2]
  }
  
  # Get survey data and replace 0 with NAs
  tmp_dat <- raw_dat[[paste0("y", which_survey)]][,use_items,2]
  tmp_dat[tmp_dat==0] <- NA ######################################### is this right...
  no_NA <- !is.na(rowMeans(tmp_dat))
  
  # Compute summed scores from raw data, order for visualization
  order_scores <- order(rowMeans(tmp_dat[no_NA,]))
  sum_score <- rowSums(tmp_dat[no_NA,])[order_scores]
  
  # Posterior predictive summed scores, same ordering as raw
  tmp_pp <- pars[[paste0("y",which_survey,"_pred")]][,no_NA,use_items]
  pp_score <- apply(tmp_pp, c(1,2), sum)[,order_scores]
  
  color_scheme_set(color_scheme[i])
  tmp_plot <- ppc_intervals(
    y = sum_score,
    yrep = pp_score,
    x = 1:length(tmp_dat[no_NA,1]),
    prob = 0.95, 
    prob_outer = .95
  ) +
    labs(
      title = survey_names[i]
    ) +
    theme_minimal(base_size = 14) +
    theme(panel.grid = element_blank(),
          legend.position = "none", 
          plot.title = element_text(hjust = 0.5),
          axis.title.x = element_blank(),
          axis.title.y = element_blank())
}
pp_surveys <- (survey_plots[[1]] | survey_plots[[2]] | survey_plots[[3]]) / 
  (survey_plots[[4]] | survey_plots[[5]] | survey_plots[[6]]) /
  (survey_plots[[7]] | survey_plots[[8]] | survey_plots[[9]]) / 
  (survey_plots[[10]] | survey_plots[[11]] | survey_plots[[12]]) / 
  (survey_plots[[13]] | survey_plots[[14]] | survey_plots[[15]]) /
  ((ggplot() + theme_minimal()) | survey_plots[[16]] | (ggplot() + theme_minimal()))

ggsave(pp_surveys, filename = "Data/3_Plotted/postpred_surveys_Q12_iter1500_warm500.png",
       unit = "in", height = 12, width = 9)


### Average discriminability for each survey and dimension ###

n_samples <- (fit@stan_args[[1]]$iter - fit@stan_args[[1]]$warmup)*4 
alpha_dat <- foreach(i=seq_along(survey_names), .combine = "rbind") %do% {
  # For breaking up subscales of SCARED
  if (survey_names[i] %in% c("SCARED Panic/Somatic",
                             "SCARED GAD", "SCARED Separation", 
                             "SCARED Social", 
                             "SCARED School Avoid")) {
    use_items <- switch(survey_names[i],
                        "SCARED Panic/Somatic" = c(1,6,9,12,15,18,19,
                                                   22,24,27,30,34,38),
                        "SCARED GAD"           = c(5,7,14,21,23,28,33,
                                                   35,37), 
                        "SCARED Separation"    = c(4,8,13,16,20,25,29,31), 
                        "SCARED Social"        = c(3,10,26,32,39,40,41), 
                        "SCARED School Avoid"  = c(2,11,17,36))
    # for extracting SCARED data 
    which_survey <- 12
  } else {
    which_survey <- i
    use_items <- 1:dim(raw_dat[[paste0("y", which_survey)]])[2]
  }
  
  # Compute average alpha across items within latent dimensions
  mu_alpha_dim <- apply(pars[[paste0("alpha",which_survey)]][,,use_items], c(1,2), mean)
  
  # Retrun as data.frame
  data.frame(Survey = c(rep(survey_names[i], n_samples*2)),
             Dimension = c(rep("Impulsivity", n_samples),
                           rep("Anxiety", n_samples)),
             mu_alpha = c(mu_alpha_dim[,1], mu_alpha_dim[,2]))
}

color_choices <- cbind(color_scheme_get("red"), color_scheme_get("blue"))
alpha_plot <- alpha_dat %>%
  mutate(Survey = factor(Survey,
                         levels = rev(survey_names), 
                         labels = rev(survey_names)),
         Dimension = factor(Dimension,
                            levels = c("Impulsivity", "Anxiety"), 
                            labels = c("Impulsivity", "Anxiety"))) %>%
  ggplot(aes(x = mu_alpha, y = Survey, fill = Dimension, color = Dimension)) +
  geom_vline(xintercept = 0, linetype = 2) +
  stat_halfeye(point_interval = mean_hdi, .width = c(.95, .95)) +
  # ggtitle("Effect of Latent Trait on Measure") +
  xlab(bquote("Average Discriminability ("~alpha[mu]~")")) +
  ylab("Self-Report") +
  scale_fill_manual(values = unlist(color_choices[2,])) +
  scale_color_manual(values = unlist(color_choices[4,])) +
  theme_minimal(base_size = 15) +
  theme(panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank())

ggsave(alpha_plot, filename = "Data/3_Plotted/mu_alpha_surveys_Q12_iter1500_warm500.png",
       unit = "in", height = 5, width = 7)  
  

### Alpha parameters for single scale
alpha_sing_dat <- foreach(i=1:raw_dat$N_items[4], .combine = "rbind") %do% {
  # Alpha for each item within ICU
  mu_alpha_dim <- pars$alpha4[,,i]
  
  # Retrun as data.frame
  data.frame(Item      = c(rep(i, n_samples*2)),
             Dimension = c(rep("Impulsivity", n_samples),
                           rep("Anxiety", n_samples)),
             mu_alpha = c(mu_alpha_dim[,1], mu_alpha_dim[,2]))
}

color_choices <- cbind(color_scheme_get("red"), color_scheme_get("blue"))
alpha_sing_plot <- alpha_sing_dat %>%
  mutate(Item = factor(Item,
                       levels = rev(1:24),
                       labels = rev(1:24)),
         Dimension = factor(Dimension,
                            levels = c("Impulsivity", "Anxiety"), 
                            labels = c("Impulsivity", "Anxiety"))) %>%
  ggplot(aes(x = mu_alpha, y = Item, fill = Dimension, color = Dimension)) +
  geom_vline(xintercept = 0, linetype = 2) +
  stat_halfeye(point_interval = mean_hdi, .width = c(.95, .95)) +
  ggtitle("Effect of Latent Trait on Item") +
  xlab(bquote("Item Discriminability ("~alpha~")")) +
  ylab("ICU Item") +
  scale_fill_manual(values = unlist(color_choices[2,])) +
  scale_color_manual(values = unlist(color_choices[4,])) +
  theme_minimal(base_size = 15) +
  theme(panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank())

### Correlations between traits and cognitive model parameters ###

par_names <- c("Arew", "Apun", "K", "betaF", "betaP") 
R_dat <- foreach(i=seq_along(par_names), .combine = "rbind") %do% {
  # Retrun as data.frame
  data.frame(Parameter = c(rep(par_names[i], n_samples*2)),
             Dimension = c(rep("Impulsivity", n_samples),
                           rep("Anxiety", n_samples)),
             corr      = c(pars$R_pars[,i,6], pars$R_pars[,i,7]))
}
R_plot <- R_dat %>%
  mutate(Parameter = factor(Parameter,
                            levels = rev(par_names), 
                            labels = rev(par_names)),
         Dimension = factor(Dimension,
                            levels = c("Impulsivity", "Anxiety"), 
                            labels = c("Impulsivity", "Anxiety"))) %>%
  ggplot(aes(x = corr, y = Parameter, fill = Dimension, color = Dimension)) +
  geom_vline(xintercept = 0, linetype = 2) +
  stat_halfeye(point_interval = mean_hdi, .width = c(.95, .95)) +
  ggtitle("Latent Trait-Behavior Correlations") +
  xlab("Latent Correlation") +
  ylab("IGT Model Parameter") +
  scale_fill_manual(values = unlist(color_choices[2,])) +
  scale_color_manual(values = unlist(color_choices[4,])) +
  scale_y_discrete(labels = rev(c(expression(A["+"]), expression(A["-"]), 
                                  expression(K), expression(beta["F"]), 
                                  expression(beta["P"])))) +
  facet_wrap("Dimension") +
  theme_minimal(base_size = 15) +
  theme(panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        legend.position = "none")

ggsave(R_plot, filename = "Data/3_Plotted/latent_trait-behavior_corr_Q12_noExp2_iter1500_warm500.png",
       unit = "in", height = 4, width = 7)

#I'm here

### IGT entire group posterior predictions ###
mu_thetas <- apply(pars$theta, c(2,3), mean)
subj_list <- list(which(mu_thetas[1,] > -10))

color_scheme_set("red")
igt_plots <- foreach(i=seq_along(subj_list)) %do% {
  foreach(cond=1:4) %do% {
    # Subject numbers in each group
    subjs <- subj_list[[i]]
    
    # Group-level means in true responses
    y <- foreach(subj=subjs, .combine = "rbind") %do% {
      raw_dat$ydata[subj,raw_dat$stim[subj,]==cond]-1
    } %>%
      colMeans()
    # Group-level posterior predictions for each trial
    yrep <- foreach(subj=subjs, .combine = "acomb") %do% {
      pars$y_pred[,subj,raw_dat$stim[subj,]==cond]-1
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
all_plot <- (igt_plots[[1]][[1]] | igt_plots[[1]][[2]]) / 
                (igt_plots[[1]][[3]] | igt_plots[[1]][[4]])

ggsave(all_plot, filename = "Data/3_Plotted/all_igt_postpred_Q12_iter1500_warm500.png",
       unit = "in", height = 4, width = 6)


### IGT Posterior predictions for those above/below avg impulsivity ###

# Group-level posterior predictions
acomb <- function(...) abind(..., along=3)

# Find over/under avg impulsivity participants (avg = 0)
mu_thetas <- apply(pars$theta, c(2,3), mean)
subj_list <- list(which(mu_thetas[1,] > 1.3),
                  which(mu_thetas[1,] < -1.3))

color_scheme_set("red")
igt_plots <- foreach(i=seq_along(subj_list)) %do% {
  foreach(cond=1:4) %do% {
    # Subject numbers in each group
    subjs <- subj_list[[i]]
    
    # Group-level means in true responses
    y <- foreach(subj=subjs, .combine = "rbind") %do% {
      raw_dat$ydata[subj,raw_dat$stim[subj,]==cond]-1
    } %>%
      colMeans()
    # Group-level posterior predictions for each trial
    yrep <- foreach(subj=subjs, .combine = "acomb") %do% {
      pars$y_pred[,subj,raw_dat$stim[subj,]==cond]-1
    } %>%
      apply(., c(1,2), mean)
    # Make plot
    tmp_plot <- ppc_intervals(
      y = y,
      yrep = yrep,
      x = 1:length(y),
      prob = 0.8, 
      prob_outer = .95
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
hilo_plot <- (igt_plots[[1]][[1]] / igt_plots[[1]][[2]] / 
    igt_plots[[1]][[3]] / igt_plots[[1]][[4]]) |
  (igt_plots[[2]][[1]] / igt_plots[[2]][[2]] / 
     igt_plots[[2]][[3]] / igt_plots[[2]][[4]])

ggsave(hilo_plot, filename = "Data/3_Plotted/hilo_imp_igt_postpred_Q12_noExp2_iter1500_warm500.png",
       unit = "in", height = 7, width = 5)




# Individual participant-level posterior predictions
pars_list <- list(pars)
subj <- 7 #(1:numSubjs)[-which_train][6]
color_scheme_set("red")
plots <- foreach(pars=pars_list) %do% {
  foreach(cond=1:4) %do% {
    y    <- rollmean(raw_dat$ydata[subj,raw_dat$stim[subj,]==cond]-1, k=5)
    yrep <- t(apply(pars$y_pred[,subj,raw_dat$stim[subj,]==cond]-1, 1, rollmean, k=5))
    tmp_plot <- ppc_intervals(
      y = y,
      yrep = yrep,
      x = 1:length(y),
      prob = 0.80, 
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
ind_plot <- (plots[[1]][[1]] | plots[[1]][[2]]) / 
  (plots[[1]][[3]] | plots[[1]][[4]])

ggsave(ind_plot, filename = "Data/3_Plotted/ind_igt_postpred_Q12_iter1500_warm500.png",
       unit = "in", height = 4, width = 6)


### Group-level ORL parameters from current study and Haines et al. (2018) ###

# Parameter names
par_names <- c("Arew", "Apun", "K", "betaF", "betaP") 

# From prior study
igt_dat  <- readRDS("~/Dropbox/Box/GitHub/IGT/Data/Fitted/fits_all_orl.rds")
orl_HC   <- rstan::extract(igt_dat[[8]])[13:17]
orl_Amph <- rstan::extract(igt_dat[[9]])[13:17]
orl_Hero <- rstan::extract(igt_dat[[10]])[13:17]
orl_Cann <- rstan::extract(igt_dat[[11]])[13:17]

# From current study
orl_current <- pars[which(names(pars) %in% paste0("mu_", par_names))]

# Remove "mu_" from parameter name
names(orl_current) <- names(orl_HC) <- names(orl_Amph) <- 
  names(orl_Hero) <- names(orl_Cann) <- par_names

# Combine to single list
group_names <- c("Current Study", "Adult Controls", 
                 "Adult SUD (Amph)", "Adult SUD (Heroin)",
                 "Adult SUD (Cannabis)")
group_list <- list(orl_current, orl_HC, orl_Amph, orl_Hero, orl_Cann)
names(group_list) <- group_names

# make into long format, combining studies/groups
group_dat <- foreach(i=group_names, .combine = "rbind") %do% {
  foreach(p=par_names, .combine = "rbind") %do% {
    
    # Retrun as data.frame
    n_samps <- length(group_list[[i]][[p]])
    n_use_samps <- sample(1:n_samps, size = 4000, replace = F)
    if (p=="K") {
      data.frame(Group     = rep(i, 4000),
                 Parameter = rep(p, 4000),
                 Value     = log(group_list[[i]][[p]][n_use_samps]))
    } else {
      data.frame(Group     = rep(i, 4000),
                 Parameter = rep(p, 4000),
                 Value     = group_list[[i]][[p]][n_use_samps]) 
    }
  }
}

plot_labels <- function(variable, value) {
  labels <- list("Arew"  = expression(A["+"]), 
                 "Apun"  = expression(A["-"]), 
                 "K"     = expression(K), 
                 "betaF" = expression(beta["F"]), 
                 "betaP" = expression(beta["P"]))
  return(labels[value])
} 
group_plot <- group_dat %>%
  ggplot(aes(x = Value, fill = Group, color = Group, alpha = Group)) +
  geom_density(color = NA) +
  xlab("Parameter Estimate") +
  ylab("Density") +
  scale_alpha_manual(values = c(1, .1, .25, .4, .55)) +
  scale_fill_manual(values = c("#C79999", "black", "black", "black", "black")) +
  scale_color_manual(values = c("#C79999", "black", "black", "black", "black")) +
  facet_wrap("Parameter", scales = "free", labeller = plot_labels) +
  theme_minimal(base_size = 15) +
  theme(panel.grid = element_blank(),
        strip.text.x = element_text(size = 15),
        legend.position = "none")

ggsave(group_plot, filename = "Data/3_Plotted/group_posteriors_Q12_iter1500_warm500.png",
       unit = "in", height = 5, width = 8)
