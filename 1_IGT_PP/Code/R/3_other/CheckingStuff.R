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


# # Check all subjects have 120 trials
# # # Getting subject IDs and number of subjects from data
# subj_ids <- unique(raw_dat1$Subject)
# n_subj <- length(subj_ids)
# # Subject loop
# for (i in 1:n_subj) {
#   # Subset out current subject
#   tmp <- subset(raw_dat1, Subject==subj_ids[i])
#   # Number of trials per subject
#   # (this assumes that everyone has the same number of trials)
#   n_trials <- length(tmp$stim)
#   #print(n_trials)
#   # can use this line to check that each subject was given 120 trials 
#   check_trials <- max(tmp$trials)
#   print(check_trials)
# }


# This code add trial number to a csv file of raw data 
# add trial number to raw_dat1
#trial<-rep(c(1:120),times=49)
#raw_dat1 <- cbind(raw_dat1, trial)

# save the raw data so I can inspect the format
#write.csv(raw_dat1, '/Users/tuo09169/Dropbox/1_Comp_Modelling/1_IGT_PlayPass/IGT_PP_Shared/Figs_Tables/Model_Validation/raw_dat1.csv')


## Check to see if cards were presented in the same order for all subjects
## NOTE: the card names for game version 2 have been recoded to match game version 1

# # set variable 'card_base' to the order of card presentation for the first subject 
# card_base <- subset(raw_dat1$cardname, raw_dat1$Subject==2049)
# 
# compare_table <- as.data.frame(array(NA, c(49, 1))) 
# 
# for (i in 1:n_subj) {  
#   # Subset out current subject
#   tmp <- subset(raw_dat1, Subject==subj_ids[i])
#   card_cur <- tmp$cardname
#   print(subj_ids[i])
#   print(card_cur)
#   compare_table[i,1] <- identical(card_base, card_cur) # value will be TRUE if files match
#   ## make sure all values in compare_table are "TRUE"
# }
