


calculate_proportions = function(posteriors, stan_data, data_type = "list_of_matrices"){
  library(dplyr)
  library(foreach)
  library(abind)
  
  # Set iterations
  iterations = nrow(posteriors$mu_p)
  
  
  # Card matrix
  cards = stan_data$card
  
  
  # Recode choice data
  choices = ifelse(stan_data$choice == 2, 0, 1)
  predicted_values = ifelse(posteriors$choice_pred == 2, 0, 1)
  
  
  # Trial matrix (to skip when participant wasn't there)
  trials = stan_data$Tsubj
  
  
  # For binding together posterior predictions
  acomb = function(...) abind(..., along=3)
  bcomb = function(...) abind(..., along=4)
  
  
  # Calculates group-level choice proportions into a 30 (trial) x 4 (card) x 2 (session) matrix
  choice_proportions = foreach(s = 1:2, .combine = "acomb") %do% {
    foreach(c = 1:4, .combine = "cbind") %do% {
      temp = foreach(i = 1:49, .combine = "rbind") %do% {
        choices[i, cards[i, , s] == c, s]
      } %>% colMeans()
    }
  }
  
  
  predicted_proportions = foreach(s = 1:2, .combine = "bcomb") %do% {
    foreach(c = 1:4, .combine = "acomb") %do% {
      foreach(i = 1:49, .combine = "acomb") %do% {
        if(trials[i, s] > 0){
          predicted_values[ , i, cards[i, , s] == c, s]
        }
      } %>% 
        apply(., c(1,2), mean)
    }
  }
  
  if(data_type == "data.frame"){
    observed = data.frame(choice_proportions)
    names(observed) = c("A_1", "B_1", "C_1", "D_1",
                        "A_2", "B_2", "C_2", "D_2")
    observed = observed %>%
      mutate(trial = 1:30) %>% 
      pivot_longer(c(everything(), -trial), names_to = c("deck", "session"),
                   names_sep = "_", values_to = "observed",
                   names_transform = list(session = as.integer))
    
    predicted = foreach(s = 1:2, .combine = "rbind") %do% {
      foreach(c = 1:4, .combine = "rbind") %do% {
        foreach(t = 1:30, .combine = "rbind") %do% {
          data.frame(predicted = predicted_proportions[,t,c,s]) %>% 
            mutate(iteration = 1:iterations,
                   deck = c,
                   trial = t,
                   session = s,
                   mu = mean(predicted),
                   lower50 = HDIofMCMC(predicted, credMass = .5)[1],
                   lower50 = case_when(lower50 < 0 ~ 0, T ~ lower50),
                   upper50 = HDIofMCMC(predicted, credMass = .5)[2],
                   upper50 = case_when(upper50 > 1 ~ 1, T ~ upper50),
                   lower95 = HDIofMCMC(predicted)[1],
                   lower95 = case_when(lower95 < 0 ~ 0, T ~ lower95),
                   upper95 = HDIofMCMC(predicted)[2],
                   upper95 = case_when(upper95 > 1 ~ 1, T ~ upper95))
        }
      }
    }
    return(left_join(observed,
                     mutate(predicted, deck = case_when(deck == 1 ~ "A", deck == 2 ~ "B",
                                                        deck == 3 ~ "C", deck == 4 ~ "D"))))
  } else {
    return(list("observed" = choice_proportions, "predicted" = predicted_proportions))
  }
}















