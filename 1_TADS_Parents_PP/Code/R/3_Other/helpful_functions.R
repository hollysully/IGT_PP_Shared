


calculate_PPCs = function(cards,               # N participants X N trials matrix of card orders
                          trials,              # N participants vector of N trials
                          predicted_choices,   # N iterations X N participants X N trials of predicted choices (NOT transformed to 0s & 1s)
                          observed_choices     # N participants X N trials matrix of observed choices (NOT transformed to 0s and 1s)
                          ){
  library(dplyr)
  library(foreach)
  library(abind)
  
  
  # Recode choice data
  choices = ifelse(observed_choices == 2, 0, 1)
  predicted_values = ifelse(predicted_choices == 2, 0, 1)
  iterations = nrow(predicted_values)
  
  
  # For binding together posterior predictions
  acomb = function(...) abind(..., along=3)
  bcomb = function(...) abind(..., along=4)
  
  
  # Calculates group-level choice proportions into a 30 (trial) x 4 (card) matrix
  choice_proportions = foreach(c = 1:4, .combine = "cbind") %do% {
    temp = foreach(i = 1:nrow(choices), .combine = "rbind") %do% {
      choices[i, cards[i, ] == c]
    } %>% colMeans()
  }
  
  
  predicted_proportions = foreach(c = 1:4, .combine = "acomb") %do% {
    foreach(i = 1:nrow(choices), .combine = "acomb") %do% {
      if(trials[i] > 0){
        predicted_values[ , i, cards[i, ] == c]
      }
    } %>% 
      apply(., c(1,2), mean)
  }
  
  observed = data.frame(choice_proportions)
  names(observed) = c("A", "B", "C", "D")
  observed = observed %>%
    mutate(trial = 1:30) %>% 
    pivot_longer(c(everything(), -trial), names_to = "deck",
                 values_to = "observed")
  
  predicted = foreach(c = 1:4, .combine = "rbind") %do% {
    foreach(t = 1:30, .combine = "rbind") %do% {
      data.frame(predicted = predicted_proportions[,t,c]) %>% 
        mutate(iteration = 1:iterations,
               deck = c,
               trial = t,
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
  return(left_join(observed,
                   mutate(predicted, deck = case_when(deck == 1 ~ "A", deck == 2 ~ "B",
                                                      deck == 3 ~ "C", deck == 4 ~ "D"))))
}




