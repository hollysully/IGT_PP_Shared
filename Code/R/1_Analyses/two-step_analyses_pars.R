Sess1 <- read.csv("/Users/tuo09169/Dropbox/1_Comp_Modelling/1_IGT_PlayPass/IGT_PP_Shared/Data/2_Fitted/IGT_PP_sep_sess1_IndPars.csv")
Sess2 <- read.csv("/Users/tuo09169/Dropbox/1_Comp_Modelling/1_IGT_PlayPass/IGT_PP_Shared/Data/2_Fitted/IGT_PP_sep_sess2_IndPars.csv")

# merge the data by subjID
BothSess <- merge(Sess1,Sess2,by="subjID") 

names(BothSess)

cor(BothSess$Fit_sep_sess1_Arew, BothSess$Fit_sep_sess2_Arew) # 0.3647699

cor(BothSess$Fit_sep_sess1_Apun, BothSess$Fit_sep_sess2_Apun) # 0.4277723

cor(BothSess$Fit_sep_sess1_K, BothSess$Fit_sep_sess2_K) # 0.06150379

cor(BothSess$Fit_sep_sess1_betaF, BothSess$Fit_sep_sess2_betaF) # 0.3963194

cor(BothSess$Fit_sep_sess1_betaP, BothSess$Fit_sep_sess2_betaP)  # 0.397117


# just for reference the two-step correlations from the last paper were: 
# A+ r = .39; A- r = .36; K r = .52; βf r = .39; βp r = .65

summary(BothSess$Fit_sep_sess1_Arew)
summary(BothSess$Fit_sep_sess2_Arew)

summary(BothSess$Fit_sep_sess1_Apun)
summary(BothSess$Fit_sep_sess2_Apun)

summary(BothSess$Fit_sep_sess1_K)
summary(BothSess$Fit_sep_sess2_K)

summary(BothSess$Fit_sep_sess1_betaF)
summary(BothSess$Fit_sep_sess2_betaF)

summary(BothSess$Fit_sep_sess1_betaP)
summary(BothSess$Fit_sep_sess2_betaP)




hist(Sess1$Fit_sep_sess1_betaP, breaks = 49)

hist(BothSess$Fit_sep_sess2_betaP, breaks = 39)
