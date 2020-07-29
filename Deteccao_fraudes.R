# Descrição dos dados
  

# Este dataset está disponível em: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data


# train.csv - o conjunto de treinamento

# test.csv - o conjunto de testes

# Cada linha dos dados de treinamento contém um registro de clique, com os seguintes recursos.
# ip: endereço IP do clique.
# app: ID do aplicativo para marketing.
# device: ID do tipo de dispositivo do celular do usuário (por exemplo, iphone 6 plus, iphone 7, huawei mate 7 etc.)
# os: ID da versão do sistema operacional do telefone móvel do usuário
# channel: ID do canal do editor de anúncios para celular
# click_time: registro de data e hora do clique (UTC)
# attributed_time: se o usuário baixar o aplicativo para depois de clicar em um anúncio, esse é o momento do download do aplicativo
# is_attributed: a meta a ser prevista, indicando que o aplicativo foi baixado
# Note-se que ip, app, device, os, e channelsão codificados.

# Os dados do teste são semelhantes, com as seguintes diferenças:
  
# click_id: referência para fazer previsões
# is_attributed: não incluso


### Carregando as bibliotecas

library(data.table)
library(xgboost)
library(dplyr)
library(highcharter)



### Carregando os arquivos train.csv, test.csv e verificando a distribuição dos dados na variável target

# O conjunto de dados está muito desequilibrado, são mais de 184 milhões de observações e apenas 0,25% das observações da 
# variável target é igual a 1.

train <- fread("train.csv", select =c("ip", "app", "device", "os", "channel","click_time", "is_attributed"), showProgress=F)
test <- fread("test.csv", drop = c("click_id"), showProgress=F)
invisible(gc())
round(prop.table(table(train$is_attributed)), digits= 4)*100 


### Coletando a amostra, 30 milhões de observações

set.seed(28)
train1 <- train[sample(.N, 30e6), ]


### Criando novas combinações de variáveis preditoras

# O Dataset possui apenas 6 variáveis preditoras, a combinação bidirecional e tridirecional entre as variáveis podeser útil
# para a criação do modelo preditivo.

y <- train1$is_attributed
nrowtrain <- 1:nrow(train1)
comb_train_test <- rbind(train1, test, fill = T)
rm(train,test)

invisible(gc())

comb_train_test[, `:=`(hour = hour(click_time)),
               ][, click_time := NULL
               ][, ip_c := .N, by = "ip"
               ][, app_c := .N, by = "app"
               ][, channel_c := .N, by = "channel"
               ][, device_c := .N, by = "device"
               ][, os_c := .N, by = "os"
               ][, ip_dev_c := .N, by = "ip,device"
               ][, ip_os_c := .N, by = "ip,os"
               ][, ip_chan_c := .N, by = "ip,channel"
               ][,ip_hour_c := .N, by = "ip,hour"
               ][,app_device_c := .N, by = "app,device"
               ][,app_channel_c := .N, by = "app,channel"
               ][,channel_hour_c := .N, by = "channel,hour"
               ][,ip_app_channel_c := .N, by = "ip,app,channel"
               ][,app_channel_hour_c := .N, by = "app,channel,hour"
               ][,ip_app_hour_c := .N, by = "ip,app,hour"
               ][, c("ip", "is_attributed") := NULL]

invisible(gc())


### Visualização com o highchart

col_comb_train_test <- colnames(comb_train_test)
h1 <- comb_train_test[, lapply(.SD, uniqueN), .SDcols = colnames(comb_train_test)] 
h2 <- (t(h1))
h3 <- data.table(variaveis= col_comb_train_test, v_unico = h2[ ,1])
h4 <- h3[, .(variaveis), by= v_unico][order(-v_unico)]

highchart() %>%
  hc_title(text = "Valores únicos") %>%
  hc_xAxis(categories = h4$variaveis) %>% hc_legend(enabled = FALSE) %>%
  hc_add_series(name = "v_unico", data = h4, type = "bar", hcaes(x = variaveis, y = v_unico)) %>%
  hc_add_theme(hc_theme_darkunica())


### Preparação dos dados

dtest <- xgb.DMatrix(data = data.matrix(comb_train_test[-nrowtrain]))
comb_train_test <- comb_train_test[nrowtrain]
nrowtrain <- caret::createDataPartition(y, p = 0.9, list = F)
dtrain <- xgb.DMatrix(data = data.matrix(comb_train_test[nrowtrain]), label = y[nrowtrain])
dval <- xgb.DMatrix(data = data.matrix(comb_train_test[-nrowtrain]), label = y[-nrowtrain])


rm(comb_train_test, y)
invisible(gc())


### Preparação do Modelo XGBoost

# Um dos parâmentros que auxiliam no balanceamento do conjunto de dados é o scale_pos_weight. O cálculo do valor  
# ideal para scale_pos_weight pode verificado em: https://xgboost.readthedocs.io/en/latest/parameter.html

p <- list(objective = "binary:logistic",
          booster = "gbtree",
          eval_metric = "auc",
          nthread = 8,
          eta = 0.07,
          max_depth = 4,
          min_child_weight = 96,
          gamma = 6,
          subsample = 1,
          colsample_bytree = 0.6,
          colsample_bylevel = 0.52,
          alpha = 0,
          lambda = 21,
          max_delta_step = 5,
          scale_pos_weight = 9.7,
          nrounds = 100)

m_xgb <- xgb.train(p, dtrain, p$nrounds, list(val = dval), print_every_n = 20, early_stopping_rounds = 30)
invisible(gc())


### Visualizações do modelo criado

imp <- xgb.importance(col_comb_train_test, model=m_xgb)

highchart() %>% 
  hc_title(text = "Ganho no modelo preditivo XGBoost") %>%
  hc_xAxis(categories = imp$Feature) %>% hc_legend(enabled = FALSE) %>% 
  hc_add_series(name = "Gain", data = imp, type = "bar", hcaes(x = Feature, y = Gain, color= -Gain)) %>%
  hc_add_theme(hc_theme_darkunica())

highchart() %>%
  hc_title(text = "Importância by Cover, Gain e Frequency") %>%
  hc_xAxis(categories = imp$Feature) %>%
  hc_add_series(name = "Cover", data = imp, type = "bar", hcaes(x = Feature, y = Cover)) %>%
  hc_add_series(name = "Gain", data = imp, type = "bar", hcaes(x = Feature, y= Gain)) %>%
  hc_add_series(name = "Frequency", data = imp, type = "bar", hcaes(x = Feature, y = Frequency)) %>%
  hc_add_theme(hc_theme_darkunica())    


### Criação do arquivo para submissão

predicted_xgb <- predict(m_xgb, dtest)
pred_xgb <- as.numeric(predicted_xgb > 0.54)
subm <- fread("sample_submission.csv") 
subm[, is_attributed := pred_xgb]
fwrite(subm, paste0("xgb_", m_xgb$best_score, ".csv"))

