
# Script used to generate graphs based on the roots length's results

### Importing packages #########################################################
library(openxlsx)
library(ggplot2)
library(data.table)
library(dplyr)

### Functions ##################################################################

create_evol_df <- function(df){
  # Creates a dataframe containing the % of evolution of roots from one day to
  # another
  # Input : df, dataframe containing the raw lentgh data
  # Output : evol_df, dataframe containing the evolutions
  
  #initialisation of the first column of evol_df
  evol_df <- data.frame(rep(1,10))
  
  #iterating over colnames of df
  for (x in colnames(df)){
    
    #calculating a new column (evolution)
    evol_col <- df %>% 
      mutate(x = ((df[x] - lag(df[x]))/df[x])*100) %>% 
      select(x)
    
    #appending to evol_df
    evol_df <- cbind(evol_df, evol_col)
  }
  
  #modifying row names
  rownames(evol_df) <- paste0(lag(rownames(evol_df)), '_to_', rownames(evol_df))
  
  #removing the first col (initialisation)
  evol_df <- evol_df[2:nrow(evol_df), 2:ncol(evol_df)]
  
  #modifying column names
  colnames(evol_df) <- colnames(df)
  
  return(evol_df)
}

create_graph <- function(df, ori, no_rhizo=F){
  # Creates a graph of the evolution of roots length for several chosen rhizoboxes
  # Input : df, dataframe containing the raw lentgh data
  #         ori, character, name of the method used
  #         no_rhizo, optional (default=F), vector of numerical values, corres-
  #           ponding to the chosen rhizoboxes
  # Output : plot, the generated graph
  
  #definition of the titles based on the 'ori' input parameter
  title = paste('Evolution de la longueur des racines au cours du temps')
  subtitle = paste('Méthode',ori)
  
  #if no_rhizo=F and if length(df)>4 (**), all rhizoboxes will be displayed, 
  #and so there's no need to plot a legend (no_legend=T)
  # (**) : for the K_means results, the only 2 rhizoboxes will be displayed
  no_legend=F
  if (!is.numeric(no_rhizo)){
    
    #however, even if no_rhizo=F, if
    if (length(df)>4){
      no_legend=T
    }
    no_rhizo <- 1:length(df)
  }
  
  #selecting the right rhizoboxes from the dataframe
  df <- df[,no_rhizo]
  rhizonames <- colnames(df)
  
  #converting the df from short to long format (more convenient for ggplot2)
  df <- as.data.table(df, keep.rownames='dates')
  colnames(df)[2:length(df)] <- rhizonames
  df <- melt(df, id.vars='dates')
  
  #creating the plot
  if (!no_legend){
    plot <- df %>% ggplot(aes(x=dates, y=value, group=variable,
                              col=variable)) +
      geom_line() +
      labs(x = 'Temps', y = 'Longueur estimée (pixels)', title=title,
           subtitle=subtitle, col='Rhizobox')
    
  } else {
    plot <- df %>% ggplot(aes(x=dates, y=value, group=variable)) +
      geom_line() +
      labs(x = 'Temps', y = 'Longueur estimée (pixels)', title=title,
           subtitle=subtitle, col='Rhizobox')
  }
  
  return(plot)
}

create_barplot <- function(L_df, no_rhizo){
  # Creates barplot graphs of the % of evolution of rhizoboxes from one day to another
  # Input : L_df, list of dataframes containing the evol data
  #         no_rhizo, the one rhizobox to display data from
  # Output : plot, the generated graph
  
  #extracting dataframes from the list L_df
  df1 <- L_df[['U_net_pix']] ;   df2 <- L_df[['autoencoder_pix']] ;   
  df3 <- L_df[['K_means_pix']] ; df4 <- L_df[['U_net_diff']] ;   
  df5 <- L_df[['autoencoder_diff']] ;   df6 <- L_df[['K_means_diff']]
  
  #creating a df for the pix method
  rownames <- rownames(df1)
  df_pix <- data.frame(U_net = df1[,no_rhizo],
                       autoencoder = df2[,no_rhizo],
                       K_means = df3[,no_rhizo])
  rownames(df_pix) <- rownames
  colnames(df_pix) <-  c('U_net','autoencoder','K_means')
  
  #creating a df for the diff method
  df_diff <- data.frame(U_net = df4[,no_rhizo],
                        autoencoder = df5[,no_rhizo],
                        K_means = df6[,no_rhizo])
  rownames(df_diff) <- rownames
  colnames(df_diff) <-  c('U_net','autoencoder','K_means')
  
  #converting from short to long format
  df_pix <- as.data.table(df_pix, keep.rownames='dates')
  df_pix <- melt(df_pix, id.vars='dates')
  
  #converting from short to long format
  df_diff <- as.data.table(df_diff, keep.rownames='dates')
  df_diff <- melt(df_diff, id.vars='dates')
  
  df_pix$dates_num <- as.numeric(as.factor(df_pix$dates))
  
  #creating the plots
  title = paste('Evolution de la longueur des racines d\'un jour à l\'autre')
  plot_pix <- df_pix %>% ggplot(aes(x=variable, y=value, group=dates,
                                    fill=variable)) +
    geom_bar(position="dodge", stat="identity") +
    labs(x = 'Temps', y = 'Evolution (%)', title=title, fill='Modèle',
         subtitle=paste('Méthode pix (',no_rhizo,')'))
  
  title = paste('Evolution de la longueur des racines d\'un jour à l\'autre')
  plot_diff <- df_diff %>% ggplot(aes(x=variable, y=value, group=dates,
                                      fill=variable)) +
    geom_bar(position="dodge", stat="identity") +
    labs(x = 'Temps', y = 'Evolution (%)', title=title, fill='Modèle',
         subtitle=paste('Méthode diff (',no_rhizo,')'))
  
  return(list(plot_pix, plot_diff))
}

save_plots <- function(L_plots){
  # Saves plots present in the input list
  # Input : L_plots, list of plots to save

  #iterating over the plots
  for (i in 1:length(L_plots)){
    
    plot <- L_plots[[i]]
    name <- names(L_plots)[i]
    
    #saving with ggsave (parameters may need to be adjusted)
    ggsave(paste0('08.Graphs/',name,'.jpg'), plot, device='jpeg',
           width=15, height=10, units='cm')
  }
}


### Script #####################################################################

# U_net_pix
#importing data
df_U_net_pix <- read.xlsx('Roots_length_U_net_pix.xlsx', sheet = 1, colNames  = T,
                          rowNames = T)
#modifying the rownames to filter out the number
rownames(df_U_net_pix) <- substr(rownames(df_U_net_pix), 6, 10)
#creating the evol df
evol_U_net_pix <- create_evol_df(df_U_net_pix)


# U_net_diff
df_U_net_diff <- read.xlsx('Roots_length_U_net_diff.xlsx', sheet = 1,
                           colNames  = T, rowNames = T)
rownames(df_U_net_diff) <- substr(rownames(df_U_net_diff), 6, 10)
evol_U_net_diff <- create_evol_df(df_U_net_diff)


# autoencoder_pix
df_autoencoder_pix <- read.xlsx('Roots_length_autoencoder_pix.xlsx', sheet = 1, 
                                colNames  = T, rowNames = T)
rownames(df_autoencoder_pix) <- substr(rownames(df_autoencoder_pix), 6, 10)
evol_autoencoder_pix <- create_evol_df(df_autoencoder_pix)


# autoencoder_diff
df_autoencoder_diff <- read.xlsx('Roots_length_autoencoder_diff.xlsx', sheet = 1, 
                                 colNames  = T, rowNames = T)
rownames(df_autoencoder_diff) <- substr(rownames(df_autoencoder_diff), 6, 10)
evol_autoencoder_diff <- create_evol_df(df_autoencoder_diff)


# kmeans_diff
df_K_means_diff <- read.xlsx('Roots_length_K_means_diff.xlsx', sheet = 1, 
                                 colNames  = T, rowNames = T)
rownames(df_K_means_diff) <- substr(rownames(df_K_means_diff), 6, 10)
evol_K_means_diff <- create_evol_df(df_K_means_diff)


# kmeans_pix
df_K_means_pix <- read.xlsx('Roots_length_K_means_pix.xlsx', sheet = 1, 
                                 colNames  = T, rowNames = T)
rownames(df_K_means_pix) <- substr(rownames(df_K_means_pix), 6, 10)
evol_K_means_pix <- create_evol_df(df_K_means_pix)

### creating graphs
range <- c(4,7,9,13,16) #selection of rhizoboxes

plot_U_net_pix <- create_graph(df_U_net_pix, 'U_net_pix', range)
plot_U_net_pix

plot_U_net_pix2 <- create_graph(df_U_net_pix, 'U_net_pix')
plot_U_net_pix2

plot_U_net_diff <- create_graph(df_U_net_diff, 'U_net_diff', range)
plot_U_net_diff

plot_autoencoder_pix <- create_graph(df_autoencoder_pix, 'autoencoder_pix', range)
plot_autoencoder_pix

plot_autoencoder_diff <- create_graph(df_autoencoder_diff, 'autoencoder_diff', range)
plot_autoencoder_diff

plot_K_means_pix <- create_graph(df_K_means_pix, 'K_means_pix')
plot_K_means_pix

plot_K_means_diff <- create_graph(df_K_means_diff, 'K_means_diff')
plot_K_means_diff


### creating barplots
res <- create_barplot(list('U_net_pix'=evol_U_net_pix,
                            'autoencoder_pix'=evol_autoencoder_pix,
                            'K_means_pix'=evol_K_means_pix,
                            'U_net_diff'=evol_U_net_diff,
                            'autoencoder_diff'=evol_autoencoder_diff,
                            'K_means_diff'=evol_K_means_diff),
                       no_rhizo='rhizo_N17')

plot_evol_pix <- res[[1]]
plot_evol_diff <- res[[2]]

plot_evol_pix
plot_evol_diff


### graph for the F1 scores
F1 <- read.xlsx('07.Compare_F1_set/F1_scores.xlsx', sheet = 1, colNames  = T,
                          rowNames = T)
F1 <- F1[1:10,] #selecting only the data

#converting from short to long format
F1 <- as.data.table(F1, keep.rownames='id')
F1 <- melt(F1, id.vars='id')

#reorder factor levels (for the ggplot legend)
F1$variable <- factor(F1$variable, 
                       levels = c('2D-Unet','Autoencoder','K-means','3D-Unet'))

#creating plot
title = paste('Comparaison des performances de prédiction des modèles')
plot_F1 <- F1 %>% ggplot(aes(x=variable,y=value,fill=variable)) +
  geom_boxplot(na.rm=T) +
  labs(x = 'Modèle', y = 'Score F1', title=title, fill='Modèle')
plot_F1


### Saving the graphs
### A folder named 'output' must be present in the script's directory
L_plots <- list('plot_autoencoder_pix'=plot_autoencoder_pix,
                'plot_autoencoder_diff'=plot_autoencoder_diff,
                'plot_U_net_pix'=plot_U_net_pix,
                'plot_U_net_pix2'=plot_U_net_pix2,
                'plot_U_net_diff'=plot_U_net_diff,
                'plot_K_means_pix'=plot_K_means_pix,
                'plot_K_means_diff'=plot_K_means_diff,
                'plot_evol_pix'=plot_evol_pix,
                'plot_evol_diff'=plot_evol_diff,
                'plot_F1'=plot_F1)

save_plots(L_plots)


