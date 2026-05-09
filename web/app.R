# ============================================================
# 5291 Final Project Interactive Shiny App
# Financial Time Series Prediction Using Machine Learning
# ============================================================

library(shiny)
library(bslib)
library(DT)
library(ggplot2)
library(plotly)
library(dplyr)
library(readr)
library(tidyr)
library(stringr)
library(tools)

# ============================================================
# 1. Helper Functions
# ============================================================

find_project_root <- function() {
  current <- normalizePath(getwd(), winslash = "/", mustWork = FALSE)
  
  markers <- c(
    "clean_process_result",
    "eda_figures",
    "feature_engineer_result",
    "modeling_outputs",
    "numerical_output",
    "numercial_output",
    "unsupervised_analysis_result",
    "modeling_return_plot"
  )
  
  candidates <- c(
    current,
    dirname(current),
    dirname(dirname(current)),
    dirname(dirname(dirname(current)))
  )
  
  for (candidate in candidates) {
    if (any(dir.exists(file.path(candidate, markers)))) {
      return(normalizePath(candidate, winslash = "/", mustWork = FALSE))
    }
  }
  
  return(current)
}

APP_ROOT <- find_project_root()
message("App root detected as: ", APP_ROOT)

P <- function(...) {
  file.path(APP_ROOT, ...)
}

resource_src <- function(path_relative) {
  paste0("project_files/", gsub("\\\\", "/", path_relative))
}

if (dir.exists(APP_ROOT)) {
  addResourcePath("project_files", APP_ROOT)
}

pick_existing_dir <- function(possible_dirs) {
  for (d in possible_dirs) {
    if (dir.exists(P(d))) {
      return(d)
    }
  }
  return(possible_dirs[1])
}

NUM_DIR <- pick_existing_dir(c("numerical_output", "numercial_output"))

message("Numerical output folder detected as: ", NUM_DIR)

safe_read_csv <- function(path_relative) {
  path_full <- P(path_relative)
  
  if (file.exists(path_full)) {
    readr::read_csv(path_full, show_col_types = FALSE)
  } else {
    data.frame(Message = paste("File not found:", path_relative))
  }
}

safe_read_csv_by_pattern <- function(folder_relative, pattern) {
  folder_full <- P(folder_relative)
  
  if (!dir.exists(folder_full)) {
    return(data.frame(Message = paste("Folder not found:", folder_relative)))
  }
  
  files <- list.files(
    folder_full,
    pattern = pattern,
    full.names = TRUE,
    ignore.case = TRUE
  )
  
  if (length(files) == 0) {
    return(data.frame(Message = paste("File pattern not found:", file.path(folder_relative, pattern))))
  }
  
  readr::read_csv(files[1], show_col_types = FALSE)
}

safe_img <- function(path_relative, width = "100%") {
  path_full <- P(path_relative)
  
  if (file.exists(path_full)) {
    tags$img(
      src = resource_src(path_relative),
      width = width,
      style = "border-radius: 8px; border: 1px solid #dee2e6; background-color: white;"
    )
  } else {
    div(
      class = "alert alert-warning",
      paste("Image not found:", path_relative)
    )
  }
}

clean_name <- function(x) {
  x %>%
    str_replace_all(" ", "_") %>%
    str_replace_all("-", "_") %>%
    str_replace_all("__+", "_")
}

find_image_by_keywords <- function(folder_relative, keywords, extension = "png") {
  folder_full <- P(folder_relative)
  
  if (!dir.exists(folder_full)) {
    return(NULL)
  }
  
  files <- list.files(
    folder_full,
    pattern = paste0("\\.", extension, "$"),
    full.names = FALSE,
    recursive = TRUE,
    ignore.case = TRUE
  )
  
  if (length(files) == 0) {
    return(NULL)
  }
  
  files_lower <- tolower(files)
  keywords_lower <- tolower(keywords)
  
  matched <- files
  
  for (kw in keywords_lower) {
    keep <- str_detect(files_lower, fixed(kw))
    matched <- matched[keep]
    files_lower <- files_lower[keep]
    
    if (length(matched) == 0) {
      return(NULL)
    }
  }
  
  file.path(folder_relative, matched[1])
}

display_or_warning <- function(path_relative) {
  if (is.null(path_relative)) {
    div(class = "alert alert-warning", "Image not found. Please check the file name or folder path.")
  } else {
    safe_img(path_relative)
  }
}

# ============================================================
# 2. Load Data
# ============================================================

prices <- safe_read_csv("clean_process_result/cleaned_asset_prices.csv")
returns <- safe_read_csv("clean_process_result/asset_log_returns.csv")
basic_features <- safe_read_csv("clean_process_result/ml_features_bitcoin_target.csv")

features <- safe_read_csv("feature_engineer_result/extended_ml_features_bitcoin_target.csv")
feature_corr <- safe_read_csv("feature_engineer_result/extended_feature_target_correlation.csv")

supervised_results <- safe_read_csv_by_pattern(
  "modeling_outputs/tables",
  "all.*asset.*model.*comparison.*\\.csv$"
)

best_model_results <- safe_read_csv_by_pattern(
  "modeling_outputs/tables",
  "best.*model.*asset.*\\.csv$"
)

numerical_results <- safe_read_csv_by_pattern(
  NUM_DIR,
  "numerical.*method.*validation.*results.*\\.csv$"
)

best_numerical_results <- safe_read_csv_by_pattern(
  NUM_DIR,
  "best.*numerical.*method.*asset.*model.*\\.csv$"
)

cluster_summary_scaled <- safe_read_csv("unsupervised_analysis_result/cluster_summary_scaled.csv")
cluster_summary_unscaled <- safe_read_csv("unsupervised_analysis_result/cluster_summary_unscaled.csv")
outlier_results <- safe_read_csv("unsupervised_analysis_result/outlier_interpretation_results.csv")
unsup_results <- safe_read_csv("unsupervised_analysis_result/unsupervised_pca_cluster_outlier_results.csv")

assets <- c("Gold", "Silver", "WTI Oil", "USD Index", "Bitcoin", "Ethereum")

models <- c(
  "Ridge Regression",
  "Random Forest",
  "Hist Gradient Boosting"
)

# ============================================================
# 3. UI
# ============================================================

ui <- page_navbar(
  title = "5291 Final Project Dashboard",
  theme = bs_theme(
    version = 5,
    bootswatch = "flatly"
  ),
  
  header = tags$head(
    tags$style(HTML("
      body {
        background-color: #f8f9fa;
      }

      .section-title {
        font-size: 26px;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 10px;
      }

      .sub-title {
        font-size: 20px;
        font-weight: 600;
        margin-top: 18px;
        margin-bottom: 8px;
      }

      .card-box {
        background-color: white;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
      }

      .metric-card {
        background-color: white;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
      }

      .metric-number {
        font-size: 28px;
        font-weight: 700;
        color: #2c3e50;
      }

      .metric-label {
        font-size: 14px;
        color: #6c757d;
      }

      .plot-note {
        font-size: 15px;
        color: #555;
        margin-top: 8px;
        margin-bottom: 12px;
      }
    "))
  ),
  
  # ============================================================
  # Home
  # ============================================================
  
  nav_panel(
    "Home",
    div(
      class = "card-box",
      h2("Financial Time Series Prediction Using Machine Learning and Numerical Validation Methods"),
      p("This interactive dashboard summarizes the full workflow of the 5291 final project, including data cleaning, exploratory data analysis, feature engineering, unsupervised learning, supervised machine learning, and numerical method validation."),
      
      h4("Assets"),
      tags$ul(
        tags$li("Gold"),
        tags$li("Silver"),
        tags$li("WTI Oil"),
        tags$li("USD Index"),
        tags$li("Bitcoin"),
        tags$li("Ethereum")
      ),
      
      h4("Main Methods"),
      tags$ul(
        tags$li("EDA and financial time series visualization"),
        tags$li("Feature engineering using lagged returns, momentum, volatility, rolling correlation, drawdown, and extreme return indicators"),
        tags$li("Unsupervised learning using PCA, K-Means clustering, and Isolation Forest"),
        tags$li("Supervised learning using Ridge Regression, Random Forest, and Hist Gradient Boosting"),
        tags$li("Numerical validation using Euler, Trapezoidal, and RK4-like methods")
      )
    )
  ),
  
  # ============================================================
  # Clean and Pre-processing
  # ============================================================
  
  nav_panel(
    "Clean & Pre-processing",
    layout_sidebar(
      sidebar = sidebar(
        selectInput(
          "clean_data_choice",
          "Select dataset:",
          choices = c(
            "Cleaned Prices" = "prices",
            "Log Returns" = "returns",
            "Basic ML Features" = "basic_features",
            "Extended ML Features" = "features"
          )
        )
      ),
      
      div(
        class = "card-box",
        div(class = "section-title", "Cleaned and Preprocessed Data"),
        
        conditionalPanel(
          condition = "input.clean_data_choice == 'prices'",
          p("This table contains the cleaned aligned price data."),
          DTOutput("prices_table")
        ),
        
        conditionalPanel(
          condition = "input.clean_data_choice == 'returns'",
          p("This table contains daily log returns computed from the cleaned price data."),
          DTOutput("returns_table")
        ),
        
        conditionalPanel(
          condition = "input.clean_data_choice == 'basic_features'",
          p("This table contains the initial machine learning feature matrix created during the cleaning and preprocessing stage."),
          DTOutput("basic_features_table")
        ),
        
        conditionalPanel(
          condition = "input.clean_data_choice == 'features'",
          p("This table contains the extended machine learning feature matrix created from lagged returns, rolling statistics, momentum, volatility ratios, correlations, drawdowns, and extreme return indicators."),
          DTOutput("features_table")
        )
      )
    )
  ),
  
  # ============================================================
  # EDA
  # ============================================================
  
  nav_panel(
    "EDA",
    layout_sidebar(
      sidebar = sidebar(
        selectInput(
          "eda_plot",
          "Select EDA plot:",
          choices = c(
            "Price Level Time Series" = "eda_figures/01_price_level_time_series.png",
            "Normalized Price Series" = "eda_figures/02_normalized_price_series.png",
            "Daily Log Returns Time Series" = "eda_figures/03_daily_log_returns_time_series.png",
            "Daily Log Returns by Asset" = "eda_figures/04_daily_log_returns_by_asset.png",
            "30-Day Rolling Volatility" = "eda_figures/05_30_day_rolling_volatility.png",
            "30-Day Annualized Rolling Volatility" = "eda_figures/06_30_day_annualized_rolling_volatility.png",
            "Correlation Heatmap" = "eda_figures/08_correlation_heatmap.png",
            "Risk-Return Plot" = "eda_figures/09_risk_return_plot.png",
            "Risk-Return Bubble Plot" = "eda_figures/10_risk_return_bubble_plot.png",
            "Rolling Correlation with Bitcoin" = "eda_figures/11_rolling_correlation_with_bitcoin.png"
          )
        )
      ),
      
      div(
        class = "card-box",
        div(class = "section-title", "Exploratory Data Analysis"),
        p("The EDA section summarizes price trends, daily return behavior, rolling volatility, cross-asset correlations, and risk-return characteristics."),
        uiOutput("eda_image")
      )
    )
  ),
  
  # ============================================================
  # Feature Engineering
  # ============================================================
  
  nav_panel(
    "Feature Engineering",
    layout_sidebar(
      sidebar = sidebar(
        selectInput(
          "feature_view",
          "Select feature view:",
          choices = c(
            "Feature Matrix Preview" = "matrix",
            "Feature-Target Correlation" = "correlation",
            "Feature Groups" = "groups"
          )
        )
      ),
      
      div(
        class = "card-box",
        div(class = "section-title", "Feature Engineering"),
        
        conditionalPanel(
          condition = "input.feature_view == 'matrix'",
          p("The final feature matrix includes lagged returns, rolling means, rolling volatility, momentum, volatility ratios, rolling correlations, regime indicators, drawdowns, and extreme return flags."),
          DTOutput("feature_matrix_table")
        ),
        
        conditionalPanel(
          condition = "input.feature_view == 'correlation'",
          p("This table shows the relationship between engineered features and the target return."),
          DTOutput("feature_corr_table")
        ),
        
        conditionalPanel(
          condition = "input.feature_view == 'groups'",
          h4("Main Feature Groups"),
          tags$ul(
            tags$li("Lagged return features"),
            tags$li("Rolling mean features"),
            tags$li("Rolling volatility features"),
            tags$li("Momentum features"),
            tags$li("Volatility ratio features"),
            tags$li("Rolling correlation features"),
            tags$li("Volatility regime indicators"),
            tags$li("Drawdown features"),
            tags$li("Extreme return indicators")
          )
        )
      )
    )
  ),
  
  # ============================================================
  # Unsupervised Learning
  # ============================================================
  
  nav_panel(
    "Unsupervised Learning",
    layout_sidebar(
      sidebar = sidebar(
        selectInput(
          "unsup_table_choice",
          "Select unsupervised result:",
          choices = c(
            "PCA, Cluster, and Outlier Results" = "unsup_results",
            "Cluster Summary - Scaled" = "cluster_scaled",
            "Cluster Summary - Unscaled" = "cluster_unscaled",
            "Outlier Interpretation Results" = "outlier_results"
          )
        )
      ),
      
      div(
        class = "card-box",
        div(class = "section-title", "Unsupervised Learning Analysis"),
        p("This section summarizes PCA, K-Means clustering, and Isolation Forest outlier detection results."),
        DTOutput("unsup_table")
      )
    )
  ),
  
  # ============================================================
  # Supervised Machine Learning
  # ============================================================
  
  nav_panel(
    "Supervised Models",
    layout_sidebar(
      sidebar = sidebar(
        selectInput(
          "sup_asset",
          "Select asset:",
          choices = assets
        ),
        selectInput(
          "sup_model",
          "Select model:",
          choices = models
        )
      ),
      
      div(
        class = "card-box",
        div(class = "section-title", "Supervised Machine Learning Results"),
        
        p("Three supervised learning models were compared for next-day log return prediction: Ridge Regression, Random Forest, and Hist Gradient Boosting."),
        
        fluidRow(
          column(
            4,
            div(
              class = "metric-card",
              div(class = "metric-number", "Ridge"),
              div(class = "metric-label", "Linear benchmark with L2 regularization")
            )
          ),
          column(
            4,
            div(
              class = "metric-card",
              div(class = "metric-number", "RF"),
              div(class = "metric-label", "Nonlinear tree ensemble model")
            )
          ),
          column(
            4,
            div(
              class = "metric-card",
              div(class = "metric-number", "HGB"),
              div(class = "metric-label", "Boosted tree-based model")
            )
          )
        ),
        
        div(class = "sub-title", "Selected Return Prediction Plot"),
        div(
          class = "plot-note",
          "Return prediction plots directly show the model output because the supervised models forecast next-day log returns."
        ),
        uiOutput("supervised_return_image"),
        
        div(class = "sub-title", "Selected Price Prediction Plot"),
        div(
          class = "plot-note",
          "Price prediction plots convert predicted returns back into price paths for easier visual interpretation."
        ),
        uiOutput("supervised_price_image"),
        
        br(),
        
        div(class = "sub-title", "Best Model by Asset"),
        DTOutput("best_model_results_table"),
        
        br(),
        
        div(class = "sub-title", "All Model Comparison Results"),
        DTOutput("supervised_results_table")
      )
    )
  ),
  
  # ============================================================
  # Numerical Validation
  # ============================================================
  
  nav_panel(
    "Numerical Validation",
    layout_sidebar(
      sidebar = sidebar(
        selectInput(
          "num_asset",
          "Select asset:",
          choices = assets
        ),
        selectInput(
          "num_model",
          "Select supervised model:",
          choices = models
        )
      ),
      
      div(
        class = "card-box",
        div(class = "section-title", "Numerical Method Validation"),
        
        p("Numerical validation converts predicted returns into predicted price paths using Euler, Trapezoidal, and RK4-like updating rules."),
        
        tags$ul(
          tags$li("Euler method directly applies the one-step predicted return."),
          tags$li("Trapezoidal method uses an average update."),
          tags$li("RK4-like method uses a weighted update based on intermediate approximations.")
        ),
        
        div(class = "sub-title", "Selected Numerical Validation Plot"),
        uiOutput("numerical_image"),
        
        br(),
        
        div(class = "sub-title", "Best Numerical Validation Results"),
        DTOutput("best_numerical_table"),
        
        br(),
        
        div(class = "sub-title", "All Numerical Validation Results"),
        DTOutput("numerical_results_table")
      )
    )
  ),
  
  # ============================================================
  # Final Summary
  # ============================================================
  
  nav_panel(
    "Summary",
    div(
      class = "card-box",
      div(class = "section-title", "Final Project Summary"),
      
      p("The project shows that the six financial assets have very different return, volatility, and risk characteristics. Cryptocurrencies such as Bitcoin and Ethereum have higher volatility and more extreme return behavior, while Gold and USD Index are relatively more stable."),
      
      p("Feature engineering plays an important role in improving the predictive framework. The final feature set captures lagged returns, momentum, volatility dynamics, rolling correlations, drawdowns, and extreme market movements."),
      
      p("In the supervised learning analysis, Random Forest is the strongest overall model, achieving the best RMSE for most assets. Hist Gradient Boosting performs best for Silver. Ridge Regression is useful as a linear benchmark, but it is generally less flexible than the tree-based models."),
      
      p("Return prediction plots show the direct model outputs, while price prediction plots help interpret how those return forecasts translate into price paths."),
      
      p("The numerical validation results further support the supervised learning conclusions. Euler updating is the most reliable numerical method for reconstructing predicted price paths from next-day return forecasts."),
      
      h4("Main Conclusion"),
      p(strong("Random Forest combined with Euler numerical updating provides the most stable overall prediction framework for this financial time series project."))
    )
  )
)

# ============================================================
# 4. Server
# ============================================================

server <- function(input, output, session) {
  
  # ------------------------------------------------------------
  # Clean and Preprocessing Tables
  # ------------------------------------------------------------
  
  output$prices_table <- renderDT({
    datatable(prices, options = list(pageLength = 10, scrollX = TRUE))
  })
  
  output$returns_table <- renderDT({
    datatable(returns, options = list(pageLength = 10, scrollX = TRUE))
  })
  
  output$basic_features_table <- renderDT({
    datatable(basic_features, options = list(pageLength = 10, scrollX = TRUE))
  })
  
  output$features_table <- renderDT({
    datatable(features, options = list(pageLength = 10, scrollX = TRUE))
  })
  
  # ------------------------------------------------------------
  # EDA
  # ------------------------------------------------------------
  
  output$eda_image <- renderUI({
    safe_img(input$eda_plot)
  })
  
  # ------------------------------------------------------------
  # Feature Engineering
  # ------------------------------------------------------------
  
  output$feature_matrix_table <- renderDT({
    datatable(features, options = list(pageLength = 10, scrollX = TRUE))
  })
  
  output$feature_corr_table <- renderDT({
    datatable(feature_corr, options = list(pageLength = 10, scrollX = TRUE))
  })
  
  # ------------------------------------------------------------
  # Unsupervised Learning
  # ------------------------------------------------------------
  
  output$unsup_table <- renderDT({
    selected_table <- switch(
      input$unsup_table_choice,
      "unsup_results" = unsup_results,
      "cluster_scaled" = cluster_summary_scaled,
      "cluster_unscaled" = cluster_summary_unscaled,
      "outlier_results" = outlier_results
    )
    
    datatable(selected_table, options = list(pageLength = 10, scrollX = TRUE))
  })
  
  # ------------------------------------------------------------
  # Supervised Model Results
  # ------------------------------------------------------------
  
  output$best_model_results_table <- renderDT({
    datatable(best_model_results, options = list(pageLength = 10, scrollX = TRUE))
  })
  
  output$supervised_results_table <- renderDT({
    datatable(supervised_results, options = list(pageLength = 10, scrollX = TRUE))
  })
  
  output$supervised_price_image <- renderUI({
    asset_file <- clean_name(input$sup_asset)
    model_file <- clean_name(input$sup_model)
    
    img_path <- find_image_by_keywords(
      folder_relative = "modeling_outputs/figures",
      keywords = c(asset_file, model_file, "price", "prediction")
    )
    
    display_or_warning(img_path)
  })
  
  output$supervised_return_image <- renderUI({
    asset_file <- clean_name(input$sup_asset)
    model_file <- clean_name(input$sup_model)
    
    img_path <- find_image_by_keywords(
      folder_relative = "modeling_return_plot",
      keywords = c(asset_file, model_file, "return", "prediction")
    )
    
    if (is.null(img_path)) {
      img_path <- find_image_by_keywords(
        folder_relative = "modeling_outputs/figures",
        keywords = c(asset_file, model_file, "return", "prediction")
      )
    }
    
    display_or_warning(img_path)
  })
  
  # ------------------------------------------------------------
  # Numerical Validation
  # ------------------------------------------------------------
  
  output$best_numerical_table <- renderDT({
    datatable(best_numerical_results, options = list(pageLength = 10, scrollX = TRUE))
  })
  
  output$numerical_results_table <- renderDT({
    datatable(numerical_results, options = list(pageLength = 10, scrollX = TRUE))
  })
  
  output$numerical_image <- renderUI({
    asset_file <- clean_name(input$num_asset)
    model_file <- clean_name(input$num_model)
    
    img_path <- find_image_by_keywords(
      folder_relative = file.path(NUM_DIR, "numerical_validation_plots"),
      keywords = c(asset_file, model_file, "price", "validation")
    )
    
    display_or_warning(img_path)
  })
}

# ============================================================
# 5. Run App
# ============================================================

shinyApp(ui = ui, server = server)