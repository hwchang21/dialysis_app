library(shiny)
library(e1071)
library(caret)
library(kernlab)

model_path <- "svm_model.RData"
if (file.exists(model_path)) {
  load(model_path)  # 直接加载模型
} else {
  stop(paste("The model file", model_path, "does not exist. Please check the path."))
}

if (exists("svm_model")) {
  print(class(svm_model))  # 打印模型的类型
} else {
  print("Model not loaded.")
}


# Define UI
ui <- fluidPage(
  tags$head(
    tags$style(HTML("
      body {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        background-color: #f9f9f9;
      }
      .title {
        text-align: center;
        margin-bottom: 15px;
        font-size: 20px;
        color: #333;
      }
      .input-label {
        font-weight: bold;
        color: #666;
      }
      .input-row {
        margin-bottom: 20px;
      }
      .predict-button-container {
        text-align: center;
      }
      .predict-button {
        background-color: #007bff;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin-top: 20px;
        cursor: pointer;
        border-radius: 4px;
        transition: background-color 0.3s;
      }
      .predict-button:hover {
        background-color: #0056b3;
      }
      .result {
        font-size: 20px;
        font-weight: bold;
        color: #333;
        text-align: center;
        margin-top: 20px;
      }
    "))
  ),
  div(style = "text-align: center;", # 添加样式以使标题居中
      titlePanel("Incremental Dialysis Patient Risk Prediction")
  ),
  sidebarLayout(
    sidebarPanel(
      fluidRow(
        column(8,
               div(class = "input-row",
                   tags$label(class = "input-label", "Urine output (L/24h):"),
                   numericInput("a", NULL, value = 0.5)
               ),
               div(class = "input-row",
                   tags$label(class = "input-label", "Std Kt/v:"),
                   numericInput("b", NULL, value = 2.0)
               ),
               div(class = "input-row",
                   tags$label(class = "input-label", "Ultrafiltration weight ratio (dL/kg):"),
                   numericInput("c", NULL, value = 4.0)
               ),
               div(class = "input-row",
                   tags$label(class = "input-label", "GFR (mL/min/1.73m²):"),
                   numericInput("d", NULL, value = 3.5)
               ),
               div(class = "input-row",
                   tags$label(class = "input-label", "Normalized Protein Catabolic Rate (g/kg/d):"),
                   numericInput("e", NULL, value = 1.1)
               ),
               div(class = "input-row",
                   tags$label(class = "input-label", "Albumin (g/L):"),
                   numericInput("f", NULL, value = 37.5)
               ),
               div(class = "input-row",
                   tags$label(class = "input-label", "White Blood Cell Count (×10⁹/L):"),
                   numericInput("g", NULL, value = 6.2)
               ),
               div(class = "input-row",
                   tags$label(class = "input-label", "Hemoglobin (g/L):"),
                   numericInput("h", NULL, value = 100.0)
               ),
               div(class = "input-row",
                   tags$label(class = "input-label", "Serum Phosphorus (mg/dL):"),
                   numericInput("i", NULL, value = 1.8)
               ),
               div(class = "input-row",
                   tags$label(class = "input-label", "Total Bilirubin (mg/L):"),
                   numericInput("j", NULL, value = 6.4)
               )
        ),
        div(class = "predict-button-container", # 将"Predict"按钮包装在一个新的div中并添加样式以居中
            actionButton("predictButton", "Predict", class = "predict-button")
        )
      )
    ),
    mainPanel(
      h3("Prediction Result", class = "title"),
      div(id = "prediction-container", style = "text-align: center;", # 添加样式以使输出的结论居中
          textOutput("prediction")
      )
    )
  )
)

# 定义 server 逻辑
server <- function(input, output) {
  # 定义预测函数
  predict_model <- function(model, a, b, c, d, e, f, g, h,i,j) {
    # 使用模型进行预测
    new_data <- data.frame(尿量 = a, stdktv = b, 超滤体重比 = c, GFR = d, 
                             npcr = e, 白蛋白 = f, 白细胞计数 = g, 血红蛋白 = h,
                             血清磷 = i, 总胆红素 = j)
    prediction <- predict(model, newdata = new_data, type = "prob")
    return(prediction)
  }
  
  # 监听预测按钮
  observeEvent(input$predictButton, {
    # 获取输入值并进行标准化
    a <- (input$a - 0.8893241) / 5.528386e-01 
    b <- (input$b - 1.8951531) / 4.592601e-01
    c <- (input$c - 4.058130) / 1.619472 
    d <- (input$d - 3.352847) / 2.674587 
    e <- (input$e - 1.1069633) / 0.2521451 
    f <- (input$f - 37.353471) / 4.758895 
    g <- (input$g - 6.221026) / 1.947241 
    h <- (input$h - 100.20252) / 16.27902
    i <- (input$i - 1.7760309) / 0.5334288
    j <- (input$j - 6.422994) / 3.394872
    
    # 调用预测函数并传入模型对象
    prediction <- predict_model(svm_model, a, b, c, d, e, f, g, h,i,j)
    
    # 设置阈值（假设中位数作为阈值）
    threshold <- 0.0289
    # 二元分类
    positive_class <- "Class1"  # 根据实际情况设置阳性类的名称
    risk_category <- ifelse(prediction[, positive_class] >= threshold, "High risk", "Low risk")
    
    # 显示预测结果
    output$prediction <- renderText({
      paste("Predicted risk rating:", risk_category
           # ,", risk probability:", sprintf("%.2f%%", prediction[1, "Class1"] * 100)
            )
    })
  })
}

# 运行应用程序
shinyApp(ui = ui, server = server)
