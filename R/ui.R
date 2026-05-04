# UI for the misinformation detection MVP.

library(shiny)

ui <- fluidPage(
  titlePanel("Misinformation Detection (MVP)"),

  fluidRow(
    column(
      width = 12,
      textInput(
        inputId = "claim",
        label = "Enter a claim",
        placeholder = "Type a claim to analyze"
      ),
      actionButton("analyze", "Analyze")
    )
  ),

  hr(),

  fluidRow(
    column(
      width = 12,
      h4("Verdict"),
      textOutput("verdict"),

      h4("Confidence Score"),
      textOutput("confidence"),

      h4("Explanation"),
      textOutput("explanation"),

      h4("Evidence"),
      uiOutput("evidence")
    )
  )
)
