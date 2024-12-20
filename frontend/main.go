package main

import (
	"github.com/johngerving/ask-alex.git/pkg/handlers"
	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
)

func main() {
	e := echo.New()

	e.Static("/static", "static")

	e.Use(middleware.Logger())

	e.GET("/chat", handlers.ChatPageHandler)
	e.GET("/chat/messages", handlers.ChatMessageHandler)

	e.Start(":8080")
}
